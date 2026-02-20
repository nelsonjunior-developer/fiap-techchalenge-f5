"""Temporal shift validation for target and model-frame features."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import get_default_dataset_path, load_pede_workbook_with_metadata, make_temporal_pairs
from src.preprocessing import (
    build_pruning_plan_from_training_frame,
    get_expected_raw_feature_columns,
    transform_raw_to_model_frame,
)
from src.training_policy import OFFICIAL_HOLDOUT_PAIR, OFFICIAL_TRAIN_PAIR
from src.training_utils import build_raw_from_ids
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_FORBIDDEN_KEYS = {"ra", "ra_list", "ids", "student_ids", "students", "records"}
_TOP_CATEGORIES_LIMIT = 5
_CATEGORICAL_REFERENCE_TOP_K = 20

DEFAULT_SHIFT_THRESHOLDS: dict[str, float | int] = {
    "target_delta_abs_warning": 0.15,
    "target_delta_abs_fail": 0.25,
    "psi_warning": 0.10,
    "psi_fail": 0.25,
    "tvd_warning": 0.10,
    "tvd_fail": 0.25,
    "missing_delta_warning": 0.10,
    "missing_delta_fail": 0.20,
    "n_fail_features": 3,
    "n_warning_features": 5,
}


def _collect_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys |= _collect_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_keys(item)
    return keys


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload (expected object): {path}")
    return payload


def _resolve_dataset_path(file_path: str | Path | None) -> Path:
    if file_path is None:
        return get_default_dataset_path()
    resolved = Path(file_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path not found: {resolved}")
    return resolved


def _hash_pruning_plan(plan: dict[str, Any]) -> str:
    payload = json.dumps(plan, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_thresholds(overrides: dict[str, float | int] | None = None) -> dict[str, float | int]:
    resolved = dict(DEFAULT_SHIFT_THRESHOLDS)
    if overrides:
        resolved.update(overrides)
    return resolved


def _target_summary(y_true: pd.Series | np.ndarray) -> dict[str, float | int]:
    y = pd.Series(y_true).astype("Int64")
    n = int(len(y))
    n_pos = int((y == 1).sum())
    prevalence = float(n_pos / n) if n else 0.0
    return {
        "n": n,
        "n_pos": n_pos,
        "prevalence": prevalence,
    }


def _target_status(delta_abs: float, thresholds: dict[str, float | int]) -> str:
    abs_delta = abs(float(delta_abs))
    if abs_delta >= float(thresholds["target_delta_abs_fail"]):
        return "FAIL"
    if abs_delta >= float(thresholds["target_delta_abs_warning"]):
        return "WARNING"
    return "PASS"


def _missing_rate(series: pd.Series) -> float:
    return float(series.isna().mean()) if len(series) else 0.0


def _as_numeric(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.to_numpy(dtype=float)


def _numeric_quantiles(values: np.ndarray) -> dict[str, float | None]:
    non_null = values[~np.isnan(values)]
    if non_null.size == 0:
        return {"p05": None, "p50": None, "p95": None}
    return {
        "p05": float(np.quantile(non_null, 0.05)),
        "p50": float(np.quantile(non_null, 0.50)),
        "p95": float(np.quantile(non_null, 0.95)),
    }


def _compute_psi(train_values: np.ndarray, holdout_values: np.ndarray, bins: int = 10) -> float:
    train_non_null = train_values[~np.isnan(train_values)]
    hold_non_null = holdout_values[~np.isnan(holdout_values)]
    if train_non_null.size == 0 and hold_non_null.size == 0:
        return 0.0
    if train_non_null.size == 0 or hold_non_null.size == 0:
        return 1.0

    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    bin_edges = np.quantile(train_non_null, quantiles)
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 2:
        bin_edges = np.array([-np.inf, np.inf], dtype=float)
    else:
        bin_edges = bin_edges.astype(float)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    train_counts, _ = np.histogram(train_non_null, bins=bin_edges)
    hold_counts, _ = np.histogram(hold_non_null, bins=bin_edges)

    train_ratio = train_counts / max(train_counts.sum(), 1)
    hold_ratio = hold_counts / max(hold_counts.sum(), 1)

    eps = 1e-6
    train_ratio = np.clip(train_ratio.astype(float), eps, None)
    hold_ratio = np.clip(hold_ratio.astype(float), eps, None)
    train_ratio = train_ratio / train_ratio.sum()
    hold_ratio = hold_ratio / hold_ratio.sum()

    psi = float(np.sum((hold_ratio - train_ratio) * np.log(hold_ratio / train_ratio)))
    if not np.isfinite(psi):
        return 0.0
    return psi


def _normalize_categorical(series: pd.Series) -> pd.Series:
    normalized = pd.Series(series, copy=False).astype("string")
    normalized = normalized.fillna("_MISSING_")
    normalized = normalized.str.strip()
    normalized = normalized.replace("", "_MISSING_")
    return normalized


def _top_categories(series: pd.Series, limit: int = _TOP_CATEGORIES_LIMIT) -> list[dict[str, Any]]:
    if len(series) == 0:
        return []
    counts = series.value_counts(dropna=False, normalize=True).head(int(limit))
    return [
        {"category": str(category), "freq": float(freq)}
        for category, freq in counts.items()
    ]


def _compute_tvd(train_series: pd.Series, holdout_series: pd.Series) -> float:
    train_norm = _normalize_categorical(train_series)
    hold_norm = _normalize_categorical(holdout_series)

    reference_counts = train_norm.value_counts(dropna=False)
    keep_categories = set(
        str(item)
        for item in reference_counts.head(_CATEGORICAL_REFERENCE_TOP_K).index.tolist()
    )
    keep_categories.add("_MISSING_")

    train_collapsed = train_norm.where(train_norm.isin(keep_categories), "_OTHER_")
    hold_collapsed = hold_norm.where(hold_norm.isin(keep_categories), "_OTHER_")

    dist_train = train_collapsed.value_counts(dropna=False, normalize=True)
    dist_holdout = hold_collapsed.value_counts(dropna=False, normalize=True)
    all_categories = sorted(set(dist_train.index.tolist()) | set(dist_holdout.index.tolist()))
    diff = 0.0
    for category in all_categories:
        diff += abs(float(dist_train.get(category, 0.0)) - float(dist_holdout.get(category, 0.0)))
    return float(0.5 * diff)


def _infer_feature_kind(train_series: pd.Series, holdout_series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(train_series) and pd.api.types.is_numeric_dtype(holdout_series):
        merged = pd.concat([train_series, holdout_series], axis=0)
        numeric = pd.to_numeric(merged, errors="coerce").dropna()
        if numeric.empty:
            return "numeric"
        unique_vals = {float(value) for value in numeric.unique().tolist()}
        if unique_vals.issubset({0.0, 1.0}) and len(unique_vals) <= 2:
            return "binary"
        return "numeric"

    merged_cat = _normalize_categorical(pd.concat([train_series, holdout_series], axis=0))
    unique_count = int(merged_cat.nunique(dropna=False))
    if unique_count <= 2:
        return "binary"
    return "categorical"


def _severity_for_feature(
    *,
    dtype_kind: str,
    drift_score: float,
    delta_missing_abs: float,
    thresholds: dict[str, float | int],
) -> str:
    if dtype_kind == "numeric":
        warning_threshold = float(thresholds["psi_warning"])
        fail_threshold = float(thresholds["psi_fail"])
    else:
        warning_threshold = float(thresholds["tvd_warning"])
        fail_threshold = float(thresholds["tvd_fail"])

    missing_warning = float(thresholds["missing_delta_warning"])
    missing_fail = float(thresholds["missing_delta_fail"])

    if drift_score >= fail_threshold or delta_missing_abs >= missing_fail:
        return "FAIL"
    if drift_score >= warning_threshold or delta_missing_abs >= missing_warning:
        return "WARNING"
    return "PASS"


def build_model_frame_for_year_pair(
    dfs: dict[int, pd.DataFrame],
    year_t: int,
    year_t1: int,
    *,
    expected_raw_cols: list[str],
    enable_feature_engineering: bool,
    enable_age_bucket: bool,
    feature_pruning_plan: dict[str, Any],
    kept_model_cols: list[str],
    strict_raw: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Build MODEL frame for a temporal pair using the same train/inference path."""
    if year_t not in dfs or year_t1 not in dfs:
        raise ValueError(f"Year pair {year_t}->{year_t1} unavailable in provided frames.")
    if not kept_model_cols:
        raise ValueError("kept_model_cols must not be empty.")

    _, y_target, ids = make_temporal_pairs(
        dfs[year_t],
        dfs[year_t1],
        year_t,
        year_t1,
    )
    X_raw = build_raw_from_ids(
        df_year_t=dfs[year_t],
        ids=ids,
        expected_raw_cols=expected_raw_cols,
    )
    X_model = transform_raw_to_model_frame(
        X_raw=X_raw,
        year_t=year_t,
        expected_raw_cols=list(expected_raw_cols),
        expected_model_cols=list(kept_model_cols),
        enable_feature_engineering=bool(enable_feature_engineering),
        enable_age_bucket=bool(enable_age_bucket),
        feature_pruning_plan=dict(feature_pruning_plan),
        strict_raw=bool(strict_raw),
        include_report=False,
        context=f"temporal_shift_{year_t}_{year_t1}",
    )
    meta = _target_summary(y_target)
    meta["pair"] = f"{year_t}->{year_t1}"
    return X_model, y_target, meta


def compute_target_shift(y_train: pd.Series | np.ndarray, y_holdout: pd.Series | np.ndarray) -> dict[str, Any]:
    """Compute target prevalence temporal shift between train and holdout."""
    train_summary = _target_summary(y_train)
    holdout_summary = _target_summary(y_holdout)
    train_prev = float(train_summary["prevalence"])
    holdout_prev = float(holdout_summary["prevalence"])
    delta_abs = float(holdout_prev - train_prev)
    delta_rel = None if np.isclose(train_prev, 0.0) else float((holdout_prev / train_prev) - 1.0)
    return {
        "train": train_summary,
        "holdout": holdout_summary,
        "delta_prevalence_abs": delta_abs,
        "delta_prevalence_rel": delta_rel,
    }


def compute_feature_shift(
    X_train_model: pd.DataFrame,
    X_holdout_model: pd.DataFrame,
    thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """Compute feature-level temporal shift on MODEL frame columns."""
    resolved_thresholds = _resolve_thresholds(thresholds)
    train_cols = list(X_train_model.columns)
    holdout_cols = list(X_holdout_model.columns)
    if train_cols != holdout_cols:
        missing_in_holdout = sorted(set(train_cols) - set(holdout_cols))
        extra_in_holdout = sorted(set(holdout_cols) - set(train_cols))
        raise ValueError(
            "Train/holdout MODEL columns mismatch. "
            f"missing_in_holdout={missing_in_holdout} extra_in_holdout={extra_in_holdout}"
        )

    feature_entries: list[dict[str, Any]] = []
    n_numeric = 0
    n_categorical = 0
    n_binary = 0

    for feature in train_cols:
        train_series = X_train_model[feature]
        holdout_series = X_holdout_model[feature]
        dtype_kind = _infer_feature_kind(train_series, holdout_series)
        if dtype_kind == "numeric":
            n_numeric += 1
            train_values = _as_numeric(train_series)
            holdout_values = _as_numeric(holdout_series)
            drift_score = _compute_psi(train_values, holdout_values, bins=10)
            mean_train = float(np.nanmean(train_values)) if np.any(~np.isnan(train_values)) else None
            mean_holdout = (
                float(np.nanmean(holdout_values)) if np.any(~np.isnan(holdout_values)) else None
            )
            std_train = float(np.nanstd(train_values)) if np.any(~np.isnan(train_values)) else None
            std_holdout = (
                float(np.nanstd(holdout_values)) if np.any(~np.isnan(holdout_values)) else None
            )
            entry: dict[str, Any] = {
                "feature": str(feature),
                "dtype_kind": "numeric",
                "missing_rate_train": _missing_rate(train_series),
                "missing_rate_holdout": _missing_rate(holdout_series),
                "delta_missing_abs": abs(_missing_rate(holdout_series) - _missing_rate(train_series)),
                "mean_train": mean_train,
                "mean_holdout": mean_holdout,
                "std_train": std_train,
                "std_holdout": std_holdout,
                "quantiles_train": _numeric_quantiles(train_values),
                "quantiles_holdout": _numeric_quantiles(holdout_values),
                "drift_score_numeric": float(drift_score),
                "drift_score": float(drift_score),
            }
        else:
            if dtype_kind == "binary":
                n_binary += 1
            else:
                n_categorical += 1
            drift_score = _compute_tvd(train_series, holdout_series)
            train_norm = _normalize_categorical(train_series)
            holdout_norm = _normalize_categorical(holdout_series)
            entry = {
                "feature": str(feature),
                "dtype_kind": dtype_kind,
                "missing_rate_train": _missing_rate(train_series),
                "missing_rate_holdout": _missing_rate(holdout_series),
                "delta_missing_abs": abs(_missing_rate(holdout_series) - _missing_rate(train_series)),
                "top_categories_train": _top_categories(train_norm),
                "top_categories_holdout": _top_categories(holdout_norm),
                "drift_score_cat": float(drift_score),
                "drift_score": float(drift_score),
            }

        entry["severity"] = _severity_for_feature(
            dtype_kind=str(entry["dtype_kind"]),
            drift_score=float(entry["drift_score"]),
            delta_missing_abs=float(entry["delta_missing_abs"]),
            thresholds=resolved_thresholds,
        )
        feature_entries.append(entry)

    counts = {"pass": 0, "warning": 0, "fail": 0}
    for entry in feature_entries:
        severity = str(entry["severity"]).lower()
        if severity == "fail":
            counts["fail"] += 1
        elif severity == "warning":
            counts["warning"] += 1
        else:
            counts["pass"] += 1

    worst = sorted(
        feature_entries,
        key=lambda item: (
            -float(item["drift_score"]),
            -float(item["delta_missing_abs"]),
            str(item["feature"]),
        ),
    )[:10]
    worst_payload = [
        {
            "feature": item["feature"],
            "dtype_kind": item["dtype_kind"],
            "drift_score": item["drift_score"],
            "severity": item["severity"],
        }
        for item in worst
    ]

    return {
        "n_features": int(len(feature_entries)),
        "n_numeric": int(n_numeric),
        "n_categorical": int(n_categorical),
        "n_binary": int(n_binary),
        "counts_by_severity": counts,
        "worst_features_by_drift": worst_payload,
        "features": feature_entries,
    }


def aggregate_shift_status(
    target_shift: dict[str, Any],
    feature_shift_summary: dict[str, Any],
    thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """Aggregate target and feature shifts into PASS/WARNING/FAIL status."""
    resolved_thresholds = _resolve_thresholds(thresholds)
    delta_abs = float(target_shift.get("delta_prevalence_abs", 0.0))
    target_status = _target_status(delta_abs, resolved_thresholds)

    counts = feature_shift_summary.get("counts_by_severity", {})
    n_fail_features = int(counts.get("fail", 0))
    n_warning_features = int(counts.get("warning", 0))

    max_fail = int(resolved_thresholds["n_fail_features"])
    max_warning = int(resolved_thresholds["n_warning_features"])

    reasons: list[str] = []
    status = "PASS"
    if target_status == "FAIL":
        reasons.append("target_shift_failed")
    if n_fail_features >= max_fail:
        reasons.append("feature_shift_failed_by_fail_count")
    if reasons:
        status = "FAIL"
    else:
        warning_reasons: list[str] = []
        if target_status == "WARNING":
            warning_reasons.append("target_shift_warning")
        if n_fail_features > 0:
            warning_reasons.append("feature_shift_contains_fail_features_below_fail_gate")
        if n_warning_features >= max_warning:
            warning_reasons.append("feature_shift_warning_by_warning_count")
        if warning_reasons:
            status = "WARNING"
            reasons.extend(warning_reasons)

    return {
        "status": status,
        "target_status": target_status,
        "n_fail_features": n_fail_features,
        "n_warning_features": n_warning_features,
        "n_pass_features": int(counts.get("pass", 0)),
        "rules": {
            "global_fail_if_target_fail": True,
            "global_fail_if_n_fail_features_gte": max_fail,
            "global_warning_if_target_warning": True,
            "global_warning_if_n_warning_features_gte": max_warning,
            "warning_is_non_blocking": True,
        },
        "reasons": reasons,
    }


def _resolve_config(
    *,
    config: str,
    models_root: Path,
    selection_path: Path,
) -> dict[str, Any]:
    config_mode = str(config).strip().lower()
    if config_mode not in {"winner", "baseline", "hgb"}:
        raise ValueError(f"Unsupported config mode: {config_mode}")

    default_map: dict[str, dict[str, Any]] = {
        "baseline": {
            "model_family": "baseline_logreg",
            "variant": "none",
            "enable_feature_engineering": True,
            "enable_age_bucket": True,
        },
        "hgb": {
            "model_family": "nonlinear_hgb",
            "variant": "default",
            "enable_feature_engineering": True,
            "enable_age_bucket": False,
        },
    }

    warnings: list[str] = []
    if config_mode == "winner":
        selection = _safe_read_json(selection_path)
        winner = selection.get("winner")
        if not isinstance(winner, dict):
            raise ValueError("Invalid selection payload: winner block missing.")
        model_family = str(winner.get("model_family") or "").strip()
        variant = str(winner.get("variant") or "").strip()
        if not model_family or not variant:
            raise ValueError("Invalid selection payload: winner model_family/variant missing.")
        metadata_path_raw = str(winner.get("path_metadata") or "").strip()
        if metadata_path_raw:
            metadata_path = Path(metadata_path_raw)
        else:
            metadata_path = models_root / model_family / variant / "metadata.json"
    else:
        defaults = default_map[config_mode]
        model_family = str(defaults["model_family"])
        variant = str(defaults["variant"])
        metadata_path = models_root / model_family / variant / "metadata.json"

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = _safe_read_json(metadata_path)
    else:
        warnings.append(f"metadata not found for config={config_mode}: {metadata_path}")

    defaults = default_map.get(
        config_mode,
        {
            "enable_feature_engineering": True,
            "enable_age_bucket": False,
        },
    )
    enable_feature_engineering = bool(
        metadata.get("enable_feature_engineering", defaults["enable_feature_engineering"])
    )
    enable_age_bucket = bool(metadata.get("enable_age_bucket", defaults["enable_age_bucket"]))
    feature_pruning_plan_hash = metadata.get("feature_pruning_plan_hash")
    train_pair = metadata.get("train_pair")
    if not isinstance(train_pair, dict):
        train_pair = {
            "year_t": int(OFFICIAL_TRAIN_PAIR[0]),
            "year_t1": int(OFFICIAL_TRAIN_PAIR[1]),
        }

    return {
        "mode": config_mode,
        "model_family": model_family,
        "variant": variant,
        "metadata_path": str(metadata_path),
        "enable_feature_engineering": enable_feature_engineering,
        "enable_age_bucket": enable_age_bucket,
        "feature_pruning_plan_hash_expected": feature_pruning_plan_hash,
        "train_pair": train_pair,
        "warnings": warnings,
    }


def _resolve_feature_pruning_plan(
    *,
    dfs: dict[int, pd.DataFrame],
    expected_raw_cols: list[str],
    enable_feature_engineering: bool,
    enable_age_bucket: bool,
    expected_hash: str | None,
) -> tuple[dict[str, Any], list[str], str]:
    train_year_t, train_year_t1 = OFFICIAL_TRAIN_PAIR
    _, _, train_ids = make_temporal_pairs(
        dfs[train_year_t],
        dfs[train_year_t1],
        train_year_t,
        train_year_t1,
    )
    X_raw_train = build_raw_from_ids(
        df_year_t=dfs[train_year_t],
        ids=train_ids,
        expected_raw_cols=expected_raw_cols,
    )
    plan = build_pruning_plan_from_training_frame(
        X_train_raw=X_raw_train,
        enable_feature_engineering=bool(enable_feature_engineering),
        enable_age_bucket=bool(enable_age_bucket),
    )
    kept_model_cols = list(plan.get("kept_model_cols", []))
    if not kept_model_cols:
        raise ValueError("Feature pruning plan produced empty kept_model_cols.")
    resolved_hash = _hash_pruning_plan(plan)

    expected = str(expected_hash).strip() if expected_hash is not None else ""
    if expected and expected != resolved_hash:
        raise ValueError(
            "feature_pruning_plan_hash mismatch between metadata and resolved plan: "
            f"expected={expected} resolved={resolved_hash}"
        )
    return plan, kept_model_cols, resolved_hash


def _build_shift_report(
    *,
    config_info: dict[str, Any],
    thresholds: dict[str, float | int],
    target_shift: dict[str, Any],
    feature_shift: dict[str, Any],
    train_meta: dict[str, Any],
    holdout_meta: dict[str, Any],
    kept_model_cols: list[str],
    pruning_hash: str,
    notes: list[str],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    status_summary = aggregate_shift_status(
        target_shift=target_shift,
        feature_shift_summary=feature_shift,
        thresholds=thresholds,
    )
    status = str(status_summary["status"])
    if errors:
        status = "FAIL"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "config_mode": config_info.get("mode"),
        "configuration": {
            "model_family": config_info.get("model_family"),
            "variant": config_info.get("variant"),
            "metadata_path": config_info.get("metadata_path"),
            "train_pair": f"{OFFICIAL_TRAIN_PAIR[0]}->{OFFICIAL_TRAIN_PAIR[1]}",
            "holdout_pair": f"{OFFICIAL_HOLDOUT_PAIR[0]}->{OFFICIAL_HOLDOUT_PAIR[1]}",
            "enable_feature_engineering": bool(config_info.get("enable_feature_engineering")),
            "enable_age_bucket": bool(config_info.get("enable_age_bucket")),
            "feature_pruning_plan_hash_expected": config_info.get("feature_pruning_plan_hash_expected"),
            "feature_pruning_plan_hash_resolved": pruning_hash,
            "kept_model_cols_count": int(len(kept_model_cols)),
            "kept_model_cols": list(kept_model_cols),
        },
        "thresholds": dict(thresholds),
        "frames": {
            "train": dict(train_meta),
            "holdout": dict(holdout_meta),
        },
        "target_shift": dict(target_shift),
        "feature_shift": dict(feature_shift),
        "status_summary": status_summary,
        "notes": list(dict.fromkeys(notes)),
        "warnings": list(dict.fromkeys(warnings)),
        "errors": list(dict.fromkeys(errors)),
    }

    keys_found = _collect_keys(report)
    forbidden = sorted(_FORBIDDEN_KEYS & keys_found)
    if forbidden:
        report["status"] = "FAIL"
        report["errors"].append(
            f"Privacy check failed: forbidden keys found in report: {forbidden}"
        )
    return report


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    """Write optional markdown summary report for human review."""
    lines: list[str] = []
    lines.append("# Temporal Shift Report")
    lines.append("")
    lines.append(f"Status: **{report.get('status', 'UNKNOWN')}**")
    lines.append("")

    config = report.get("configuration", {})
    if isinstance(config, dict):
        lines.append("## Configuration")
        lines.append("")
        lines.append(
            "- mode=`{}` model=`{}/{}`".format(
                report.get("config_mode", "-"),
                config.get("model_family", "-"),
                config.get("variant", "-"),
            )
        )
        lines.append(
            "- MODEL frame: enable_feature_engineering=`{}` enable_age_bucket=`{}`".format(
                config.get("enable_feature_engineering"),
                config.get("enable_age_bucket"),
            )
        )
        lines.append(
            "- kept_model_cols_count=`{}`".format(config.get("kept_model_cols_count", "-"))
        )

    target_shift = report.get("target_shift", {})
    if isinstance(target_shift, dict):
        train = target_shift.get("train", {})
        holdout = target_shift.get("holdout", {})
        lines.append("")
        lines.append("## Target Shift")
        lines.append("")
        lines.append("| Slice | n | n_pos | prevalence |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            "| Train (2022->2023) | {n} | {n_pos} | {prev:.4f} |".format(
                n=int((train or {}).get("n", 0)),
                n_pos=int((train or {}).get("n_pos", 0)),
                prev=float((train or {}).get("prevalence", 0.0)),
            )
        )
        lines.append(
            "| Holdout (2023->2024) | {n} | {n_pos} | {prev:.4f} |".format(
                n=int((holdout or {}).get("n", 0)),
                n_pos=int((holdout or {}).get("n_pos", 0)),
                prev=float((holdout or {}).get("prevalence", 0.0)),
            )
        )
        lines.append("")
        lines.append(
            "- delta_prevalence_abs = `{:.4f}`".format(
                float(target_shift.get("delta_prevalence_abs", 0.0))
            )
        )
        delta_rel = target_shift.get("delta_prevalence_rel")
        if delta_rel is None:
            lines.append("- delta_prevalence_rel = `null`")
        else:
            lines.append("- delta_prevalence_rel = `{:.4f}`".format(float(delta_rel)))

    feature_shift = report.get("feature_shift", {})
    if isinstance(feature_shift, dict):
        counts = feature_shift.get("counts_by_severity", {})
        lines.append("")
        lines.append("## Feature Shift Summary")
        lines.append("")
        lines.append(
            "- n_features={} n_numeric={} n_categorical={} n_binary={}".format(
                int(feature_shift.get("n_features", 0)),
                int(feature_shift.get("n_numeric", 0)),
                int(feature_shift.get("n_categorical", 0)),
                int(feature_shift.get("n_binary", 0)),
            )
        )
        lines.append(
            "- severity counts: PASS={} WARNING={} FAIL={}".format(
                int((counts or {}).get("pass", 0)),
                int((counts or {}).get("warning", 0)),
                int((counts or {}).get("fail", 0)),
            )
        )

        worst = feature_shift.get("worst_features_by_drift", [])
        if isinstance(worst, list) and worst:
            lines.append("")
            lines.append("## Top Drift Features")
            lines.append("")
            lines.append("| Feature | Type | Drift score | Severity |")
            lines.append("|---|---|---:|---|")
            for row in worst:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "| {feature} | {dtype} | {score:.4f} | {severity} |".format(
                        feature=row.get("feature", "-"),
                        dtype=row.get("dtype_kind", "-"),
                        score=float(row.get("drift_score", 0.0)),
                        severity=row.get("severity", "-"),
                    )
                )

    status_summary = report.get("status_summary", {})
    if isinstance(status_summary, dict):
        lines.append("")
        lines.append("## Status Rules")
        lines.append("")
        lines.append(
            "- target_status=`{}` global_status=`{}`".format(
                status_summary.get("target_status", "-"),
                status_summary.get("status", "-"),
            )
        )
        reasons = status_summary.get("reasons", [])
        if isinstance(reasons, list) and reasons:
            lines.append("- reasons: {}".format(", ".join(str(item) for item in reasons)))
        lines.append("- WARNING is non-blocking by policy (governance visibility).")

    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")

    errors = report.get("errors", [])
    if isinstance(errors, list) and errors:
        lines.append("")
        lines.append("## Errors")
        for error in errors:
            lines.append(f"- {error}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_temporal_shift(
    *,
    file_path: str | Path | None = None,
    models_root: str | Path = "artifacts/models",
    selection_path: str | Path = "artifacts/model_selection.json",
    out_json: str | Path = "artifacts/temporal_shift_report.json",
    out_md: str | Path = "artifacts/temporal_shift_report.md",
    write_markdown: bool = True,
    strict: bool = False,
    config: str = "winner",
) -> dict[str, Any]:
    resolved_dataset_path = _resolve_dataset_path(file_path)
    dfs, _, _ = load_pede_workbook_with_metadata(file_path=resolved_dataset_path)
    models_root_path = Path(models_root)
    selection_path_obj = Path(selection_path)
    thresholds = _resolve_thresholds()

    notes: list[str] = [
        "Official gate is evaluated on MODEL frame (post feature engineering and pruning).",
        "WARNING is non-blocking and intended for governance visibility.",
    ]
    warnings: list[str] = []
    errors: list[str] = []

    config_info = _resolve_config(
        config=config,
        models_root=models_root_path,
        selection_path=selection_path_obj,
    )
    warnings.extend(config_info.get("warnings", []))

    expected_raw_cols = list(get_expected_raw_feature_columns())
    try:
        pruning_plan, kept_model_cols, pruning_hash = _resolve_feature_pruning_plan(
            dfs=dfs,
            expected_raw_cols=expected_raw_cols,
            enable_feature_engineering=bool(config_info["enable_feature_engineering"]),
            enable_age_bucket=bool(config_info["enable_age_bucket"]),
            expected_hash=(
                str(config_info.get("feature_pruning_plan_hash_expected")).strip()
                if config_info.get("feature_pruning_plan_hash_expected") is not None
                else None
            ),
        )
    except Exception as exc:
        errors.append(str(exc))
        report = _build_shift_report(
            config_info=config_info,
            thresholds=thresholds,
            target_shift={
                "train": {"n": 0, "n_pos": 0, "prevalence": 0.0},
                "holdout": {"n": 0, "n_pos": 0, "prevalence": 0.0},
                "delta_prevalence_abs": 0.0,
                "delta_prevalence_rel": None,
            },
            feature_shift={
                "n_features": 0,
                "n_numeric": 0,
                "n_categorical": 0,
                "n_binary": 0,
                "counts_by_severity": {"pass": 0, "warning": 0, "fail": 0},
                "worst_features_by_drift": [],
                "features": [],
            },
            train_meta={"pair": f"{OFFICIAL_TRAIN_PAIR[0]}->{OFFICIAL_TRAIN_PAIR[1]}", "n": 0, "n_pos": 0, "prevalence": 0.0},
            holdout_meta={"pair": f"{OFFICIAL_HOLDOUT_PAIR[0]}->{OFFICIAL_HOLDOUT_PAIR[1]}", "n": 0, "n_pos": 0, "prevalence": 0.0},
            kept_model_cols=[],
            pruning_hash="",
            notes=notes,
            warnings=warnings,
            errors=errors,
        )
    else:
        X_train_model, y_train, train_meta = build_model_frame_for_year_pair(
            dfs=dfs,
            year_t=int(OFFICIAL_TRAIN_PAIR[0]),
            year_t1=int(OFFICIAL_TRAIN_PAIR[1]),
            expected_raw_cols=expected_raw_cols,
            enable_feature_engineering=bool(config_info["enable_feature_engineering"]),
            enable_age_bucket=bool(config_info["enable_age_bucket"]),
            feature_pruning_plan=pruning_plan,
            kept_model_cols=kept_model_cols,
            strict_raw=bool(strict),
        )
        X_holdout_model, y_holdout, holdout_meta = build_model_frame_for_year_pair(
            dfs=dfs,
            year_t=int(OFFICIAL_HOLDOUT_PAIR[0]),
            year_t1=int(OFFICIAL_HOLDOUT_PAIR[1]),
            expected_raw_cols=expected_raw_cols,
            enable_feature_engineering=bool(config_info["enable_feature_engineering"]),
            enable_age_bucket=bool(config_info["enable_age_bucket"]),
            feature_pruning_plan=pruning_plan,
            kept_model_cols=kept_model_cols,
            strict_raw=bool(strict),
        )
        target_shift = compute_target_shift(y_train, y_holdout)
        feature_shift = compute_feature_shift(
            X_train_model,
            X_holdout_model,
            thresholds=thresholds,
        )
        report = _build_shift_report(
            config_info=config_info,
            thresholds=thresholds,
            target_shift=target_shift,
            feature_shift=feature_shift,
            train_meta=train_meta,
            holdout_meta=holdout_meta,
            kept_model_cols=kept_model_cols,
            pruning_hash=pruning_hash,
            notes=notes,
            warnings=warnings,
            errors=errors,
        )

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if write_markdown:
        write_markdown_report(report, Path(out_md))

    if strict and str(report.get("status", "")).upper() == "FAIL":
        raise SystemExit(1)
    return report


def _parse_bool_flag(value: int) -> bool:
    return bool(int(value))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute temporal shift report (target + model-frame features)."
    )
    parser.add_argument(
        "--file-path",
        "--dataset-path",
        dest="file_path",
        type=str,
        default=None,
        help="Path to XLSX dataset. Defaults to DATASET_PATH or project fallback.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="artifacts/models",
        help="Models root directory.",
    )
    parser.add_argument(
        "--selection-path",
        type=str,
        default="artifacts/model_selection.json",
        help="Path to model_selection.json (required when --config=winner).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="artifacts/temporal_shift_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="artifacts/temporal_shift_report.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable Markdown output.",
    )
    parser.add_argument(
        "--strict",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, return non-zero exit when report status is FAIL.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="winner",
        choices=["winner", "baseline", "hgb"],
        help="Configuration to evaluate shift on MODEL frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    try:
        report = run_temporal_shift(
            file_path=args.file_path,
            models_root=args.models_root,
            selection_path=args.selection_path,
            out_json=args.out_json,
            out_md=args.out_md,
            write_markdown=not bool(args.no_markdown),
            strict=_parse_bool_flag(args.strict),
            config=args.config,
        )
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("%s", exc)
        raise SystemExit(1) from exc
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive branch
        _logger.error("Unexpected temporal shift failure: %s", exc)
        raise SystemExit(1) from exc

    _logger.info(
        "Temporal shift report generated | status=%s config=%s out_json=%s",
        report.get("status"),
        report.get("config_mode"),
        args.out_json,
    )


if __name__ == "__main__":
    main()
