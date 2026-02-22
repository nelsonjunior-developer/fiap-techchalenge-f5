"""Promote the selected champion model artifact to a fixed serving path."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.dataset_versioning import safe_path_hint
from src.features import get_engineered_feature_names
from src.metadata_schema import validate_metadata
from src.privacy import find_forbidden_json_keys
from src.promotion_policy import promotion_decision
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload (expected object): {path}")
    return payload


def _safe_rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool_flag(value: int) -> bool:
    return bool(int(value))


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_winner_paths(
    *,
    selection_payload: dict[str, Any],
    models_root: Path,
) -> dict[str, Any]:
    winner = selection_payload.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("Selection artifact missing winner block.")

    model_family = str(winner.get("model_family") or "").strip()
    variant = str(winner.get("variant") or "").strip()
    if not model_family or not variant:
        raise ValueError("Selection winner must include model_family and variant.")

    src_model_raw = str(winner.get("path_model") or "").strip()
    src_meta_raw = str(winner.get("path_metadata") or "").strip()
    src_model = (
        Path(src_model_raw)
        if src_model_raw
        else models_root / model_family / variant / "model.joblib"
    )
    src_meta = (
        Path(src_meta_raw)
        if src_meta_raw
        else models_root / model_family / variant / "metadata.json"
    )

    if not src_model.exists():
        raise FileNotFoundError(f"Source model.joblib not found: {src_model}")
    if not src_meta.exists():
        raise FileNotFoundError(f"Source metadata.json not found: {src_meta}")

    return {
        "winner": {
            "model_family": model_family,
            "variant": variant,
        },
        "src_model": src_model,
        "src_meta": src_meta,
    }


def _resolve_staging_paths(staging_dir: Path) -> dict[str, Any]:
    src_model = staging_dir / "model.joblib"
    src_meta = staging_dir / "metadata.json"
    if not src_model.exists():
        raise FileNotFoundError(f"Staging model.joblib not found: {src_model}")
    if not src_meta.exists():
        raise FileNotFoundError(f"Staging metadata.json not found: {src_meta}")

    metadata = _safe_read_json(src_meta)
    return {
        "winner": {
            "model_family": str(metadata.get("model_family") or "unknown"),
            "variant": str(metadata.get("variant") or "unknown"),
        },
        "src_model": src_model,
        "src_meta": src_meta,
        "metadata": metadata,
    }


def _backup_existing_destination(
    *,
    out_dir: Path,
    dest_model: Path,
    dest_meta: Path,
) -> Path | None:
    has_existing = dest_model.exists() or dest_meta.exists()
    if not has_existing:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = out_dir / "backups" / timestamp
    suffix = 1
    while backup_dir.exists():
        backup_dir = out_dir / "backups" / f"{timestamp}_{suffix:02d}"
        suffix += 1
    backup_dir.mkdir(parents=True, exist_ok=False)

    if dest_model.exists():
        shutil.copy2(dest_model, backup_dir / "model.joblib")
    if dest_meta.exists():
        shutil.copy2(dest_meta, backup_dir / "metadata.json")
    return backup_dir


def _build_versions_block(metadata: dict[str, Any]) -> dict[str, Any]:
    versions_raw = metadata.get("versions")
    versions: dict[str, Any] = dict(versions_raw) if isinstance(versions_raw, dict) else {}

    sklearn_version = versions.get("scikit_learn")
    if sklearn_version is None:
        sklearn_version = versions.get("sklearn")
    if sklearn_version is None:
        try:
            import sklearn

            sklearn_version = sklearn.__version__
        except ModuleNotFoundError:
            sklearn_version = None

    joblib_version = versions.get("joblib")
    if joblib_version is None:
        try:
            import joblib

            joblib_version = joblib.__version__
        except ModuleNotFoundError:
            joblib_version = None

    return {
        "python": str(versions.get("python") or sys.version.split(" ")[0]),
        "pandas": str(versions.get("pandas") or pd.__version__),
        "numpy": str(versions.get("numpy") or np.__version__),
        "scikit_learn": None if sklearn_version is None else str(sklearn_version),
        "joblib": None if joblib_version is None else str(joblib_version),
        # Backward compatibility key kept for older tooling.
        "sklearn": None if sklearn_version is None else str(sklearn_version),
    }


def _normalize_eval_block(
    *,
    raw: dict[str, Any] | None,
    threshold_default: float,
    fallback_confusion_key: str,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    metrics_raw = raw.get("metrics")
    if not isinstance(metrics_raw, dict):
        return None

    cm_raw = raw.get("confusion_matrix")
    if not isinstance(cm_raw, dict):
        cm_raw = raw.get(fallback_confusion_key)
    if not isinstance(cm_raw, dict):
        cm_raw = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

    positive_rate = metrics_raw.get("positive_rate")
    if positive_rate is None:
        positive_rate = metrics_raw.get("positive_rate_at_threshold")

    threshold_value = _to_float_or_none(raw.get("threshold"))
    if threshold_value is None:
        threshold_value = float(threshold_default)

    block = {
        "threshold": float(threshold_value),
        "metrics": {
            "recall": _to_float_or_none(metrics_raw.get("recall")),
            "precision": _to_float_or_none(metrics_raw.get("precision")),
            "f1": _to_float_or_none(metrics_raw.get("f1")),
            "roc_auc": _to_float_or_none(metrics_raw.get("roc_auc")),
            "pr_auc": _to_float_or_none(metrics_raw.get("pr_auc")),
            "positive_rate": _to_float_or_none(positive_rate),
        },
        "confusion_matrix": {
            "tn": _to_int(cm_raw.get("tn"), 0),
            "fp": _to_int(cm_raw.get("fp"), 0),
            "fn": _to_int(cm_raw.get("fn"), 0),
            "tp": _to_int(cm_raw.get("tp"), 0),
        },
        "pred_proba_summary": (
            dict(raw.get("pred_proba_summary"))
            if isinstance(raw.get("pred_proba_summary"), dict)
            else None
        ),
        "notes": list(raw.get("notes", [])) if isinstance(raw.get("notes"), list) else [],
    }
    return block


def _resolve_eval_blocks(metadata: dict[str, Any]) -> dict[str, Any]:
    train_05 = _normalize_eval_block(
        raw=(
            metadata.get("evaluation_train_at_0.5")
            if isinstance(metadata.get("evaluation_train_at_0.5"), dict)
            else metadata.get("evaluation_train")
        ),
        threshold_default=0.5,
        fallback_confusion_key="confusion_matrix_at_0.5",
    )
    train_030 = _normalize_eval_block(
        raw=metadata.get("evaluation_train_at_0.30")
        if isinstance(metadata.get("evaluation_train_at_0.30"), dict)
        else None,
        threshold_default=0.30,
        fallback_confusion_key="confusion_matrix_at_0.30",
    )
    if train_030 is None and isinstance(train_05, dict):
        train_030 = dict(train_05)
        train_030["threshold"] = 0.30
        train_030["notes"] = list(train_030.get("notes", [])) + [
            "fallback_from_train_0.5_block",
        ]

    holdout_05 = _normalize_eval_block(
        raw=(
            metadata.get("evaluation_holdout_at_0.5")
            if isinstance(metadata.get("evaluation_holdout_at_0.5"), dict)
            else metadata.get("evaluation_holdout")
        ),
        threshold_default=0.5,
        fallback_confusion_key="confusion_matrix_at_0.5",
    )
    holdout_030 = _normalize_eval_block(
        raw=metadata.get("evaluation_holdout_at_0.30")
        if isinstance(metadata.get("evaluation_holdout_at_0.30"), dict)
        else None,
        threshold_default=0.30,
        fallback_confusion_key="confusion_matrix_at_0.30",
    )
    if holdout_030 is None and isinstance(holdout_05, dict):
        holdout_030 = dict(holdout_05)
        holdout_030["threshold"] = 0.30
        holdout_030["notes"] = list(holdout_030.get("notes", [])) + [
            "fallback_from_holdout_0.5_block",
        ]

    holdout_calibrated = _normalize_eval_block(
        raw=(
            metadata.get("evaluation_holdout_at_calibrated_threshold")
            if isinstance(metadata.get("evaluation_holdout_at_calibrated_threshold"), dict)
            else metadata.get("evaluation_holdout_at_threshold_selected")
        ),
        threshold_default=_to_float_or_none(
            metadata.get("threshold_calibration", {}).get("threshold_selected")
            if isinstance(metadata.get("threshold_calibration"), dict)
            else None
        )
        or 0.30,
        fallback_confusion_key="confusion_matrix_at_threshold_selected",
    )

    return {
        "evaluation_train_at_0.5": train_05,
        "evaluation_train_at_0.30": train_030,
        "evaluation_holdout_at_0.5": holdout_05,
        "evaluation_holdout_at_0.30": holdout_030,
        "evaluation_holdout_at_calibrated_threshold": holdout_calibrated,
    }


def _resolve_threshold_policy(metadata: dict[str, Any]) -> dict[str, Any]:
    policy_raw = metadata.get("threshold_policy")
    policy = dict(policy_raw) if isinstance(policy_raw, dict) else {}
    threshold_calibration = (
        metadata.get("threshold_calibration")
        if isinstance(metadata.get("threshold_calibration"), dict)
        else {}
    )

    operational_threshold = _to_float_or_none(policy.get("operational_fixed_threshold"))
    if operational_threshold is None:
        operational_threshold = _to_float_or_none(
            policy.get("operational", {}).get("threshold")
            if isinstance(policy.get("operational"), dict)
            else None
        )
    if operational_threshold is None:
        operational_threshold = 0.30

    topk_fraction = _to_float_or_none(policy.get("topk_fallback_fraction"))
    if topk_fraction is None:
        topk_fraction = _to_float_or_none(
            policy.get("capacity_fallback", {}).get("topk_fraction")
            if isinstance(policy.get("capacity_fallback"), dict)
            else None
        )
    if topk_fraction is None:
        topk_fraction = 0.20

    recall_target = _to_float_or_none(policy.get("recall_target_for_calibration"))
    if recall_target is None:
        recall_target = _to_float_or_none(threshold_calibration.get("recall_target"))
    if recall_target is None:
        recall_target = 0.90

    calibrated_threshold = _to_float_or_none(policy.get("calibrated_threshold"))
    if calibrated_threshold is None:
        calibrated_threshold = _to_float_or_none(
            threshold_calibration.get("threshold_selected")
        )

    notes = list(policy.get("notes", [])) if isinstance(policy.get("notes"), list) else []
    if not notes:
        notes = [
            "Operational threshold is fixed (0.30).",
            "Top-k is a batch/ranking policy used only when capacity cannot handle the fixed-threshold volume.",
            "Holdout is evaluation-only and never used to choose thresholds.",
        ]

    return {
        "operational_fixed_threshold": float(operational_threshold),
        "recall_target_for_calibration": float(recall_target),
        "calibrated_threshold": calibrated_threshold,
        "topk_fallback_fraction": float(topk_fraction),
        "operational": {
            "mode": "fixed",
            "threshold": float(operational_threshold),
            "rule": "alert_if_proba>=0.30",
        },
        "capacity_fallback": {
            "mode": "topk",
            "topk_fraction": float(topk_fraction),
            "rule": "alert_top_20_percent_by_score",
        },
        "notes": notes,
    }


def _resolve_feature_engineering_block(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("feature_engineering")
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled", False))
        enable_age_bucket = bool(raw.get("enable_age_bucket", False))
        engineered_cols = raw.get("engineered_cols")
        if not isinstance(engineered_cols, list):
            names = get_engineered_feature_names(enable_age_bucket=enable_age_bucket)
            engineered_cols = (
                list(names["numeric"]) + list(names["categorical"])
                if enabled
                else []
            )
        return {
            "enabled": enabled,
            "enable_age_bucket": enable_age_bucket,
            "engineered_cols": [str(col) for col in engineered_cols],
        }

    enabled = bool(metadata.get("enable_feature_engineering", False))
    enable_age_bucket = bool(metadata.get("enable_age_bucket", False))
    names = get_engineered_feature_names(enable_age_bucket=enable_age_bucket)
    engineered_cols = (
        list(names["numeric"]) + list(names["categorical"]) if enabled else []
    )
    return {
        "enabled": enabled,
        "enable_age_bucket": enable_age_bucket,
        "engineered_cols": [str(col) for col in engineered_cols],
    }


def _resolve_feature_pruning_block(
    metadata: dict[str, Any], expected_model_cols: list[str]
) -> dict[str, Any]:
    raw = metadata.get("feature_pruning")
    if isinstance(raw, dict):
        plan_hash = raw.get("plan_hash")
        kept_count = _to_int(raw.get("kept_model_cols_count"), len(expected_model_cols))
        dropped_summary = raw.get("dropped_summary")
        if dropped_summary is not None and not isinstance(dropped_summary, dict):
            dropped_summary = None
        return {
            "plan_hash": None if plan_hash is None else str(plan_hash),
            "kept_model_cols_count": int(kept_count),
            "dropped_summary": dropped_summary,
        }

    plan_hash = metadata.get("feature_pruning_plan_hash")
    pruning_plan = metadata.get("feature_pruning_plan")
    dropped_summary: dict[str, Any] | None = None
    if isinstance(pruning_plan, dict):
        dropped_summary = {
            "dropped_all_missing_cols_count": int(
                len(pruning_plan.get("dropped_all_missing_cols", []))
            ),
            "dropped_constant_numeric_cols_count": int(
                len(pruning_plan.get("dropped_constant_numeric_cols", []))
            ),
            "dropped_constant_categorical_cols_count": int(
                len(pruning_plan.get("dropped_constant_categorical_cols", []))
            ),
            "dropped_high_cardinality_cols_count": int(
                len(pruning_plan.get("dropped_high_cardinality_cols", []))
            ),
            "blocked_by_leakage_cols_count": int(
                len(pruning_plan.get("blocked_by_leakage_cols", []))
            ),
            "dropped_excluded_cols_count": int(
                len(pruning_plan.get("dropped_excluded_cols", []))
            ),
        }

    return {
        "plan_hash": None if plan_hash is None else str(plan_hash),
        "kept_model_cols_count": int(len(expected_model_cols)),
        "dropped_summary": dropped_summary,
    }


def _resolve_dataset_block(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("dataset")
    path_hint: str | None = None
    basename: str | None = None
    size_bytes: int | None = None
    mtime_utc: str | None = None
    sha256: str | None = None

    if isinstance(raw, dict):
        raw_path_hint = raw.get("path_hint")
        if isinstance(raw_path_hint, str) and raw_path_hint.strip():
            path_hint = safe_path_hint(raw_path_hint)

        raw_basename = raw.get("basename")
        if isinstance(raw_basename, str) and raw_basename.strip():
            basename = raw_basename

        raw_size = raw.get("bytes")
        if isinstance(raw_size, int):
            size_bytes = int(raw_size)

        raw_mtime = raw.get("mtime_utc")
        if isinstance(raw_mtime, str):
            mtime_utc = raw_mtime

        raw_sha = raw.get("sha256")
        if isinstance(raw_sha, str) and raw_sha.strip():
            sha256 = raw_sha

    fallback_path = (
        metadata.get("dataset_path_hint")
        or metadata.get("dataset_path")
        or metadata.get("file_path")
    )
    if path_hint is None and isinstance(fallback_path, str) and fallback_path.strip():
        path_hint = safe_path_hint(fallback_path)

    if basename is None:
        raw_dataset_basename = metadata.get("dataset_basename")
        if isinstance(raw_dataset_basename, str) and raw_dataset_basename.strip():
            basename = raw_dataset_basename
        elif path_hint:
            basename = str(path_hint)

    if sha256 is None:
        raw_dataset_sha = metadata.get("dataset_sha256")
        if isinstance(raw_dataset_sha, str) and raw_dataset_sha.strip():
            sha256 = raw_dataset_sha

    return {
        "path_hint": path_hint,
        "basename": basename,
        "bytes": size_bytes,
        "mtime_utc": mtime_utc,
        "sha256": sha256,
    }


def _resolve_train_pair(
    metadata: dict[str, Any],
    evaluation_train_at_030: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = metadata.get("train_pair")
    year_t = _to_int(raw.get("year_t"), 2022) if isinstance(raw, dict) else 2022
    year_t1 = _to_int(raw.get("year_t1"), 2023) if isinstance(raw, dict) else 2023
    n_default = _to_int(metadata.get("n_samples_train"), 0)
    n_pos_default = int(round(float(metadata.get("y_prevalence", 0.0)) * n_default))
    prevalence_default = _to_float_or_none(metadata.get("y_prevalence")) or 0.0

    if isinstance(evaluation_train_at_030, dict):
        n_default = _to_int(evaluation_train_at_030.get("n"), n_default)
        n_pos_default = _to_int(evaluation_train_at_030.get("n_pos"), n_pos_default)
        prevalence_default = _to_float_or_none(
            evaluation_train_at_030.get("prevalence")
        ) or prevalence_default

    if isinstance(raw, dict):
        return {
            "year_t": year_t,
            "year_t1": year_t1,
            "n": _to_int(raw.get("n"), n_default),
            "n_pos": _to_int(raw.get("n_pos"), n_pos_default),
            "prevalence": _to_float_or_none(raw.get("prevalence")) or prevalence_default,
        }

    return {
        "year_t": year_t,
        "year_t1": year_t1,
        "n": n_default,
        "n_pos": n_pos_default,
        "prevalence": prevalence_default,
    }


def _resolve_holdout_pair(
    metadata: dict[str, Any],
    evaluation_holdout_at_030: dict[str, Any] | None,
) -> dict[str, Any] | None:
    raw = metadata.get("holdout_pair")
    if isinstance(raw, dict):
        return {
            "year_t": _to_int(raw.get("year_t"), 2023),
            "year_t1": _to_int(raw.get("year_t1"), 2024),
            "n": _to_int(raw.get("n"), 0),
            "n_pos": _to_int(raw.get("n_pos"), 0),
            "prevalence": _to_float_or_none(raw.get("prevalence")) or 0.0,
        }
    if not isinstance(evaluation_holdout_at_030, dict):
        return None
    return {
        "year_t": 2023,
        "year_t1": 2024,
        "n": _to_int(evaluation_holdout_at_030.get("n"), 0),
        "n_pos": _to_int(evaluation_holdout_at_030.get("n_pos"), 0),
        "prevalence": _to_float_or_none(evaluation_holdout_at_030.get("prevalence"))
        or 0.0,
    }


def _enrich_metadata_for_serving(
    *,
    metadata: dict[str, Any],
    winner: dict[str, str],
    model_sha256: str,
    promoted_at: str,
) -> dict[str, Any]:
    eval_blocks = _resolve_eval_blocks(metadata)
    dataset_block = _resolve_dataset_block(metadata)
    expected_raw_cols = metadata.get("expected_raw_cols")
    if not isinstance(expected_raw_cols, list):
        expected_raw_cols = metadata.get("expected_cols")
    if not isinstance(expected_raw_cols, list):
        expected_raw_cols = []
    expected_model_cols = metadata.get("expected_model_cols")
    if not isinstance(expected_model_cols, list):
        expected_model_cols = (
            metadata.get("feature_pruning_plan", {}).get("kept_model_cols")
            if isinstance(metadata.get("feature_pruning_plan"), dict)
            else []
        )
    if not isinstance(expected_model_cols, list):
        expected_model_cols = []
    excluded_cols = metadata.get("excluded_cols")
    if not isinstance(excluded_cols, list):
        excluded_cols = []

    train_pair = _resolve_train_pair(
        metadata,
        eval_blocks["evaluation_train_at_0.30"],
    )
    holdout_pair = _resolve_holdout_pair(
        metadata,
        eval_blocks["evaluation_holdout_at_0.30"],
    )

    trained_at = str(
        metadata.get("trained_at") or metadata.get("created_at") or promoted_at
    )
    model_version = str(
        metadata.get("model_version")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    )
    notes_raw = metadata.get("notes")
    notes = list(notes_raw) if isinstance(notes_raw, list) else []
    if dataset_block.get("sha256") is None:
        notes.append(
            "dataset.sha256 unavailable in legacy metadata; value set to null during promotion."
        )

    enriched = dict(metadata)
    enriched.update(
        {
            "model_family": winner["model_family"],
            "variant": winner["variant"],
            "model_version": model_version,
            "trained_at": trained_at,
            "promoted_at": promoted_at,
            "random_state": int(metadata.get("random_state", RANDOM_STATE)),
            "train_pair": train_pair,
            "holdout_pair": holdout_pair,
            "dataset": dataset_block,
            "expected_raw_cols": [str(col) for col in expected_raw_cols],
            "expected_model_cols": [str(col) for col in expected_model_cols],
            "excluded_cols": [str(col) for col in excluded_cols],
            "feature_engineering": _resolve_feature_engineering_block(metadata),
            "feature_pruning": _resolve_feature_pruning_block(
                metadata,
                [str(col) for col in expected_model_cols],
            ),
            "threshold_policy": _resolve_threshold_policy(metadata),
            "evaluation_train_at_0.5": eval_blocks["evaluation_train_at_0.5"],
            "evaluation_train_at_0.30": eval_blocks["evaluation_train_at_0.30"],
            "evaluation_holdout_at_0.5": eval_blocks["evaluation_holdout_at_0.5"],
            "evaluation_holdout_at_0.30": eval_blocks["evaluation_holdout_at_0.30"],
            "evaluation_holdout_at_calibrated_threshold": eval_blocks[
                "evaluation_holdout_at_calibrated_threshold"
            ],
            "versions": _build_versions_block(metadata),
            "artifact_hashes": {
                "model_joblib_sha256": model_sha256,
                "metadata_sha256": None,
            },
            "notes": list(dict.fromkeys(str(note) for note in notes)),
        }
    )
    return enriched


def _resolve_policy_decision(
    *,
    selection_path: Path,
    source_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not selection_path.exists():
        raise FileNotFoundError(f"Selection artifact not found: {selection_path}")
    selection_payload = _safe_read_json(selection_path)
    decision = promotion_decision(selection_payload, winner_metadata=source_metadata)
    return selection_payload, decision


def run_model_promotion(
    *,
    selection_path: str | Path = "artifacts/model_selection.json",
    models_root: str | Path = "artifacts/models",
    out_dir: str | Path = "app/model",
    force: bool = False,
    backup: bool = True,
    stage_only: bool = False,
    promote: bool = False,
    from_staging: str | Path | None = None,
    allow_warning: bool = False,
) -> dict[str, Any]:
    if bool(stage_only) and bool(promote):
        raise ValueError("Flags --stage-only and --promote are mutually exclusive.")
    if bool(promote) and not from_staging:
        raise ValueError("Flag --promote 1 requires --from-staging <dir>.")
    if from_staging and not bool(promote):
        raise ValueError("Flag --from-staging requires --promote 1.")

    selection_path_obj = Path(selection_path)

    mode = "direct"
    if bool(stage_only):
        mode = "stage_only"
    elif bool(promote):
        mode = "promote_from_staging"

    if bool(stage_only) and str(out_dir) == "app/model":
        out_dir = str(Path("app/model") / "staging")

    source_metadata_for_policy: dict[str, Any]
    if bool(promote):
        staging_dir = Path(from_staging or "")
        resolved = _resolve_staging_paths(staging_dir)
        src_model = Path(resolved["src_model"])
        src_meta = Path(resolved["src_meta"])
        winner = dict(resolved["winner"])
        source_metadata_for_policy = dict(resolved["metadata"])
    else:
        if not selection_path_obj.exists():
            raise FileNotFoundError(f"Selection artifact not found: {selection_path_obj}")
        selection_payload = _safe_read_json(selection_path_obj)
        resolved = _resolve_winner_paths(
            selection_payload=selection_payload,
            models_root=Path(models_root),
        )
        src_model = Path(resolved["src_model"])
        src_meta = Path(resolved["src_meta"])
        winner = dict(resolved["winner"])
        source_metadata_for_policy = _safe_read_json(src_meta)

    selection_payload, policy_decision = _resolve_policy_decision(
        selection_path=selection_path_obj,
        source_metadata=source_metadata_for_policy,
    )
    if policy_decision["decision"] == "BLOCK":
        raise ValueError(f"Promotion blocked by policy: {policy_decision['reason']}")
    if policy_decision["decision"] == "ALLOW_WITH_OVERRIDE" and not bool(allow_warning):
        raise ValueError(
            "Promotion requires --allow-warning 1 because selection/policy status is WARNING."
        )

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    if bool(promote) and from_staging:
        staging_dir_resolved = Path(from_staging)
        if out_dir_path.resolve() == staging_dir_resolved.resolve():
            raise ValueError("Destination out_dir must differ from --from-staging directory.")
    dest_model = out_dir_path / "model.joblib"
    dest_meta = out_dir_path / "metadata.json"

    if dest_model.exists() and not bool(force):
        raise ValueError(
            "Destination exists. Use --force 1 to overwrite (backup enabled by default)."
        )

    backup_path: Path | None = None
    if bool(backup):
        backup_path = _backup_existing_destination(
            out_dir=out_dir_path,
            dest_model=dest_model,
            dest_meta=dest_meta,
        )

    shutil.copy2(src_model, dest_model)
    shutil.copy2(src_meta, dest_meta)

    model_sha = _sha256(dest_model)
    promoted_at = datetime.now(timezone.utc).isoformat()
    metadata_payload = _safe_read_json(dest_meta)
    metadata_payload = _enrich_metadata_for_serving(
        metadata=metadata_payload,
        winner=winner,
        model_sha256=model_sha,
        promoted_at=promoted_at,
    )

    forbidden_metadata = find_forbidden_json_keys(metadata_payload)
    if forbidden_metadata:
        raise ValueError(
            f"Privacy check failed: forbidden keys found in metadata payload: {forbidden_metadata}"
        )

    ok, errors = validate_metadata(metadata_payload)
    if not ok:
        error_excerpt = "; ".join(errors[:10])
        raise ValueError(f"Promoted metadata is invalid: {error_excerpt}")

    dest_meta.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    policy_eval = (
        dict(policy_decision.get("policy_evaluation"))
        if isinstance(policy_decision.get("policy_evaluation"), dict)
        else {}
    )
    metrics_used = dict(policy_eval.get("metrics")) if isinstance(policy_eval.get("metrics"), dict) else {}
    promoted_model_version = str(metadata_payload.get("model_version") or "")

    manifest_filename = "staging_manifest.json" if bool(stage_only) else "promoted_model.json"
    manifest_path = out_dir_path / manifest_filename

    promoted_payload = {
        "mode": mode,
        "promoted_at": promoted_at,
        "decision": {
            "status": str(policy_decision.get("status") or "UNKNOWN"),
            "decision": str(policy_decision.get("decision") or "BLOCK"),
            "reason": str(policy_decision.get("reason") or ""),
            "selection_status": str(policy_decision.get("selection_status") or "UNKNOWN"),
            "allow_warning_used": bool(allow_warning),
        },
        "policy_evaluation": policy_eval,
        "winner": {
            "model_family": winner["model_family"],
            "variant": winner["variant"],
        },
        "model_version": promoted_model_version or None,
        "source_paths": {
            "model": _safe_rel_path(src_model),
            "metadata": _safe_rel_path(src_meta),
        },
        "dest_paths": {
            "model": _safe_rel_path(dest_model),
            "metadata": _safe_rel_path(dest_meta),
            "manifest": _safe_rel_path(manifest_path),
        },
        "sha256": {
            "model": model_sha,
            "metadata": _sha256(dest_meta),
        },
        "summary": {
            "threshold_used": _to_float_or_none(policy_eval.get("threshold_used")),
            "recall_holdout": _to_float_or_none(metrics_used.get("recall")),
            "pr_auc_holdout": _to_float_or_none(metrics_used.get("pr_auc")),
            "positive_rate_holdout": _to_float_or_none(metrics_used.get("positive_rate")),
        },
        "backup": {
            "enabled": bool(backup),
            "path": str(backup_path) if backup_path is not None else None,
        },
        "notes": [
            "promotion copies a selected pipeline artifact for local serving",
            "app/model/model.joblib is the fixed serving path for future API loading",
            "staging mode writes app/model/staging/* and does not overwrite production serving path",
        ],
    }

    forbidden = find_forbidden_json_keys(promoted_payload)
    if forbidden:
        raise ValueError(
            f"Privacy check failed: forbidden keys found in promotion payload: {forbidden}"
        )

    manifest_path.write_text(
        json.dumps(promoted_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return promoted_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote winner model artifact to fixed serving path app/model."
    )
    parser.add_argument(
        "--selection-path",
        type=str,
        default="artifacts/model_selection.json",
        help="Path to model_selection artifact.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="artifacts/models",
        help="Root directory containing trained model variants.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="app/model",
        help="Serving destination directory.",
    )
    parser.add_argument(
        "--force",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, allow overwrite in destination.",
    )
    parser.add_argument(
        "--backup",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, backup existing destination model/metadata before overwrite.",
    )
    parser.add_argument(
        "--stage-only",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, copy winner to staging directory (does not touch prod app/model).",
    )
    parser.add_argument(
        "--promote",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, promote from --from-staging directory into --out-dir.",
    )
    parser.add_argument(
        "--from-staging",
        type=str,
        default=None,
        help="Staging directory containing model.joblib + metadata.json for promote mode.",
    )
    parser.add_argument(
        "--allow-warning",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, allow promotion when policy decision is ALLOW_WITH_OVERRIDE (WARNING).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    try:
        promoted = run_model_promotion(
            selection_path=args.selection_path,
            models_root=args.models_root,
            out_dir=args.out_dir,
            force=_parse_bool_flag(args.force),
            backup=_parse_bool_flag(args.backup),
            stage_only=_parse_bool_flag(args.stage_only),
            promote=_parse_bool_flag(args.promote),
            from_staging=args.from_staging,
            allow_warning=_parse_bool_flag(args.allow_warning),
        )
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("%s", exc)
        raise SystemExit(1) from exc

    _logger.info(
        "Model promotion completed | winner=%s/%s dest=%s",
        promoted["winner"]["model_family"],
        promoted["winner"]["variant"],
        promoted["dest_paths"]["model"],
    )


if __name__ == "__main__":
    main()
