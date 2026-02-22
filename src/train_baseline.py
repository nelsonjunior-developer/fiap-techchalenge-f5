"""CLI and helpers to train LogisticRegression baseline on temporal pair 2022->2023."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.cv import run_stratified_cv
from src.data import (
    get_default_dataset_path,
    load_pede_workbook_with_metadata,
    make_temporal_pairs,
)
from src.dataset_versioning import (
    get_dataset_fingerprint,
    persist_dataset_version_event,
)
from src.preprocessing import (
    build_preprocessing_bundle,
    build_pruning_plan_from_training_frame,
)
from src.features import get_engineered_feature_names
from src.metrics import (
    build_default_prediction_policy,
    compute_metrics_threshold,
    compute_metrics_topk,
    compute_prevalence,
    summarize_proba,
    select_threshold_for_target_recall,
)
from src.model_versioning import make_model_version
from src.thresholding import evaluate_at_threshold, select_threshold_by_recall
from src.train_pipeline import build_model_pipeline
from src.training_policy import OFFICIAL_TRAIN_PAIR, enforce_official_train_pair
from src.training_utils import build_raw_from_ids
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_ALLOWED_SCALERS = {"standard", "robust", "none"}
_ALLOWED_VARIANTS = {"none", "balanced"}
_FORBIDDEN_METADATA_KEYS = {"ids", "ra_list", "students", "rows"}
_OPERATIONAL_THRESHOLD = 0.30
_CAPACITY_TOPK_FRACTION = 0.20
_CALIBRATION_RECALL_TARGET = 0.90
_CALIBRATION_GRID_SIZE = 2001


def _require_training_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import sklearn
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "scikit-learn is required to train baseline. Install requirements-dev.txt"
        ) from exc

    return {
        "joblib": joblib,
        "sklearn_version": sklearn.__version__,
        "LogisticRegression": LogisticRegression,
    }


def _parse_bool_flag(value: int) -> bool:
    return bool(int(value))


def _parse_variants(raw_variants: str) -> list[str]:
    parsed = [variant.strip().lower() for variant in raw_variants.split(",") if variant.strip()]
    if not parsed:
        raise ValueError("No baseline variant provided. Use: none or balanced.")

    invalid = sorted({variant for variant in parsed if variant not in _ALLOWED_VARIANTS})
    if invalid:
        raise ValueError(
            f"Invalid variants: {invalid}. Allowed variants: {sorted(_ALLOWED_VARIANTS)}"
        )

    return list(dict.fromkeys(parsed))


def _resolve_dataset_path(dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        return get_default_dataset_path()
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return path


def _compute_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_pruning_plan(plan: dict[str, Any]) -> str:
    payload = json.dumps(plan, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_training_inputs(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    if X_train.empty:
        raise ValueError("X_train is empty.")
    if y_train.empty:
        raise ValueError("y_train is empty.")
    if "RA" in X_train.columns:
        raise ValueError("RA must not be present in X_train features.")

    unique_values = set(pd.Series(y_train).dropna().astype(int).unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Target is not binary: {sorted(unique_values)}")


def _build_evaluation_block(
    *,
    pair: str,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    threshold_label: str,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    payload = evaluate_at_threshold(
        y_true,
        scores,
        threshold=threshold,
    )
    notes = list(payload.get("notes", []))
    if extra_notes:
        notes.extend(extra_notes)
    confusion_key = f"confusion_matrix_at_{threshold_label}"
    return {
        "pair": pair,
        "threshold": float(threshold),
        "n": int(payload["n"]),
        "n_pos": int(payload["n_pos"]),
        "prevalence": float(payload["prevalence"]),
        "metrics": dict(payload["metrics"]),
        "confusion_matrix": dict(payload["confusion_matrix"]),
        confusion_key: dict(payload["confusion_matrix"]),
        "pred_proba_summary": summarize_proba(scores),
        "notes": notes,
    }


def _build_threshold_policy() -> dict[str, Any]:
    notes = [
        "Operational threshold is fixed (0.30).",
        "Top-k is a batch/ranking policy used only when capacity cannot handle the fixed-threshold volume.",
        "Holdout is evaluation-only and never used to choose thresholds.",
    ]
    return {
        "operational_fixed_threshold": float(_OPERATIONAL_THRESHOLD),
        "recall_target_for_calibration": float(_CALIBRATION_RECALL_TARGET),
        "calibrated_threshold": None,
        "topk_fallback_fraction": float(_CAPACITY_TOPK_FRACTION),
        "operational": {
            "mode": "fixed",
            "threshold": float(_OPERATIONAL_THRESHOLD),
            "rule": "alert_if_proba>=0.30",
        },
        "capacity_fallback": {
            "mode": "topk",
            "topk_fraction": float(_CAPACITY_TOPK_FRACTION),
            "rule": "alert_top_20_percent_by_score",
        },
        "notes": notes,
    }


def _build_feature_pruning_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "dropped_all_missing_cols_count": int(
            len(plan.get("dropped_all_missing_cols", []))
        ),
        "dropped_constant_numeric_cols_count": int(
            len(plan.get("dropped_constant_numeric_cols", []))
        ),
        "dropped_constant_categorical_cols_count": int(
            len(plan.get("dropped_constant_categorical_cols", []))
        ),
        "dropped_high_cardinality_cols_count": int(
            len(plan.get("dropped_high_cardinality_cols", []))
        ),
        "blocked_by_leakage_cols_count": int(
            len(plan.get("blocked_by_leakage_cols", []))
        ),
        "dropped_excluded_cols_count": int(len(plan.get("dropped_excluded_cols", []))),
    }


def _extract_pair_years(pair: str, fallback_t: int, fallback_t1: int) -> tuple[int, int]:
    raw = str(pair or "").strip()
    if "->" not in raw:
        return fallback_t, fallback_t1
    left, right = raw.split("->", 1)
    try:
        return int(left.strip()), int(right.strip())
    except ValueError:
        return fallback_t, fallback_t1


def _build_metadata_payload(
    *,
    model_path: Path,
    model_joblib_sha256: str,
    model_family: str,
    variant: str,
    class_weight: str | None,
    year_t: int,
    year_t1: int,
    dataset_fingerprint: dict[str, Any],
    scaler_strategy: str,
    enable_feature_engineering: bool,
    enable_age_bucket: bool,
    expected_raw_cols: list[str],
    expected_model_cols: list[str],
    excluded_cols: list[str],
    feature_pruning_plan: dict[str, Any],
    feature_pruning_plan_hash: str,
    dataset_basename: str | None,
    dataset_sha256: str | None,
    n_samples_train: int,
    y_prevalence: float,
    evaluation_train_at_05: dict[str, Any],
    evaluation_train_at_030: dict[str, Any],
    threshold_calibration: dict[str, Any],
    threshold_policy: dict[str, Any],
    sklearn_version: str,
    joblib_version: str | None,
    class_imbalance_strategy: dict[str, Any],
    prediction_policy: dict[str, Any],
    evaluation_holdout_at_05: dict[str, Any] | None,
    evaluation_holdout_at_030: dict[str, Any] | None,
    evaluation_holdout_at_calibrated_threshold: dict[str, Any] | None,
    topk_holdout_summary: dict[str, Any] | None,
    cv_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trained_at = datetime.now(timezone.utc).isoformat()
    model_version = make_model_version(trained_at, model_joblib_sha256)
    engineered_names = get_engineered_feature_names(enable_age_bucket=enable_age_bucket)
    engineered_cols: list[str] = []
    if enable_feature_engineering:
        engineered_cols = list(engineered_names["numeric"]) + list(
            engineered_names["categorical"]
        )

    holdout_pair: dict[str, Any] | None = None
    if isinstance(evaluation_holdout_at_030, dict):
        holdout_year_t, holdout_year_t1 = _extract_pair_years(
            str(evaluation_holdout_at_030.get("pair", "2023->2024")),
            2023,
            2024,
        )
        holdout_pair = {
            "year_t": int(holdout_year_t),
            "year_t1": int(holdout_year_t1),
            "n": int(evaluation_holdout_at_030.get("n", 0)),
            "n_pos": int(evaluation_holdout_at_030.get("n_pos", 0)),
            "prevalence": float(evaluation_holdout_at_030.get("prevalence", 0.0)),
        }

    threshold_policy_payload = dict(threshold_policy)
    threshold_policy_payload["operational_fixed_threshold"] = float(
        _OPERATIONAL_THRESHOLD
    )
    threshold_policy_payload["recall_target_for_calibration"] = float(
        _CALIBRATION_RECALL_TARGET
    )
    threshold_policy_payload["calibrated_threshold"] = float(
        threshold_calibration.get("threshold_selected")
    )
    threshold_policy_payload["topk_fallback_fraction"] = float(_CAPACITY_TOPK_FRACTION)

    versions_payload = {
        "python": sys.version.split(" ")[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "joblib": joblib_version,
        # Backward compatibility with older metadata consumers.
        "sklearn": sklearn_version,
    }

    notes = [
        "operational default policy uses fixed threshold=0.30 for recall-focused mode",
        "threshold_calibration is train-only evidence and does not replace operational threshold",
        "holdout evaluation is read-only (no fitting on 2023->2024)",
    ]
    if evaluation_holdout_at_05 is None:
        notes.append("holdout evaluation disabled by flag")
    payload = {
        "model_family": model_family,
        "model_kind": "LogisticRegression",
        "variant": variant,
        "model_version": model_version,
        "trained_at": trained_at,
        "promoted_at": None,
        "random_state": int(RANDOM_STATE),
        "class_weight": class_weight,
        "train_pair": {
            "year_t": int(year_t),
            "year_t1": int(year_t1),
            "n": int(evaluation_train_at_030.get("n", n_samples_train)),
            "n_pos": int(evaluation_train_at_030.get("n_pos", 0)),
            "prevalence": float(evaluation_train_at_030.get("prevalence", y_prevalence)),
        },
        "holdout_pair": holdout_pair,
        "dataset": {
            "path_hint": dataset_fingerprint.get("path_hint"),
            "basename": dataset_fingerprint.get("basename", dataset_basename),
            "bytes": dataset_fingerprint.get("bytes"),
            "mtime_utc": dataset_fingerprint.get("mtime_utc"),
            "sha256": dataset_fingerprint.get("sha256", dataset_sha256),
        },
        "expected_raw_cols": list(expected_raw_cols),
        "expected_model_cols": list(expected_model_cols),
        "excluded_cols": list(excluded_cols),
        "feature_engineering": {
            "enabled": bool(enable_feature_engineering),
            "enable_age_bucket": bool(enable_age_bucket),
            "engineered_cols": engineered_cols,
        },
        "feature_pruning": {
            "plan_hash": feature_pruning_plan_hash,
            "kept_model_cols_count": int(len(expected_model_cols)),
            "dropped_summary": _build_feature_pruning_summary(feature_pruning_plan),
        },
        "scaler_strategy": scaler_strategy,
        "enable_feature_engineering": bool(enable_feature_engineering),
        "enable_age_bucket": bool(enable_age_bucket),
        "feature_pruning_plan_hash": feature_pruning_plan_hash,
        "dataset_basename": dataset_basename,
        "dataset_sha256": dataset_sha256,
        "n_samples_train": n_samples_train,
        "y_prevalence": y_prevalence,
        "train_pred_proba_summary": evaluation_train_at_05["pred_proba_summary"],
        "metrics_train_at_0.5": evaluation_train_at_05["metrics"],
        "metrics_holdout_at_0.5": (
            evaluation_holdout_at_05.get("metrics")
            if isinstance(evaluation_holdout_at_05, dict)
            else None
        ),
        "evaluation_train": evaluation_train_at_05,
        "evaluation_holdout": evaluation_holdout_at_05,
        "evaluation_train_at_0.5": evaluation_train_at_05,
        "evaluation_train_at_0.30": evaluation_train_at_030,
        "evaluation_holdout_at_0.5": evaluation_holdout_at_05,
        "evaluation_holdout_at_0.30": evaluation_holdout_at_030,
        "evaluation_holdout_at_calibrated_threshold": evaluation_holdout_at_calibrated_threshold,
        "evaluation_holdout_at_threshold_selected": evaluation_holdout_at_calibrated_threshold,
        "threshold_policy": threshold_policy_payload,
        "threshold_calibration": threshold_calibration,
        "topk_holdout_summary": topk_holdout_summary,
        "class_imbalance_strategy": class_imbalance_strategy,
        "prediction_policy": prediction_policy,
        "created_at": trained_at,
        "versions": versions_payload,
        "artifact_hashes": {
            "model_joblib_sha256": model_joblib_sha256,
            "metadata_sha256": None,
        },
        "notes": notes,
        "path_model": str(model_path),
    }
    forbidden_present = _FORBIDDEN_METADATA_KEYS & set(payload.keys())
    if forbidden_present:
        raise ValueError(f"Forbidden metadata keys found: {sorted(forbidden_present)}")
    if cv_result is not None:
        payload["cv"] = cv_result
    return payload


def _build_class_imbalance_strategy(
    *,
    y_train: pd.Series,
    y_holdout: pd.Series | None,
    variant_scores_train: dict[str, np.ndarray],
    variant_scores_holdout: dict[str, np.ndarray | None],
    class_weight_by_variant: dict[str, str],
) -> dict[str, Any]:
    train_prev = compute_prevalence(y_train)
    hold_prev = compute_prevalence(y_holdout) if y_holdout is not None else None

    by_variant_threshold_05: dict[str, dict[str, Any]] = {}
    class_weight_tested = sorted(
        {class_weight_by_variant.get(variant, "none") for variant in variant_scores_train}
    )
    for variant, train_scores in variant_scores_train.items():
        hold_scores = variant_scores_holdout.get(variant)
        variant_entry: dict[str, Any] = {
            "class_weight": class_weight_by_variant.get(variant, "none"),
            "train": compute_metrics_threshold(y_train, train_scores, threshold=0.5),
            "holdout": (
                compute_metrics_threshold(y_holdout, hold_scores, threshold=0.5)
                if (y_holdout is not None and hold_scores is not None)
                else None
            ),
        }
        by_variant_threshold_05[str(variant)] = variant_entry

    tuned_threshold_info: dict[str, Any] | None = None
    if "none" in variant_scores_train:
        tuned_thr = select_threshold_for_target_recall(
            y_train,
            variant_scores_train["none"],
            target_recall=_CALIBRATION_RECALL_TARGET,
        )
        tuned_threshold_info = {
            "source_variant": "none",
            "selection_rule": "max precision subject to recall>=0.90 on train",
            "target_recall": float(_CALIBRATION_RECALL_TARGET),
            "selected_threshold": float(tuned_thr),
            "train": compute_metrics_threshold(
                y_train,
                variant_scores_train["none"],
                threshold=tuned_thr,
            ),
            "holdout": (
                compute_metrics_threshold(
                    y_holdout,
                    variant_scores_holdout["none"],  # type: ignore[arg-type]
                    threshold=tuned_thr,
                )
                if (
                    y_holdout is not None
                    and variant_scores_holdout.get("none") is not None
                )
                else None
            ),
        }

    topk_reference_variant = "none" if "none" in variant_scores_holdout else next(
        iter(variant_scores_holdout.keys()),
        None,
    )
    topk_on_holdout: dict[str, Any] | None = None
    if (
        y_holdout is not None
        and topk_reference_variant is not None
        and variant_scores_holdout.get(topk_reference_variant) is not None
    ):
        ref_scores = variant_scores_holdout[topk_reference_variant]
        topk_on_holdout = {
            "reference_variant": topk_reference_variant,
            "topk_10pct": compute_metrics_topk(y_holdout, ref_scores, k_frac=0.10),
            "topk_20pct": compute_metrics_topk(y_holdout, ref_scores, k_frac=0.20),
        }

    evidence: dict[str, Any] = {
        "by_variant_threshold_0.5": by_variant_threshold_05,
        "threshold_tuned_from_train": tuned_threshold_info,
        "topk_holdout": topk_on_holdout,
    }
    notes = [
        "Class positive prevalence is higher in train than holdout; temporal drift is present.",
        "class_weight='balanced' is not default; keep class_weight='none' unless new evidence supports change.",
        "Threshold fixo e o modo operacional default para foco em recall; top-k fica como contingencia por capacidade.",
    ]
    if hold_prev is None:
        notes.append("Holdout prevalence and holdout evidence are null because holdout evaluation was disabled.")

    return {
        "train_prevalence": train_prev["prevalence"],
        "holdout_prevalence": None if hold_prev is None else hold_prev["prevalence"],
        "default": {"kind": "threshold", "threshold": float(_OPERATIONAL_THRESHOLD)},
        "alternatives": [
            {"kind": "top_k", "k_frac": float(_CAPACITY_TOPK_FRACTION)},
            {"kind": "threshold", "threshold": 0.50},
        ],
        "class_weight_tested": class_weight_tested,
        "decision": (
            "Use class_weight=none by default; use fixed threshold=0.30 for recall-focused operation; "
            "use top-k=20% when capacity is constrained."
        ),
        "evidence": evidence,
        "notes": notes,
    }


def run_baseline_training(
    *,
    dataset_path: str | Path | None = None,
    year_t: int = 2022,
    year_t1: int = 2023,
    out_dir: str | Path = "artifacts/models/baseline_logreg",
    scaler_strategy: str = "standard",
    variants: str = "none",
    enable_feature_engineering: bool = True,
    enable_age_bucket: bool = True,
    allow_nontrain_pair: bool = False,
    allow_holdout_training: bool = False,
    eval_holdout: bool = True,
    enable_cv: bool = False,
    cv_splits: int = 5,
    cv_repeat: int = 1,
    strict: bool = False,
) -> dict[str, Any]:
    enforce_official_train_pair(
        year_t=year_t,
        year_t1=year_t1,
        allow_nontrain_pair=allow_nontrain_pair,
        allow_holdout_training=allow_holdout_training,
    )
    if (year_t, year_t1) != OFFICIAL_TRAIN_PAIR:
        _logger.warning(
            "Non-official training pair enabled: %s->%s",
            year_t,
            year_t1,
        )
    if enable_cv and (year_t, year_t1) != OFFICIAL_TRAIN_PAIR:
        raise ValueError(
            "Internal CV is restricted to official training pair 2022->2023."
        )

    deps = _require_training_dependencies()
    if scaler_strategy not in _ALLOWED_SCALERS:
        raise ValueError(
            f"Invalid scaler '{scaler_strategy}'. Allowed: {sorted(_ALLOWED_SCALERS)}"
        )

    variants_list = _parse_variants(variants)
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    yearly_frames, _, _ = load_pede_workbook_with_metadata(file_path=resolved_dataset_path)

    if year_t not in yearly_frames or year_t1 not in yearly_frames:
        raise ValueError(
            f"Year pair {year_t}->{year_t1} not available in loaded data: {sorted(yearly_frames)}"
        )

    X_pairs, y_train, ids = make_temporal_pairs(
        yearly_frames[year_t],
        yearly_frames[year_t1],
        year_t,
        year_t1,
    )
    _validate_training_inputs(X_pairs, y_train)

    raw_bundle = build_preprocessing_bundle(
        numeric_scaler=scaler_strategy,
        enable_feature_engineering=enable_feature_engineering,
        enable_age_bucket=enable_age_bucket,
    )
    expected_raw_cols = list(raw_bundle["expected_raw_cols"])
    excluded_cols = list(raw_bundle.get("excluded_cols", []))
    X_raw_train = build_raw_from_ids(yearly_frames[year_t], ids, expected_raw_cols)
    if len(X_raw_train) != len(y_train):
        raise ValueError("Inconsistent training rows between X_raw_train and y_train.")

    y_holdout: pd.Series | None = None
    X_raw_holdout: pd.DataFrame | None = None
    if bool(eval_holdout):
        if 2023 not in yearly_frames or 2024 not in yearly_frames:
            raise ValueError("Holdout evaluation requested but years 2023/2024 are unavailable.")
        _, y_holdout, ids_holdout = make_temporal_pairs(
            yearly_frames[2023],
            yearly_frames[2024],
            2023,
            2024,
        )
        X_raw_holdout = build_raw_from_ids(
            yearly_frames[2023],
            ids_holdout,
            expected_raw_cols,
        )
        if len(X_raw_holdout) != len(y_holdout):
            raise ValueError("Inconsistent holdout rows between X_raw_holdout and y_holdout.")

    feature_pruning_plan = build_pruning_plan_from_training_frame(
        X_train_raw=X_raw_train,
        enable_feature_engineering=enable_feature_engineering,
        enable_age_bucket=enable_age_bucket,
    )
    if not feature_pruning_plan.get("kept_model_cols"):
        raise ValueError("Feature pruning plan produced empty kept_model_cols.")
    expected_model_cols = list(feature_pruning_plan.get("kept_model_cols", []))

    pruning_hash = _hash_pruning_plan(feature_pruning_plan)
    dataset_basename = resolved_dataset_path.name
    dataset_fingerprint = get_dataset_fingerprint(resolved_dataset_path)
    dataset_sha256 = str(dataset_fingerprint["sha256"])
    persist_dataset_version_event(
        context="train_baseline",
        dataset_fingerprint=dataset_fingerprint,
    )
    base_output_dir = Path(out_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    successes: dict[str, Any] = {}
    failures: dict[str, str] = {}
    variant_scores_train: dict[str, np.ndarray] = {}
    variant_scores_holdout: dict[str, np.ndarray | None] = {}
    class_weight_by_variant: dict[str, str] = {}
    metadata_payload_by_variant: dict[str, dict[str, Any]] = {}
    metadata_path_by_variant: dict[str, Path] = {}

    for variant in variants_list:
        class_weight: str | None = None if variant == "none" else "balanced"
        try:
            def _model_factory() -> Any:
                return deps["LogisticRegression"](
                    class_weight=class_weight,
                    max_iter=2000,
                    solver="lbfgs",
                )

            estimator = _model_factory()
            pipeline = build_model_pipeline(
                model=estimator,
                year_t=year_t,
                scaler_strategy=scaler_strategy,
                enable_feature_engineering=enable_feature_engineering,
                feature_pruning_plan=feature_pruning_plan,
                strict_raw=bool(strict),
                enable_age_bucket=enable_age_bucket,
            )
            pipeline.fit(X_raw_train, y_train)

            scores = pipeline.predict_proba(X_raw_train)[:, 1]
            holdout_scores = (
                pipeline.predict_proba(X_raw_holdout)[:, 1]
                if X_raw_holdout is not None
                else None
            )
            variant_scores_train[variant] = scores
            variant_scores_holdout[variant] = holdout_scores
            class_weight_by_variant[variant] = "none" if class_weight is None else str(class_weight)
            evaluation_train_at_05 = _build_evaluation_block(
                pair=f"{year_t}->{year_t1}",
                y_true=y_train,
                scores=scores,
                threshold=0.5,
                threshold_label="0.5",
                extra_notes=[
                    "train-only metrics computed on the same data used for fit",
                ],
            )
            evaluation_train_at_030 = _build_evaluation_block(
                pair=f"{year_t}->{year_t1}",
                y_true=y_train,
                scores=scores,
                threshold=_OPERATIONAL_THRESHOLD,
                threshold_label="0.30",
                extra_notes=[
                    "train-only metrics computed on the same data used for fit",
                ],
            )
            threshold_calibration_raw = select_threshold_by_recall(
                y_true=y_train,
                y_proba=scores,
                recall_target=_CALIBRATION_RECALL_TARGET,
                grid_size=_CALIBRATION_GRID_SIZE,
            )
            threshold_selected = float(threshold_calibration_raw["threshold_selected"])
            evaluation_train_at_threshold_selected = _build_evaluation_block(
                pair=f"{year_t}->{year_t1}",
                y_true=y_train,
                scores=scores,
                threshold=threshold_selected,
                threshold_label="threshold_selected",
                extra_notes=[
                    "threshold selected on train to satisfy recall target",
                ],
            )
            threshold_calibration = {
                "mode": "fixed_calibrated_on_train",
                "recall_target": float(_CALIBRATION_RECALL_TARGET),
                "selection_rule": str(threshold_calibration_raw["selection_rule"]),
                "threshold_selected": threshold_selected,
                "grid_size": int(threshold_calibration_raw["grid_size"]),
                "achieved_recall": float(threshold_calibration_raw["achieved_recall"]),
                "achieved_precision": float(threshold_calibration_raw["achieved_precision"]),
                "achieved_positive_rate": float(threshold_calibration_raw["achieved_positive_rate"]),
                "confusion_matrix_at_selected": dict(
                    threshold_calibration_raw["confusion_matrix_at_selected"]
                ),
                "notes": list(threshold_calibration_raw["notes"]),
                "evaluation_train_at_threshold_selected": evaluation_train_at_threshold_selected,
            }
            evaluation_holdout_at_05: dict[str, Any] | None = None
            evaluation_holdout_at_030: dict[str, Any] | None = None
            evaluation_holdout_at_threshold_selected: dict[str, Any] | None = None
            topk_holdout_summary: dict[str, Any] | None = None
            if y_holdout is not None and holdout_scores is not None:
                holdout_notes = [
                    "holdout evaluation is read-only; model was not fitted on holdout data",
                ]
                evaluation_holdout_at_05 = _build_evaluation_block(
                    pair="2023->2024",
                    y_true=y_holdout,
                    scores=holdout_scores,
                    threshold=0.5,
                    threshold_label="0.5",
                    extra_notes=holdout_notes,
                )
                evaluation_holdout_at_030 = _build_evaluation_block(
                    pair="2023->2024",
                    y_true=y_holdout,
                    scores=holdout_scores,
                    threshold=_OPERATIONAL_THRESHOLD,
                    threshold_label="0.30",
                    extra_notes=holdout_notes,
                )
                evaluation_holdout_at_threshold_selected = _build_evaluation_block(
                    pair="2023->2024",
                    y_true=y_holdout,
                    scores=holdout_scores,
                    threshold=threshold_selected,
                    threshold_label="threshold_selected",
                    extra_notes=holdout_notes,
                )
                topk_holdout_summary = compute_metrics_topk(
                    y_holdout,
                    holdout_scores,
                    k_frac=_CAPACITY_TOPK_FRACTION,
                )

            metrics_at_05 = evaluation_train_at_05["metrics"]

            variant_dir = base_output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            model_path = variant_dir / "model.joblib"
            metadata_path = variant_dir / "metadata.json"
            deps["joblib"].dump(pipeline, model_path)
            model_joblib_sha256 = _compute_sha256(model_path)
            if not model_joblib_sha256:
                raise ValueError(f"Unable to compute model sha256: {model_path}")
            metadata_path_by_variant[variant] = metadata_path

            metadata_payload_by_variant[variant] = {
                "model_path": model_path,
                "model_joblib_sha256": model_joblib_sha256,
                "model_family": "baseline_logreg",
                "variant": variant,
                "class_weight": class_weight,
                "year_t": year_t,
                "year_t1": year_t1,
                "dataset_fingerprint": dataset_fingerprint,
                "scaler_strategy": scaler_strategy,
                "enable_feature_engineering": enable_feature_engineering,
                "enable_age_bucket": enable_age_bucket,
                "expected_raw_cols": expected_raw_cols,
                "expected_model_cols": expected_model_cols,
                "excluded_cols": excluded_cols,
                "feature_pruning_plan": feature_pruning_plan,
                "feature_pruning_plan_hash": pruning_hash,
                "dataset_basename": dataset_basename,
                "dataset_sha256": dataset_sha256,
                "n_samples_train": int(len(y_train)),
                "y_prevalence": float(np.mean(y_train.astype(int).to_numpy())),
                "evaluation_train_at_05": evaluation_train_at_05,
                "evaluation_train_at_030": evaluation_train_at_030,
                "threshold_calibration": threshold_calibration,
                "evaluation_holdout_at_05": evaluation_holdout_at_05,
                "evaluation_holdout_at_030": evaluation_holdout_at_030,
                "evaluation_holdout_at_calibrated_threshold": evaluation_holdout_at_threshold_selected,
                "topk_holdout_summary": topk_holdout_summary,
                "sklearn_version": str(deps["sklearn_version"]),
                "joblib_version": (
                    str(getattr(deps["joblib"], "__version__", ""))
                    if getattr(deps["joblib"], "__version__", None) is not None
                    else None
                ),
            }

            successes[variant] = {
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "n_samples_train": int(len(y_train)),
                "y_prevalence": float(np.mean(y_train.astype(int).to_numpy())),
                "metrics_train_at_0.5": metrics_at_05,
                "cv_enabled": bool(enable_cv),
            }
            _logger.info(
                "Baseline variant trained | pair=%s->%s variant=%s scaler=%s n=%d prevalence=%.4f",
                year_t,
                year_t1,
                variant,
                scaler_strategy,
                len(y_train),
                float(np.mean(y_train.astype(int).to_numpy())),
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            failures[variant] = str(exc)
            _logger.error(
                "Baseline variant failed | variant=%s error=%s",
                variant,
                exc,
            )
            if strict:
                raise

    if not successes:
        raise RuntimeError(
            f"No baseline variant trained successfully. Failures: {failures}"
        )

    strategy_block = _build_class_imbalance_strategy(
        y_train=y_train,
        y_holdout=y_holdout,
        variant_scores_train=variant_scores_train,
        variant_scores_holdout=variant_scores_holdout,
        class_weight_by_variant=class_weight_by_variant,
    )
    threshold_policy = _build_threshold_policy()
    prediction_policy = build_default_prediction_policy(
        kind="threshold",
        k_frac=_CAPACITY_TOPK_FRACTION,
        threshold=_OPERATIONAL_THRESHOLD,
        score_name="risk_proba",
    )
    cv_payload_by_variant: dict[str, dict[str, Any] | None] = {}
    if enable_cv:
        for variant in variants_list:
            class_weight = None if variant == "none" else "balanced"
            model_factory = lambda cw=class_weight: deps["LogisticRegression"](
                class_weight=cw,
                max_iter=2000,
                solver="lbfgs",
            )
            cv_payload_by_variant[variant] = run_stratified_cv(
                model_name=f"logreg_{variant}",
                model_factory=model_factory,
                build_pipeline_fn=build_model_pipeline,
                X_raw_train=X_raw_train,
                y_train=y_train,
                year_t=year_t,
                feature_pruning_plan=feature_pruning_plan,
                scaler_strategy=scaler_strategy,
                enable_feature_engineering=enable_feature_engineering,
                enable_age_bucket=enable_age_bucket,
                strict_raw=bool(strict),
                n_splits=int(cv_splits),
                repeat_n=int(cv_repeat),
                random_state=RANDOM_STATE,
            )
    else:
        for variant in variants_list:
            cv_payload_by_variant[variant] = None

    for variant in successes:
        base_payload = metadata_payload_by_variant[variant]
        metadata_payload = _build_metadata_payload(
            model_path=base_payload["model_path"],
            model_joblib_sha256=base_payload["model_joblib_sha256"],
            model_family=base_payload["model_family"],
            variant=base_payload["variant"],
            class_weight=base_payload["class_weight"],
            year_t=base_payload["year_t"],
            year_t1=base_payload["year_t1"],
            dataset_fingerprint=base_payload["dataset_fingerprint"],
            scaler_strategy=base_payload["scaler_strategy"],
            enable_feature_engineering=base_payload["enable_feature_engineering"],
            enable_age_bucket=base_payload["enable_age_bucket"],
            expected_raw_cols=base_payload["expected_raw_cols"],
            expected_model_cols=base_payload["expected_model_cols"],
            excluded_cols=base_payload["excluded_cols"],
            feature_pruning_plan=base_payload["feature_pruning_plan"],
            feature_pruning_plan_hash=base_payload["feature_pruning_plan_hash"],
            dataset_basename=base_payload["dataset_basename"],
            dataset_sha256=base_payload["dataset_sha256"],
            n_samples_train=base_payload["n_samples_train"],
            y_prevalence=base_payload["y_prevalence"],
            evaluation_train_at_05=base_payload["evaluation_train_at_05"],
            evaluation_train_at_030=base_payload["evaluation_train_at_030"],
            threshold_calibration=base_payload["threshold_calibration"],
            threshold_policy=threshold_policy,
            sklearn_version=base_payload["sklearn_version"],
            joblib_version=base_payload["joblib_version"],
            class_imbalance_strategy=strategy_block,
            prediction_policy=prediction_policy,
            evaluation_holdout_at_05=base_payload["evaluation_holdout_at_05"],
            evaluation_holdout_at_030=base_payload["evaluation_holdout_at_030"],
            evaluation_holdout_at_calibrated_threshold=base_payload[
                "evaluation_holdout_at_calibrated_threshold"
            ],
            topk_holdout_summary=base_payload["topk_holdout_summary"],
            cv_result=cv_payload_by_variant.get(variant),
        )
        metadata_path = metadata_path_by_variant[variant]
        metadata_path.write_text(
            json.dumps(metadata_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "dataset_path": str(resolved_dataset_path),
        "out_dir": str(base_output_dir),
        "year_t": year_t,
        "year_t1": year_t1,
        "variants": successes,
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LogisticRegression baseline on temporal pair t->t+1."
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to XLSX dataset.")
    parser.add_argument("--year-t", type=int, default=2022, help="Training feature year.")
    parser.add_argument("--year-t1", type=int, default=2023, help="Target year.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/models/baseline_logreg",
        help="Output directory for baseline artifacts.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="standard",
        choices=sorted(_ALLOWED_SCALERS),
        help="Numeric scaler strategy.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="none",
        help="Comma-separated variants: none,balanced",
    )
    parser.add_argument(
        "--enable-feature-engineering",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable engineered features before preprocessing.",
    )
    parser.add_argument(
        "--enable-age-bucket",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable age_bucket engineered categorical feature.",
    )
    parser.add_argument(
        "--strict",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, fail immediately on any variant error.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable optional internal stratified CV on 2022->2023 train pair.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of folds for stratified CV (when --cv=1).",
    )
    parser.add_argument(
        "--cv-repeat",
        type=int,
        default=1,
        help="Number of repeats for stratified CV (1 = StratifiedKFold).",
    )
    parser.add_argument(
        "--eval-holdout",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, evaluate trained model on holdout pair 2023->2024 (no fitting on holdout).",
    )
    parser.add_argument(
        "--allow-nontrain-pair",
        type=int,
        default=0,
        choices=[0, 1],
        help="Allow training with pair different from 2022->2023 (not recommended).",
    )
    parser.add_argument(
        "--allow-holdout-training",
        type=int,
        default=0,
        choices=[0, 1],
        help="Allow training on holdout pair 2023->2024 when nontrain override is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    try:
        report = run_baseline_training(
            dataset_path=args.dataset_path,
            year_t=int(args.year_t),
            year_t1=int(args.year_t1),
            out_dir=args.out_dir,
            scaler_strategy=args.scaler,
            variants=args.variants,
            enable_feature_engineering=_parse_bool_flag(args.enable_feature_engineering),
            enable_age_bucket=_parse_bool_flag(args.enable_age_bucket),
            allow_nontrain_pair=_parse_bool_flag(args.allow_nontrain_pair),
            allow_holdout_training=_parse_bool_flag(args.allow_holdout_training),
            enable_cv=_parse_bool_flag(args.cv),
            cv_splits=int(args.cv_splits),
            cv_repeat=int(args.cv_repeat),
            eval_holdout=_parse_bool_flag(args.eval_holdout),
            strict=_parse_bool_flag(args.strict),
        )
    except ValueError as exc:
        _logger.error("%s", exc)
        raise SystemExit(1) from exc

    _logger.info(
        "Baseline training completed | pair=%s->%s variants=%s out_dir=%s",
        report["year_t"],
        report["year_t1"],
        sorted(report["variants"].keys()),
        report["out_dir"],
    )


if __name__ == "__main__":
    main()
