"""CLI and helpers to train HistGradientBoostingClassifier on temporal pair 2022->2023."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
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
from src.preprocessing import (
    build_preprocessing_bundle,
    build_pruning_plan_from_training_frame,
)
from src.metrics import (
    build_default_prediction_policy,
    compute_metrics_threshold,
    compute_metrics_topk,
    compute_prevalence,
    summarize_proba,
)
from src.thresholding import evaluate_at_threshold, select_threshold_by_recall
from src.train_pipeline import build_model_pipeline
from src.training_policy import OFFICIAL_TRAIN_PAIR, enforce_official_train_pair
from src.training_utils import build_raw_from_ids
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_ALLOWED_VARIANTS = {"default", "tuned"}
_FORBIDDEN_METADATA_KEYS = {"ids", "ra_list", "students", "rows"}
_OPERATIONAL_THRESHOLD = 0.30
_CAPACITY_TOPK_FRACTION = 0.20
_CALIBRATION_RECALL_TARGET = 0.90
_CALIBRATION_GRID_SIZE = 2001


def _require_training_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "scikit-learn/joblib is required to train nonlinear model. Install requirements-dev.txt"
        ) from exc

    return {
        "joblib": joblib,
        "sklearn_version": sklearn.__version__,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    }


def _parse_bool_flag(value: int) -> bool:
    return bool(int(value))


def _parse_variants(raw_variants: str) -> list[str]:
    parsed = [variant.strip().lower() for variant in raw_variants.split(",") if variant.strip()]
    if not parsed:
        raise ValueError("No nonlinear variant provided. Use: default or tuned.")

    invalid = sorted({variant for variant in parsed if variant not in _ALLOWED_VARIANTS})
    if invalid:
        raise ValueError(
            f"Invalid variants: {invalid}. Allowed variants: {sorted(_ALLOWED_VARIANTS)}"
        )
    return list(dict.fromkeys(parsed))


def _resolve_dataset_path(file_path: str | Path | None) -> Path:
    if file_path is None:
        return get_default_dataset_path()
    resolved = Path(file_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path not found: {resolved}")
    return resolved


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


def _validate_training_inputs(
    X_pairs: pd.DataFrame,
    y_train: pd.Series,
    ids: pd.Series,
) -> None:
    if X_pairs.empty:
        raise ValueError("X_pairs is empty.")
    if y_train.empty:
        raise ValueError("y_train is empty.")
    if ids.empty:
        raise ValueError("ids is empty.")
    if len(y_train) != len(ids):
        raise ValueError("y_train and ids length mismatch.")

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
    return {
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
        "notes": [
            "Operational threshold is fixed (0.30).",
            "Top-k is a batch/ranking policy used only when capacity cannot handle the fixed-threshold volume.",
            "Holdout is evaluation-only and never used to choose thresholds.",
        ],
    }


def _resolve_hgb_params(variant: str) -> dict[str, Any]:
    if variant == "default":
        return {
            "random_state": RANDOM_STATE,
            "max_iter": 300,
            "learning_rate": 0.1,
        }
    if variant == "tuned":
        return {
            "random_state": RANDOM_STATE,
            "max_iter": 600,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_samples_leaf": 20,
        }
    raise ValueError(f"Unsupported nonlinear variant: {variant}")


def _instantiate_hgb(
    hgb_cls: type,
    variant: str,
) -> tuple[Any, dict[str, Any], list[str]]:
    requested = _resolve_hgb_params(variant)
    signature = inspect.signature(hgb_cls.__init__)
    supported_keys = set(signature.parameters)
    filtered = {k: v for k, v in requested.items() if k in supported_keys}
    dropped = sorted(set(requested) - set(filtered))
    estimator = hgb_cls(**filtered)
    return estimator, filtered, dropped


def _build_metadata_payload(
    *,
    variant: str,
    resolved_params: dict[str, Any],
    dropped_params: list[str],
    year_t: int,
    year_t1: int,
    enable_feature_engineering: bool,
    enable_age_bucket: bool,
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
    class_imbalance_strategy: dict[str, Any],
    prediction_policy: dict[str, Any],
    evaluation_holdout_at_05: dict[str, Any] | None,
    evaluation_holdout_at_030: dict[str, Any] | None,
    evaluation_holdout_at_threshold_selected: dict[str, Any] | None,
    topk_holdout_summary: dict[str, Any] | None,
    cv_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    notes = [
        "operational default policy uses fixed threshold=0.30 for recall-focused mode",
        "threshold_calibration is train-only evidence and does not replace operational threshold",
        "holdout evaluation is read-only (no fitting on 2023->2024)",
    ]
    if dropped_params:
        notes.append(f"unsupported_params_ignored={dropped_params}")
    if evaluation_holdout_at_05 is None:
        notes.append("holdout evaluation disabled by flag")

    payload = {
        "model_kind": "HistGradientBoostingClassifier",
        "variant": variant,
        "resolved_params": resolved_params,
        "train_pair": {"year_t": year_t, "year_t1": year_t1},
        "scaler_strategy": "none",
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
        "evaluation_holdout_at_threshold_selected": evaluation_holdout_at_threshold_selected,
        "threshold_policy": threshold_policy,
        "threshold_calibration": threshold_calibration,
        "topk_holdout_summary": topk_holdout_summary,
        "class_imbalance_strategy": class_imbalance_strategy,
        "prediction_policy": prediction_policy,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "sklearn": sklearn_version,
        },
        "notes": notes,
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
) -> dict[str, Any]:
    train_prev = compute_prevalence(y_train)
    hold_prev = compute_prevalence(y_holdout) if y_holdout is not None else None
    by_variant_threshold_05: dict[str, Any] = {}
    for variant, train_scores in variant_scores_train.items():
        hold_scores = variant_scores_holdout.get(variant)
        by_variant_threshold_05[variant] = {
            "train": compute_metrics_threshold(y_train, train_scores, threshold=0.5),
            "holdout": (
                compute_metrics_threshold(y_holdout, hold_scores, threshold=0.5)
                if (y_holdout is not None and hold_scores is not None)
                else None
            ),
        }

    topk_on_holdout: dict[str, Any] | None = None
    reference_variant = next(
        (variant for variant, scores in variant_scores_holdout.items() if scores is not None),
        None,
    )
    if y_holdout is not None and reference_variant is not None:
        ref_scores = variant_scores_holdout[reference_variant]
        topk_on_holdout = {
            "reference_variant": reference_variant,
            "topk_10pct": compute_metrics_topk(y_holdout, ref_scores, k_frac=0.10),
            "topk_20pct": compute_metrics_topk(
                y_holdout,
                ref_scores,
                k_frac=_CAPACITY_TOPK_FRACTION,
            ),
        }

    notes = [
        "class_weight tuning is not applicable to HistGradientBoostingClassifier in this setup.",
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
        "class_weight_tested": [],
        "decision": (
            "class_weight is not used; use fixed threshold=0.30 for recall-focused operation; "
            "use top-k=20% when capacity is constrained."
        ),
        "evidence": {
            "by_variant_threshold_0.5": by_variant_threshold_05,
            "topk_holdout": topk_on_holdout,
        },
        "notes": notes,
    }


def run_hgb_training(
    *,
    file_path: str | Path | None = None,
    year_t: int = 2022,
    year_t1: int = 2023,
    out_dir: str | Path = "artifacts/models/nonlinear_hgb",
    variants: str = "default",
    enable_feature_engineering: bool = True,
    enable_age_bucket: bool = False,
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
    variants_list = _parse_variants(variants)
    resolved_dataset_path = _resolve_dataset_path(file_path)
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
    _validate_training_inputs(X_pairs, y_train, ids)

    raw_bundle = build_preprocessing_bundle(
        numeric_scaler="none",
        enable_feature_engineering=enable_feature_engineering,
        enable_age_bucket=enable_age_bucket,
    )
    expected_raw_cols = list(raw_bundle["expected_raw_cols"])
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

    pruning_hash = _hash_pruning_plan(feature_pruning_plan)
    dataset_basename = resolved_dataset_path.name
    dataset_sha256 = _compute_sha256(resolved_dataset_path)
    base_output_dir = Path(out_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    successes: dict[str, Any] = {}
    failures: dict[str, str] = {}
    variant_scores_train: dict[str, np.ndarray] = {}
    variant_scores_holdout: dict[str, np.ndarray | None] = {}
    metadata_payload_by_variant: dict[str, dict[str, Any]] = {}
    metadata_path_by_variant: dict[str, Path] = {}
    for variant in variants_list:
        try:
            estimator, resolved_params, dropped_params = _instantiate_hgb(
                deps["HistGradientBoostingClassifier"],
                variant,
            )
            pipeline = build_model_pipeline(
                model=estimator,
                year_t=year_t,
                scaler_strategy="none",
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
            metadata_path_by_variant[variant] = metadata_path

            metadata_payload_by_variant[variant] = {
                "variant": variant,
                "resolved_params": resolved_params,
                "dropped_params": dropped_params,
                "year_t": year_t,
                "year_t1": year_t1,
                "enable_feature_engineering": enable_feature_engineering,
                "enable_age_bucket": enable_age_bucket,
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
                "evaluation_holdout_at_threshold_selected": evaluation_holdout_at_threshold_selected,
                "topk_holdout_summary": topk_holdout_summary,
                "sklearn_version": str(deps["sklearn_version"]),
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
                "Nonlinear variant trained | pair=%s->%s variant=%s n=%d prevalence=%.4f",
                year_t,
                year_t1,
                variant,
                len(y_train),
                float(np.mean(y_train.astype(int).to_numpy())),
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            failures[variant] = str(exc)
            _logger.error("Nonlinear variant failed | variant=%s error=%s", variant, exc)
            if strict:
                raise

    if not successes:
        raise RuntimeError(f"No nonlinear variant trained successfully. Failures: {failures}")

    strategy_block = _build_class_imbalance_strategy(
        y_train=y_train,
        y_holdout=y_holdout,
        variant_scores_train=variant_scores_train,
        variant_scores_holdout=variant_scores_holdout,
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
            model_factory = (
                lambda v=variant: _instantiate_hgb(
                    deps["HistGradientBoostingClassifier"],
                    v,
                )[0]
            )
            cv_payload_by_variant[variant] = run_stratified_cv(
                model_name=f"hgb_{variant}",
                model_factory=model_factory,
                build_pipeline_fn=build_model_pipeline,
                X_raw_train=X_raw_train,
                y_train=y_train,
                year_t=year_t,
                feature_pruning_plan=feature_pruning_plan,
                scaler_strategy="none",
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
            variant=base_payload["variant"],
            resolved_params=base_payload["resolved_params"],
            dropped_params=base_payload["dropped_params"],
            year_t=base_payload["year_t"],
            year_t1=base_payload["year_t1"],
            enable_feature_engineering=base_payload["enable_feature_engineering"],
            enable_age_bucket=base_payload["enable_age_bucket"],
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
            class_imbalance_strategy=strategy_block,
            prediction_policy=prediction_policy,
            evaluation_holdout_at_05=base_payload["evaluation_holdout_at_05"],
            evaluation_holdout_at_030=base_payload["evaluation_holdout_at_030"],
            evaluation_holdout_at_threshold_selected=base_payload[
                "evaluation_holdout_at_threshold_selected"
            ],
            topk_holdout_summary=base_payload["topk_holdout_summary"],
            cv_result=cv_payload_by_variant.get(variant),
        )
        metadata_path_by_variant[variant].write_text(
            json.dumps(metadata_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "file_path": str(resolved_dataset_path),
        "out_dir": str(base_output_dir),
        "year_t": year_t,
        "year_t1": year_t1,
        "variants": successes,
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HistGradientBoostingClassifier on temporal pair t->t+1."
    )
    parser.add_argument(
        "--file-path",
        "--dataset-path",
        dest="file_path",
        type=str,
        default=None,
        help="Path to XLSX dataset.",
    )
    parser.add_argument("--year-t", type=int, default=2022, help="Training feature year.")
    parser.add_argument("--year-t1", type=int, default=2023, help="Target year.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/models/nonlinear_hgb",
        help="Output directory for nonlinear artifacts.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="default",
        help="Comma-separated variants: default,tuned",
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
        default=0,
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
        report = run_hgb_training(
            file_path=args.file_path,
            year_t=int(args.year_t),
            year_t1=int(args.year_t1),
            out_dir=args.out_dir,
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
        "Nonlinear training completed | pair=%s->%s variants=%s out_dir=%s",
        report["year_t"],
        report["year_t1"],
        sorted(report["variants"].keys()),
        report["out_dir"],
    )


if __name__ == "__main__":
    main()
