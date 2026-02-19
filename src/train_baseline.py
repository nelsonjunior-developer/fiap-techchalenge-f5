"""CLI and helpers to train LogisticRegression baseline on temporal pair 2022->2023."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from src.train_pipeline import build_model_pipeline
from src.training_policy import OFFICIAL_TRAIN_PAIR, enforce_official_train_pair
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_ALLOWED_SCALERS = {"standard", "robust", "none"}
_ALLOWED_VARIANTS = {"none", "balanced"}
_FORBIDDEN_METADATA_KEYS = {"ids", "ra_list", "students", "rows"}


def _require_training_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "scikit-learn is required to train baseline. Install requirements-dev.txt"
        ) from exc

    return {
        "joblib": joblib,
        "sklearn_version": sklearn.__version__,
        "LogisticRegression": LogisticRegression,
        "average_precision_score": average_precision_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
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


def _build_raw_from_ids(
    df_t: pd.DataFrame,
    ids: pd.Series,
    expected_raw_cols: list[str],
) -> pd.DataFrame:
    ids_df = pd.DataFrame({"RA": ids.astype("string")})
    raw_df = ids_df.merge(df_t, on="RA", how="left")
    missing_cols = sorted(set(expected_raw_cols) - set(raw_df.columns))
    if missing_cols:
        raise ValueError(f"Raw training frame missing expected columns: {missing_cols}")
    return raw_df.loc[:, expected_raw_cols].copy()


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


def _safe_metric(metric_fn: Callable[..., float], *args: Any, **kwargs: Any) -> float | None:
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return None


def _compute_probability_summary(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(scores)),
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
        "p05": float(np.quantile(scores, 0.05)),
        "p50": float(np.quantile(scores, 0.50)),
        "p95": float(np.quantile(scores, 0.95)),
    }


def _build_metadata_payload(
    *,
    variant: str,
    class_weight: str | None,
    year_t: int,
    year_t1: int,
    scaler_strategy: str,
    enable_feature_engineering: bool,
    enable_age_bucket: bool,
    feature_pruning_plan_hash: str,
    dataset_basename: str | None,
    dataset_sha256: str | None,
    n_samples_train: int,
    y_prevalence: float,
    probability_summary: dict[str, float],
    metrics_train_at_05: dict[str, float | None],
    sklearn_version: str,
    cv_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_kind": "LogisticRegression",
        "variant": variant,
        "class_weight": class_weight,
        "train_pair": {"year_t": year_t, "year_t1": year_t1},
        "scaler_strategy": scaler_strategy,
        "enable_feature_engineering": bool(enable_feature_engineering),
        "enable_age_bucket": bool(enable_age_bucket),
        "feature_pruning_plan_hash": feature_pruning_plan_hash,
        "dataset_basename": dataset_basename,
        "dataset_sha256": dataset_sha256,
        "n_samples_train": n_samples_train,
        "y_prevalence": y_prevalence,
        "train_pred_proba_summary": probability_summary,
        "metrics_train_at_0.5": metrics_train_at_05,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "sklearn": sklearn_version,
        },
        "notes": [
            "train-only metrics (no temporal holdout here)",
            "threshold tuning is a later task",
        ],
    }
    forbidden_present = _FORBIDDEN_METADATA_KEYS & set(payload.keys())
    if forbidden_present:
        raise ValueError(f"Forbidden metadata keys found: {sorted(forbidden_present)}")
    if cv_result is not None:
        payload["cv"] = cv_result
    return payload


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
    X_raw_train = _build_raw_from_ids(yearly_frames[year_t], ids, expected_raw_cols)
    if len(X_raw_train) != len(y_train):
        raise ValueError("Inconsistent training rows between X_raw_train and y_train.")

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

    for variant in variants_list:
        class_weight: str | None = None if variant == "none" else "balanced"
        try:
            def _model_factory() -> Any:
                return deps["LogisticRegression"](
                    class_weight=class_weight,
                    max_iter=2000,
                    solver="lbfgs",
                )

            cv_result: dict[str, Any] | None = None
            if enable_cv:
                cv_result = run_stratified_cv(
                    model_name=f"logreg_{variant}",
                    model_factory=_model_factory,
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
            labels = (scores >= 0.5).astype(int)
            y_true = y_train.astype(int).to_numpy()
            probability_summary = _compute_probability_summary(scores)
            metrics_at_05 = {
                "recall": _safe_metric(
                    deps["recall_score"], y_true, labels, zero_division=0
                ),
                "precision": _safe_metric(
                    deps["precision_score"], y_true, labels, zero_division=0
                ),
                "f1": _safe_metric(deps["f1_score"], y_true, labels, zero_division=0),
                "roc_auc": _safe_metric(deps["roc_auc_score"], y_true, scores),
                "pr_auc": _safe_metric(
                    deps["average_precision_score"], y_true, scores
                ),
                "positive_rate_at_0.5": float(np.mean(labels)),
            }

            variant_dir = base_output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            model_path = variant_dir / "model.joblib"
            metadata_path = variant_dir / "metadata.json"
            deps["joblib"].dump(pipeline, model_path)

            metadata_payload = _build_metadata_payload(
                variant=variant,
                class_weight=class_weight,
                year_t=year_t,
                year_t1=year_t1,
                scaler_strategy=scaler_strategy,
                enable_feature_engineering=enable_feature_engineering,
                enable_age_bucket=enable_age_bucket,
                feature_pruning_plan_hash=pruning_hash,
                dataset_basename=dataset_basename,
                dataset_sha256=dataset_sha256,
                n_samples_train=int(len(y_train)),
                y_prevalence=float(np.mean(y_true)),
                probability_summary=probability_summary,
                metrics_train_at_05=metrics_at_05,
                sklearn_version=str(deps["sklearn_version"]),
                cv_result=cv_result,
            )
            metadata_path.write_text(
                json.dumps(metadata_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            successes[variant] = {
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "n_samples_train": int(len(y_train)),
                "y_prevalence": float(np.mean(y_true)),
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
                float(np.mean(y_true)),
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
