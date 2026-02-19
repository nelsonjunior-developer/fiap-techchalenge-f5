"""Cross-validation helpers for internal stratified validation on train pair."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE


def _require_cv_dependencies() -> dict[str, Any]:
    try:
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "scikit-learn is required to run stratified CV. Install requirements-dev.txt"
        ) from exc

    return {
        "StratifiedKFold": StratifiedKFold,
        "RepeatedStratifiedKFold": RepeatedStratifiedKFold,
        "average_precision_score": average_precision_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def _safe_metric(metric_fn: Callable[..., float], *args: Any, **kwargs: Any) -> float | None:
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return None


def build_cv_splitter(
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
    repeat_n: int = 1,
) -> Any:
    """Build stratified splitter (single or repeated) with deterministic seed."""
    if int(n_splits) < 2:
        raise ValueError("n_splits must be >= 2.")
    if int(repeat_n) < 1:
        raise ValueError("repeat_n must be >= 1.")

    deps = _require_cv_dependencies()
    if int(repeat_n) == 1:
        return deps["StratifiedKFold"](
            n_splits=int(n_splits),
            shuffle=bool(shuffle),
            random_state=int(random_state),
        )

    return deps["RepeatedStratifiedKFold"](
        n_splits=int(n_splits),
        n_repeats=int(repeat_n),
        random_state=int(random_state),
    )


def _aggregate_fold_metrics(
    folds: list[dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    metric_keys = [
        "recall",
        "precision",
        "f1",
        "roc_auc",
        "pr_auc",
        "positive_rate_at_0.5",
    ]
    mean_payload: dict[str, float | None] = {}
    std_payload: dict[str, float | None] = {}
    for key in metric_keys:
        raw_values = [fold["metrics_at_0.5"].get(key) for fold in folds]
        values = np.array(
            [np.nan if value is None else float(value) for value in raw_values],
            dtype=float,
        )
        if np.all(np.isnan(values)):
            mean_payload[key] = None
            std_payload[key] = None
            continue
        mean_payload[key] = float(np.nanmean(values))
        std_payload[key] = float(np.nanstd(values))
    return mean_payload, std_payload


def run_stratified_cv(
    *,
    model_name: str,
    model_factory: Callable[[], Any],
    build_pipeline_fn: Callable[..., Any],
    X_raw_train: pd.DataFrame,
    y_train: pd.Series,
    year_t: int,
    feature_pruning_plan: dict[str, Any] | None,
    scaler_strategy: str,
    enable_feature_engineering: bool,
    enable_age_bucket: bool,
    strict_raw: bool = True,
    n_splits: int = 5,
    repeat_n: int = 1,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Run stratified CV on raw train frame using end-to-end pipeline per fold."""
    if feature_pruning_plan is None:
        raise ValueError("feature_pruning_plan is required to run stratified CV.")
    if not isinstance(X_raw_train, pd.DataFrame):
        raise TypeError(f"X_raw_train must be pandas.DataFrame, got {type(X_raw_train)}")

    y_series = pd.Series(y_train).reset_index(drop=True)
    if len(X_raw_train) != len(y_series):
        raise ValueError("X_raw_train and y_train must have the same number of rows.")
    if len(X_raw_train) == 0:
        raise ValueError("X_raw_train is empty.")

    unique_values = sorted(set(y_series.dropna().astype(int).unique().tolist()))
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(f"y_train must be binary with values in {{0,1}}. Got: {unique_values}")
    if len(unique_values) < 2:
        raise ValueError("y_train must contain both classes for stratified CV.")

    min_class_count = int(y_series.value_counts().min())
    if min_class_count < int(n_splits):
        raise ValueError(
            "Not enough samples in minority class for requested n_splits: "
            f"min_class_count={min_class_count}, n_splits={int(n_splits)}"
        )

    deps = _require_cv_dependencies()
    splitter = build_cv_splitter(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(random_state),
        repeat_n=int(repeat_n),
    )

    folds: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(X_raw_train, y_series), start=1
    ):
        X_tr = X_raw_train.iloc[train_idx].copy()
        y_tr = y_series.iloc[train_idx].copy()
        X_va = X_raw_train.iloc[val_idx].copy()
        y_va = y_series.iloc[val_idx].copy()

        pipeline = build_pipeline_fn(
            model=model_factory(),
            year_t=int(year_t),
            scaler_strategy=scaler_strategy,
            enable_feature_engineering=bool(enable_feature_engineering),
            feature_pruning_plan=feature_pruning_plan,
            strict_raw=bool(strict_raw),
            enable_age_bucket=bool(enable_age_bucket),
        )
        pipeline.fit(X_tr, y_tr)

        pred_proba = pipeline.predict_proba(X_va)[:, 1]
        pred_label = (pred_proba >= 0.5).astype(int)
        y_true = y_va.astype(int).to_numpy()
        metrics_at_05 = {
            "recall": _safe_metric(
                deps["recall_score"], y_true, pred_label, zero_division=0
            ),
            "precision": _safe_metric(
                deps["precision_score"], y_true, pred_label, zero_division=0
            ),
            "f1": _safe_metric(
                deps["f1_score"], y_true, pred_label, zero_division=0
            ),
            "roc_auc": _safe_metric(deps["roc_auc_score"], y_true, pred_proba),
            "pr_auc": _safe_metric(
                deps["average_precision_score"], y_true, pred_proba
            ),
            "positive_rate_at_0.5": float(np.mean(pred_label)),
        }
        folds.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "metrics_at_0.5": metrics_at_05,
            }
        )

    metrics_mean, metrics_std = _aggregate_fold_metrics(folds)
    y_array = y_series.astype(int).to_numpy()
    return {
        "config": {
            "model_name": str(model_name),
            "n_splits": int(n_splits),
            "repeat_n": int(repeat_n),
            "random_state": int(random_state),
            "threshold": 0.5,
        },
        "n_samples": int(len(y_array)),
        "n_pos": int(np.sum(y_array == 1)),
        "prevalence": float(np.mean(y_array)),
        "folds": folds,
        "metrics_cv_mean": metrics_mean,
        "metrics_cv_std": metrics_std,
        "notes": [
            "cv runs only inside official train pair (2022->2023)",
            (
                "feature_pruning_plan is fitted once on full train frame and reused "
                "across folds to preserve train/inference contract"
            ),
        ],
    }
