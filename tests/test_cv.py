from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_STATE
from src.cv import run_stratified_cv
from src.preprocessing import (
    NUMERIC_COLS,
    build_pruning_plan_from_training_frame,
    get_expected_raw_feature_columns,
)
from src.train_pipeline import build_model_pipeline

sklearn = pytest.importorskip("sklearn")
LogisticRegression = sklearn.linear_model.LogisticRegression


def _build_raw_train_frame(n_rows: int = 30) -> pd.DataFrame:
    expected_raw_cols = get_expected_raw_feature_columns()
    numeric_cols = set(NUMERIC_COLS)
    payload: dict[str, Any] = {}
    for idx, col in enumerate(expected_raw_cols):
        if col in numeric_cols:
            values = np.linspace(1.0, float(n_rows), n_rows) + idx
            if col == "Ing":
                values[::6] = np.nan
            payload[col] = pd.Series(values, dtype="Float64")
        else:
            cats = (["A", "B", "C"] * ((n_rows // 3) + 1))[:n_rows]
            payload[col] = pd.Series(cats, dtype="string")
    return pd.DataFrame(payload)


def _collect_dict_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys |= _collect_dict_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_dict_keys(item)
    return keys


def test_run_stratified_cv_smoke_returns_expected_shape() -> None:
    X_raw_train = _build_raw_train_frame()
    y_train = pd.Series(([0, 1] * 15), dtype="Int64")
    feature_pruning_plan = build_pruning_plan_from_training_frame(
        X_train_raw=X_raw_train,
        enable_feature_engineering=True,
        enable_age_bucket=False,
    )

    result = run_stratified_cv(
        model_name="logreg_none",
        model_factory=lambda: LogisticRegression(max_iter=500, solver="lbfgs"),
        build_pipeline_fn=build_model_pipeline,
        X_raw_train=X_raw_train,
        y_train=y_train,
        year_t=2022,
        feature_pruning_plan=feature_pruning_plan,
        scaler_strategy="standard",
        enable_feature_engineering=True,
        enable_age_bucket=False,
        strict_raw=True,
        n_splits=5,
        repeat_n=1,
        random_state=RANDOM_STATE,
    )

    assert set(result.keys()) >= {
        "config",
        "n_samples",
        "n_pos",
        "prevalence",
        "folds",
        "metrics_cv_mean",
        "metrics_cv_std",
    }
    assert len(result["folds"]) == 5
    assert result["n_samples"] == len(y_train)


def test_run_stratified_cv_reproducible_with_fixed_seed() -> None:
    X_raw_train = _build_raw_train_frame()
    y_train = pd.Series(([0, 1] * 15), dtype="Int64")
    feature_pruning_plan = build_pruning_plan_from_training_frame(
        X_train_raw=X_raw_train,
        enable_feature_engineering=False,
        enable_age_bucket=False,
    )

    kwargs = dict(
        model_name="logreg_none",
        model_factory=lambda: LogisticRegression(max_iter=500, solver="lbfgs"),
        build_pipeline_fn=build_model_pipeline,
        X_raw_train=X_raw_train,
        y_train=y_train,
        year_t=2022,
        feature_pruning_plan=feature_pruning_plan,
        scaler_strategy="standard",
        enable_feature_engineering=False,
        enable_age_bucket=False,
        strict_raw=True,
        n_splits=5,
        repeat_n=1,
        random_state=RANDOM_STATE,
    )
    result_a = run_stratified_cv(**kwargs)
    result_b = run_stratified_cv(**kwargs)

    for metric, value_a in result_a["metrics_cv_mean"].items():
        value_b = result_b["metrics_cv_mean"][metric]
        if value_a is None or value_b is None:
            assert value_a is value_b
        else:
            assert value_a == pytest.approx(value_b, rel=0.0, abs=1e-12)


def test_run_stratified_cv_privacy_no_identifier_keys() -> None:
    X_raw_train = _build_raw_train_frame()
    y_train = pd.Series(([0, 1] * 15), dtype="Int64")
    feature_pruning_plan = build_pruning_plan_from_training_frame(
        X_train_raw=X_raw_train,
        enable_feature_engineering=False,
        enable_age_bucket=False,
    )

    result = run_stratified_cv(
        model_name="logreg_none",
        model_factory=lambda: LogisticRegression(max_iter=500, solver="lbfgs"),
        build_pipeline_fn=build_model_pipeline,
        X_raw_train=X_raw_train,
        y_train=y_train,
        year_t=2022,
        feature_pruning_plan=feature_pruning_plan,
        scaler_strategy="standard",
        enable_feature_engineering=False,
        enable_age_bucket=False,
        strict_raw=True,
        n_splits=5,
        repeat_n=1,
        random_state=RANDOM_STATE,
    )
    keys = _collect_dict_keys(result)
    forbidden = {"ra", "ids", "ra_list", "students", "student_ids", "rows"}
    assert forbidden.isdisjoint(keys)


def test_run_stratified_cv_requires_pruning_plan() -> None:
    X_raw_train = _build_raw_train_frame()
    y_train = pd.Series(([0, 1] * 15), dtype="Int64")

    with pytest.raises(ValueError, match="feature_pruning_plan is required"):
        run_stratified_cv(
            model_name="logreg_none",
            model_factory=lambda: LogisticRegression(max_iter=500, solver="lbfgs"),
            build_pipeline_fn=build_model_pipeline,
            X_raw_train=X_raw_train,
            y_train=y_train,
            year_t=2022,
            feature_pruning_plan=None,
            scaler_strategy="standard",
            enable_feature_engineering=False,
            enable_age_bucket=False,
            strict_raw=True,
            n_splits=5,
            repeat_n=1,
            random_state=RANDOM_STATE,
        )
