from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    NUMERIC_COLS,
    build_preprocessing_bundle,
    build_pruning_plan_from_training_frame,
)
from src.train_pipeline import build_model_pipeline


def _build_raw_frame(expected_raw_cols: list[str], n_rows: int = 12) -> pd.DataFrame:
    data: dict[str, object] = {}
    numeric_set = set(NUMERIC_COLS)
    for idx, col in enumerate(expected_raw_cols):
        if col in numeric_set:
            values = np.linspace(1, n_rows, n_rows, dtype=float) + idx
            if n_rows > 4:
                values[4] = np.nan
            data[col] = pd.Series(values, dtype="Float64")
        else:
            values = ["A", "B", "C", "A", "B", "C"] * ((n_rows // 6) + 1)
            series = pd.Series(values[:n_rows], dtype="string")
            if n_rows > 3:
                series.iloc[3] = pd.NA
            data[col] = series
    return pd.DataFrame(data).loc[:, expected_raw_cols]


def _build_binary_target(n_rows: int) -> pd.Series:
    values = np.array([0, 1] * (n_rows // 2 + 1))[:n_rows]
    return pd.Series(values, dtype="Int64")


def _build_pipeline_and_data(n_rows: int = 12) -> tuple[object, pd.DataFrame, pd.Series]:
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=False,
        enable_age_bucket=False,
    )
    expected_raw_cols = list(bundle["expected_raw_cols"])
    X_raw = _build_raw_frame(expected_raw_cols, n_rows=n_rows)
    y = _build_binary_target(len(X_raw))
    pruning_plan = build_pruning_plan_from_training_frame(
        X_train_raw=X_raw,
        enable_feature_engineering=False,
        enable_age_bucket=False,
    )
    pipeline = build_model_pipeline(
        model=LogisticRegression(max_iter=200),
        year_t=2022,
        scaler_strategy="standard",
        enable_feature_engineering=False,
        feature_pruning_plan=pruning_plan,
        strict_raw=True,
        enable_age_bucket=False,
    )
    return pipeline, X_raw, y


def test_train_inference_same_model_cols_after_fit() -> None:
    pipeline, X_raw, y = _build_pipeline_and_data()
    pipeline.fit(X_raw, y)

    raw_to_model = pipeline.named_steps["raw_to_model"]
    X_model_1 = raw_to_model.transform(X_raw.head(5))
    X_model_2 = raw_to_model.transform(X_raw.head(5).assign(extra_ok=1))

    assert list(X_model_1.columns) == list(raw_to_model.expected_model_cols_)
    assert list(X_model_2.columns) == list(raw_to_model.expected_model_cols_)
    assert hasattr(raw_to_model, "feature_pruning_plan_")
    assert hasattr(raw_to_model, "consistency_report_")


def test_pipeline_predict_proba_smoke_from_raw() -> None:
    pipeline, X_raw, y = _build_pipeline_and_data()
    pipeline.fit(X_raw, y)
    probs = pipeline.predict_proba(X_raw.iloc[:4])
    assert probs.shape == (4, 2)


def test_missing_raw_column_fails_with_clear_error() -> None:
    pipeline, X_raw, y = _build_pipeline_and_data()
    pipeline.fit(X_raw, y)
    missing_col = X_raw.columns[0]
    X_infer = X_raw.drop(columns=[missing_col])
    with pytest.raises(ValueError, match=missing_col):
        pipeline.predict_proba(X_infer)


def test_joblib_roundtrip_pipeline_smoke(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    pipeline, X_raw, y = _build_pipeline_and_data()
    pipeline.fit(X_raw, y)

    output_path = tmp_path / "train_inference_pipeline.joblib"
    joblib.dump(pipeline, output_path)
    loaded = joblib.load(output_path)
    probs = loaded.predict_proba(X_raw.iloc[:3])
    assert probs.shape == (3, 2)


def test_engineered_features_are_present_in_bundle_contract() -> None:
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=False,
        enable_age_bucket=False,
    )
    expected_raw_cols = list(bundle["expected_raw_cols"])
    X_raw = _build_raw_frame(expected_raw_cols, n_rows=12)
    pruning_plan = build_pruning_plan_from_training_frame(
        X_train_raw=X_raw,
        enable_feature_engineering=True,
        enable_age_bucket=True,
    )

    pipeline = build_model_pipeline(
        model=LogisticRegression(max_iter=200),
        year_t=2022,
        scaler_strategy="standard",
        enable_feature_engineering=True,
        feature_pruning_plan=pruning_plan,
        strict_raw=True,
        enable_age_bucket=True,
    )
    pipeline.fit(X_raw, _build_binary_target(len(X_raw)))
    raw_to_model = pipeline.named_steps["raw_to_model"]
    expected_model_cols = set(raw_to_model.expected_model_cols_)
    preprocessing_spec = raw_to_model.preprocessing_spec_
    numeric_used = set(preprocessing_spec.get("numeric_cols_used", []))

    # Guardrail: engineered features must remain in the final model contract when enabled.
    for engineered_col in {"avg_grades", "defasagem_abs"}:
        assert engineered_col in expected_model_cols
        assert engineered_col in numeric_used
