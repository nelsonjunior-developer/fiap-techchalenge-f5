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


def _build_raw_frame(expected_cols: list[str], n_rows: int = 10) -> pd.DataFrame:
    numeric_set = set(NUMERIC_COLS)
    data: dict[str, object] = {}
    for idx, col in enumerate(expected_cols):
        if col in numeric_set:
            values = np.linspace(1, n_rows, n_rows, dtype=float) + idx
            if n_rows > 3:
                values[3] = np.nan
            data[col] = pd.Series(values, dtype="Float64")
        else:
            values = ["A", "B", "C", "A", "B"] * ((n_rows // 5) + 1)
            series = pd.Series(values[:n_rows], dtype="string")
            if n_rows > 2:
                series.iloc[2] = pd.NA
            data[col] = series
    return pd.DataFrame(data).loc[:, expected_cols]


def test_pipeline_accepts_raw_api_contract_end_to_end() -> None:
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=False,
        enable_age_bucket=False,
    )
    expected_cols = list(bundle["expected_raw_cols"])
    X_raw = _build_raw_frame(expected_cols, n_rows=10)
    y = pd.Series(([0, 1] * 5)[: len(X_raw)], dtype="Int64")

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

    pipeline.fit(X_raw, y)
    probs = pipeline.predict_proba(X_raw)
    assert probs.shape == (len(X_raw), 2)

