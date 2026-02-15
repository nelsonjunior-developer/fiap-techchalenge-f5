import numpy as np
import pandas as pd
import pytest

from src.features import get_engineered_feature_names
from src.preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    _SKLEARN_AVAILABLE,
    build_preprocessing_bundle,
    get_expected_raw_feature_columns,
)


def _build_raw_frame(n_rows: int = 6) -> pd.DataFrame:
    raw_cols = get_expected_raw_feature_columns()
    data: dict[str, object] = {}

    for col in NUMERIC_COLS:
        values = np.linspace(1, n_rows, n_rows, dtype=float)
        if n_rows >= 3:
            values[2] = np.nan
        data[col] = pd.Series(values, dtype="Float64")

    for col in CATEGORICAL_COLS:
        values = ["A", "B", "A", "C", "B", "A"][:n_rows]
        series = pd.Series(values, dtype="string")
        if n_rows >= 4:
            series.iloc[3] = pd.NA
        data[col] = series

    return pd.DataFrame(data).loc[:, raw_cols]


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_bundle_expected_raw_and_model_columns() -> None:
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=True,
        enable_age_bucket=True,
    )
    expected_raw_cols = bundle["expected_raw_cols"]
    expected_model_cols = bundle["expected_model_cols"]
    engineered = get_engineered_feature_names(enable_age_bucket=True)["all"]

    assert set(engineered).isdisjoint(expected_raw_cols)
    assert set(engineered).issubset(set(expected_model_cols))


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_bundle_fit_transform_with_feature_engineering_enabled() -> None:
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=True,
        enable_age_bucket=True,
    )
    preprocessor = bundle["preprocessor"]
    to_model_frame = bundle["transform_raw_to_model_frame"]

    X_train_raw = _build_raw_frame()
    X_train_model, report = to_model_frame(X_train_raw, context="train")
    assert isinstance(report["features_added"], list)

    preprocessor.fit(X_train_model)

    X_inf_raw = _build_raw_frame(n_rows=3)
    X_inf_raw.loc[:, "Fase"] = pd.Series(["NOVA_FASE", "B", "A"], dtype="string")
    X_inf_model, _ = to_model_frame(X_inf_raw, context="inference")
    Xt = preprocessor.transform(X_inf_model)
    assert Xt.shape[0] == len(X_inf_raw)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_transform_raw_to_model_frame_blocks_suspicious_model_columns() -> None:
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=True,
        enable_age_bucket=True,
    )
    to_model_frame = bundle["transform_raw_to_model_frame"]
    raw_cols = bundle["expected_raw_cols"]
    model_cols = bundle["expected_model_cols"]

    # Force model-frame gate by injecting a suspicious model column and expecting it.
    if "my_feature_t1" not in model_cols:
        model_cols.append("my_feature_t1")
    X_raw = _build_raw_frame(n_rows=3).loc[:, raw_cols]
    X_raw["my_feature_t1"] = pd.Series([0.1, 0.2, 0.3], dtype="Float64")

    with pytest.raises(ValueError, match="Leakage detected"):
        to_model_frame(X_raw, context="inference")
