import numpy as np
import pandas as pd
import pytest

from src.contracts import PII_COLUMNS
from src.preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    _SKLEARN_AVAILABLE,
    build_preprocessing_bundle,
    get_excluded_columns,
    get_expected_raw_feature_columns,
    get_feature_columns_for_model,
    validate_inference_frame,
)


def _build_expected_frame(n_rows: int = 6) -> pd.DataFrame:
    expected_cols = get_expected_raw_feature_columns()
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

    frame = pd.DataFrame(data)
    return frame.loc[:, expected_cols]


def test_expected_cols_reuses_model_feature_fn() -> None:
    expected_cols = get_expected_raw_feature_columns()
    model_cols = get_feature_columns_for_model()
    assert expected_cols == model_cols


def test_excluded_cols_from_pii() -> None:
    excluded = get_excluded_columns()
    assert excluded == sorted(list(PII_COLUMNS))


def test_validate_inference_requires_dataframe() -> None:
    with pytest.raises(TypeError, match=r"\[api\] Expected pandas\.DataFrame"):
        validate_inference_frame({"a": 1}, context="api")  # type: ignore[arg-type]


def test_validate_inference_missing_raises() -> None:
    expected_cols = get_expected_raw_feature_columns()
    missing_col = expected_cols[0]
    X = pd.DataFrame({col: [1] for col in expected_cols[1:]})

    with pytest.raises(ValueError, match=r"\[inference\] Missing expected columns"):
        validate_inference_frame(X, expected_cols=expected_cols)
    with pytest.raises(ValueError, match=missing_col):
        validate_inference_frame(X, expected_cols=expected_cols)


def test_validate_inference_allows_extras() -> None:
    expected_cols = get_expected_raw_feature_columns()
    X = pd.DataFrame({col: [1] for col in expected_cols})
    X["extra_col"] = 123
    validate_inference_frame(X, expected_cols=expected_cols)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_bundle_keys_and_no_pii_in_expected() -> None:
    bundle = build_preprocessing_bundle(numeric_scaler="standard")
    expected_subset = {
        "expected_cols",
        "expected_raw_cols",
        "expected_model_cols",
        "excluded_cols",
        "numeric_scaler",
        "preprocessor",
        "enable_feature_engineering",
        "enable_age_bucket",
        "transform_raw_to_model_frame",
    }
    assert expected_subset.issubset(set(bundle.keys()))
    assert not (set(bundle["expected_raw_cols"]) & set(bundle["excluded_cols"]))


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_preprocessor_fit_transform_inference_schema() -> None:
    expected_cols = get_expected_raw_feature_columns()
    n_expected = len(NUMERIC_COLS) + len(CATEGORICAL_COLS)
    assert len(expected_cols) == n_expected

    X_train = _build_expected_frame()
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=True,
        enable_age_bucket=True,
    )
    preprocessor = bundle["preprocessor"]
    transform_raw_to_model_frame = bundle["transform_raw_to_model_frame"]
    X_train_model, _ = transform_raw_to_model_frame(X_train, context="train")
    preprocessor.fit(X_train_model)

    X_inf = _build_expected_frame(n_rows=3)
    X_inf.loc[:, "Fase"] = pd.Series(["NOVA_FASE", "B", "A"], dtype="string")
    X_inf_model, _ = transform_raw_to_model_frame(X_inf, context="inference")
    transformed = preprocessor.transform(X_inf_model)
    assert transformed.shape[0] == len(X_inf)
