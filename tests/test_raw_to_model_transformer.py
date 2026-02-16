import numpy as np
import pandas as pd
import pytest

from src.pipeline_components import RawToModelFrameTransformer
from src.preprocessing import (
    NUMERIC_COLS,
    get_expected_raw_feature_columns,
    get_feature_columns_for_model,
)


def _build_raw_frame(n_rows: int = 6) -> pd.DataFrame:
    expected_raw_cols = get_expected_raw_feature_columns()
    data: dict[str, object] = {}
    numeric_set = set(NUMERIC_COLS)

    for idx, col in enumerate(expected_raw_cols):
        if col in numeric_set:
            values = np.linspace(1, n_rows, n_rows, dtype=float) + idx
            data[col] = pd.Series(values, dtype="Float64")
        else:
            values = ["A", "B", "A", "C", "B", "A"][:n_rows]
            data[col] = pd.Series(values, dtype="string")

    return pd.DataFrame(data).loc[:, expected_raw_cols]


def _build_transformer() -> RawToModelFrameTransformer:
    expected_raw_cols = get_expected_raw_feature_columns()
    expected_model_cols = get_feature_columns_for_model()
    return RawToModelFrameTransformer(
        year_t=2022,
        expected_raw_cols=expected_raw_cols,
        expected_model_cols=expected_model_cols,
        enable_feature_engineering=False,
        feature_pruning_plan=None,
        strict_raw=True,
        enable_age_bucket=False,
    )


def test_transformer_requires_dataframe_in_fit_and_transform() -> None:
    transformer = _build_transformer()
    with pytest.raises(TypeError, match="Expected pandas.DataFrame"):
        transformer.fit({"x": 1})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Expected pandas.DataFrame"):
        transformer.transform({"x": 1})  # type: ignore[arg-type]


def test_transformer_allows_non_suspicious_extra_column() -> None:
    transformer = _build_transformer()
    X_raw = _build_raw_frame()
    X_raw["extra_ok"] = pd.Series(["ok"] * len(X_raw), dtype="string")
    X_model = transformer.fit(X_raw).transform(X_raw)
    assert isinstance(X_model, pd.DataFrame)
    assert list(X_model.columns) == transformer.expected_model_cols


def test_transformer_blocks_suspicious_extra_column() -> None:
    transformer = _build_transformer()
    X_raw = _build_raw_frame()
    X_raw["Defasagem_t1"] = pd.Series(
        np.linspace(-1, 1, len(X_raw)),
        dtype="Float64",
    )
    with pytest.raises(ValueError, match="(?i)leakage-like"):
        transformer.transform(X_raw)


def test_transformer_returns_exact_expected_model_columns() -> None:
    transformer = _build_transformer()
    X_raw = _build_raw_frame()
    X_model = transformer.transform(X_raw)
    assert list(X_model.columns) == transformer.expected_model_cols
