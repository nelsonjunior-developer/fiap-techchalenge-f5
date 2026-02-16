import numpy as np
import pandas as pd
import pytest

import src.preprocessing as prep

pytestmark = pytest.mark.skipif(
    not prep._SKLEARN_AVAILABLE,
    reason="scikit-learn não disponível no ambiente",
)


def _build_raw_frame(n_rows: int = 4) -> pd.DataFrame:
    raw_cols = prep.get_expected_raw_feature_columns()
    data: dict[str, object] = {}

    for idx, col in enumerate(prep.NUMERIC_COLS):
        values = np.linspace(1, n_rows, n_rows, dtype=float) + idx
        data[col] = pd.Series(values, dtype="Float64")

    for col in prep.CATEGORICAL_COLS:
        values = ["A", "B", "A", "C"][:n_rows]
        data[col] = pd.Series(values, dtype="string")

    return pd.DataFrame(data).loc[:, raw_cols]


def test_preprocessing_spec_contains_expected_keys() -> None:
    bundle = prep.build_preprocessing_bundle(enable_feature_engineering=False)
    spec = bundle["preprocessing_spec"]
    assert set(spec.keys()) == {
        "numeric_cols_used",
        "categorical_cols_used",
        "datetime_cols_excluded",
        "scaler_strategy",
        "ohe_sparse_flag_used",
        "notes",
    }
    assert spec["ohe_sparse_flag_used"] in {"sparse_output", "sparse"}


def test_ohe_fallback_uses_sparse_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOneHotEncoder:
        def __init__(self, **kwargs: object) -> None:
            if "sparse_output" in kwargs:
                raise TypeError("unexpected sparse_output")
            self.kwargs = kwargs
            self.handle_unknown = kwargs.get("handle_unknown")

        def fit(self, X: pd.DataFrame, y: object = None) -> "FakeOneHotEncoder":
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            return X

    monkeypatch.setattr(prep, "OneHotEncoder", FakeOneHotEncoder)
    bundle = prep.build_preprocessing_bundle(enable_feature_engineering=False)
    assert bundle["preprocessing_spec"]["ohe_sparse_flag_used"] == "sparse"


def test_overlap_numeric_categorical_raises_clear_error() -> None:
    with pytest.raises(ValueError, match=r"overlap.*Mat"):
        prep.build_preprocessor(
            numeric_cols=["Mat"],
            categorical_cols=["Mat"],
            enable_feature_engineering=False,
        )


def test_missing_model_columns_reports_missing_names() -> None:
    fake_plan = {
        "kept_numeric_cols": ["Ano ingresso", "ghost_feature"],
        "kept_categorical_cols": [],
        "kept_model_cols": ["Ano ingresso", "ghost_feature"],
    }
    bundle = prep.build_preprocessing_bundle(
        enable_feature_engineering=False,
        feature_pruning_plan=fake_plan,
    )
    X_raw = _build_raw_frame()
    with pytest.raises(ValueError, match=r"Missing expected model columns: \['ghost_feature'\]"):
        bundle["transform_raw_to_model_frame"](X_raw, context="inference")
