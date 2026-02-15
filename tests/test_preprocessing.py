import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_COLS,
    DATETIME_COLS,
    DEFAULT_SCALER_FOR_LINEAR,
    DEFAULT_SCALER_FOR_TREE,
    NUMERIC_COLS,
    _SKLEARN_AVAILABLE,
    assert_no_pii_in_features,
    build_preprocessor,
    get_feature_columns_for_model,
)

if _SKLEARN_AVAILABLE:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler, StandardScaler
else:  # pragma: no cover - import path depends on runtime environment
    ColumnTransformer = None  # type: ignore[assignment]
    RobustScaler = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]


def _build_synthetic_frame(n_rows: int = 6) -> pd.DataFrame:
    data: dict[str, object] = {}

    for idx, col in enumerate(NUMERIC_COLS):
        values = np.linspace(1, n_rows, n_rows, dtype=float) + idx
        if n_rows >= 3:
            values[2] = np.nan
        data[col] = pd.Series(values, dtype="Float64")

    for col in CATEGORICAL_COLS:
        values = ["A", "B", "A", "C", "B", "A"][:n_rows]
        series = pd.Series(values, dtype="string")
        if n_rows >= 4:
            series.iloc[3] = pd.NA
        data[col] = series

    # Present in raw dataframe but excluded from this phase processing.
    data["Data_Nasc"] = pd.to_datetime(["2010-01-01"] * n_rows)
    return pd.DataFrame(data)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_build_preprocessor_types() -> None:
    preprocessor = build_preprocessor(
        numeric_scaler=DEFAULT_SCALER_FOR_LINEAR,
        enable_feature_engineering=False,
    )

    assert isinstance(preprocessor, ColumnTransformer)
    names = [name for name, _, _ in preprocessor.transformers]
    assert names == ["num", "cat"]
    num_pipeline = preprocessor.transformers[0][1]
    assert "scaler" in num_pipeline.named_steps
    assert isinstance(num_pipeline.named_steps["scaler"], StandardScaler)

    cat_pipeline = preprocessor.named_transformers_["cat"] if hasattr(
        preprocessor, "named_transformers_"
    ) else preprocessor.transformers[1][1]
    onehot = cat_pipeline.named_steps["onehot"]
    assert onehot.handle_unknown == "ignore"


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_preprocessor_fit_transform_shape_stable() -> None:
    X_train = _build_synthetic_frame()
    model_cols = get_feature_columns_for_model()

    preprocessor = build_preprocessor(
        numeric_scaler=DEFAULT_SCALER_FOR_LINEAR,
        enable_feature_engineering=False,
    )
    Xt = preprocessor.fit_transform(X_train[model_cols])

    assert Xt.ndim == 2
    assert Xt.shape[0] == len(X_train)
    assert Xt.shape[1] > len(model_cols)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_handle_unknown_does_not_break() -> None:
    X_train = _build_synthetic_frame()
    X_test = _build_synthetic_frame(n_rows=3)
    X_test.loc[:, "Fase"] = pd.Series(["NOVA_FASE", "B", "A"], dtype="string")

    model_cols = get_feature_columns_for_model()
    preprocessor = build_preprocessor(
        numeric_scaler=DEFAULT_SCALER_FOR_LINEAR,
        enable_feature_engineering=False,
    )
    preprocessor.fit(X_train[model_cols])

    Xt = preprocessor.transform(X_test[model_cols])
    assert Xt.shape[0] == len(X_test)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_no_datetime_in_features() -> None:
    model_cols = get_feature_columns_for_model()

    assert set(DATETIME_COLS).isdisjoint(model_cols)
    preprocessor = build_preprocessor(
        numeric_scaler=DEFAULT_SCALER_FOR_LINEAR,
        enable_feature_engineering=False,
    )
    used_cols = set(preprocessor.transformers[0][2]) | set(preprocessor.transformers[1][2])
    assert "Data_Nasc" not in used_cols


def test_no_pii_in_features_uses_source_of_truth() -> None:
    model_cols = get_feature_columns_for_model()
    assert_no_pii_in_features(model_cols)

    with pytest.raises(ValueError, match="PII"):
        assert_no_pii_in_features(model_cols + ["Nome_Anon"])


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_disable_numeric_scaling_removes_scaler_step() -> None:
    preprocessor = build_preprocessor(
        numeric_scaler="none",
        enable_feature_engineering=False,
    )
    num_pipeline = preprocessor.transformers[0][1]
    assert "scaler" not in num_pipeline.named_steps


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_build_preprocessor_uses_robust_scaler_when_requested() -> None:
    preprocessor = build_preprocessor(
        numeric_scaler="robust",
        enable_feature_engineering=False,
    )
    num_pipeline = preprocessor.transformers[0][1]
    assert isinstance(num_pipeline.named_steps["scaler"], RobustScaler)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_build_preprocessor_tree_default_has_no_scaler() -> None:
    preprocessor = build_preprocessor(enable_feature_engineering=False)
    num_pipeline = preprocessor.transformers[0][1]
    assert DEFAULT_SCALER_FOR_TREE == "none"
    assert "scaler" not in num_pipeline.named_steps


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_build_preprocessor_invalid_scaler_raises_error() -> None:
    with pytest.raises(ValueError, match="Escalonador inválido"):
        build_preprocessor(numeric_scaler="foobar")


def test_build_preprocessor_raises_clear_error_when_sklearn_missing() -> None:
    if _SKLEARN_AVAILABLE:
        pytest.skip("Ambiente com sklearn disponível; teste aplicável apenas ao fallback.")
    with pytest.raises(ImportError, match="scikit-learn"):
        build_preprocessor()
