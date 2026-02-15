"""Preprocessing builders for numeric/categorical encoding in sklearn."""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd

from src.contracts import PII_COLUMNS
from src.features import add_engineered_features, get_engineered_feature_names
from src.leakage import detect_leakage_columns, assert_no_leakage
from src.utils import get_logger

NUMERIC_COLS: list[str] = [
    "Ano ingresso",
    "Cf",
    "Cg",
    "Ct",
    "Defasagem",
    "IAA",
    "IAN",
    "IDA",
    "IEG",
    "INDE",
    "INDE 22",
    "IPS",
    "IPV",
    "Idade",
    "Ing",
    "Mat",
    "Nº Av",
    "Por",
]

CATEGORICAL_COLS: list[str] = [
    "Atingiu PV",
    "Destaque IDA",
    "Destaque IEG",
    "Destaque IPV",
    "Fase",
    "Fase_Ideal",
    "Gênero",
    "Indicado",
    "Instituição de ensino",
    "Pedra 20",
    "Pedra 21",
    "Pedra 22",
    "Pedra_Ano",
    "Rec Av1",
    "Rec Av2",
    "Rec Av3",
    "Rec Av4",
    "Turma",
]

DATETIME_COLS: list[str] = ["Data_Nasc"]

DROPPED_ALL_MISSING_COLS: list[str] = [
    "Ativo/ Inativo",
    "Ativo/ Inativo__dup1",
    "Destaque IPV__dup1",
    "Escola",
    "INDE 2023",
    "INDE 2024",
    "INDE 23",
    "IPP",
    "Pedra 2023",
    "Pedra 2024",
    "Pedra 23",
    "Rec Psicologia",
]

DEFAULT_SCALER_FOR_LINEAR = "standard"
DEFAULT_SCALER_FOR_TREE = "none"
_logger = get_logger(__name__)

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

    _SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    ColumnTransformer = Any  # type: ignore[assignment]
    SimpleImputer = Any  # type: ignore[assignment]
    Pipeline = Any  # type: ignore[assignment]
    OneHotEncoder = Any  # type: ignore[assignment]
    RobustScaler = Any  # type: ignore[assignment]
    StandardScaler = Any  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def get_feature_columns_for_model(
    numeric_cols: list[str] = NUMERIC_COLS,
    categorical_cols: list[str] = CATEGORICAL_COLS,
) -> list[str]:
    """Return canonical feature columns used by the model preprocessor."""
    numeric = _stable_unique(list(numeric_cols))
    categorical = _stable_unique(list(categorical_cols))
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise ValueError(
            f"Colunas duplicadas entre numéricas e categóricas: {overlap}"
        )
    return numeric + categorical


def get_expected_raw_feature_columns(
    numeric_cols: list[str] = NUMERIC_COLS,
    categorical_cols: list[str] = CATEGORICAL_COLS,
) -> list[str]:
    """Return expected raw feature columns for model input at inference time."""
    return get_feature_columns_for_model(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def get_excluded_columns() -> list[str]:
    """Return excluded raw columns derived from contract PII source of truth."""
    return sorted(list(PII_COLUMNS))


def assert_no_pii_in_features(feature_cols: list[str]) -> None:
    """Fail if feature list contains any PII columns from contracts source of truth."""
    violations = sorted(set(feature_cols) & set(PII_COLUMNS))
    if violations:
        raise ValueError(
            f"Features contêm colunas sensíveis (PII): {violations}"
        )


def validate_inference_frame(
    X: pd.DataFrame,
    expected_cols: list[str] | None = None,
    context: str = "inference",
    log_extras: bool = True,
) -> None:
    """Validate raw inference dataframe schema against expected model features."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"[{context}] Expected pandas.DataFrame, got {type(X)}")

    expected = (
        get_expected_raw_feature_columns()
        if expected_cols is None
        else list(expected_cols)
    )
    expected_set = set(expected)
    observed_set = set(X.columns)
    missing = sorted(expected_set - observed_set)
    if missing:
        raise ValueError(f"[{context}] Missing expected columns: {missing}")

    extras = sorted(observed_set - expected_set)
    if extras:
        extras_df = X.loc[:, extras].copy()
        leakage_report = detect_leakage_columns(
            X=extras_df,
            year_t=None,
            year_t1=None,
            include_year_specific=False,
        )
        if leakage_report["n_suspect"] > 0:
            suspects = ", ".join(leakage_report["suspect_columns"])
            raise ValueError(
                f"[{context}] Leakage-like extra columns detected in payload: {suspects}"
            )
        if log_extras:
            _logger.info(
                "[%s] Extra columns received (ignored by preprocessor): count=%d cols=%s",
                context,
                len(extras),
                extras,
            )


def _build_ohe() -> Any:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def _build_column_transformer_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"remainder": "drop"}
    if "verbose_feature_names_out" in inspect.signature(ColumnTransformer).parameters:
        kwargs["verbose_feature_names_out"] = False
    return kwargs


def _build_numeric_scaler(scaler: str) -> Any:
    scaler_normalized = scaler.strip().lower()
    if scaler_normalized == "standard":
        return StandardScaler(with_mean=True, with_std=True)
    if scaler_normalized == "robust":
        return RobustScaler(with_centering=True, with_scaling=True)
    if scaler_normalized == "none":
        return None
    raise ValueError(
        "Escalonador inválido. Use 'standard', 'robust' ou 'none'."
    )


def _expand_model_feature_columns(
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    enable_feature_engineering: bool,
    enable_age_bucket: bool,
) -> tuple[list[str], list[str]]:
    expanded_numeric = _stable_unique(list(numeric_cols))
    expanded_categorical = _stable_unique(list(categorical_cols))
    if not enable_feature_engineering:
        return expanded_numeric, expanded_categorical

    engineered = get_engineered_feature_names(enable_age_bucket=enable_age_bucket)
    expanded_numeric = _stable_unique(expanded_numeric + engineered["numeric"])
    expanded_categorical = _stable_unique(
        expanded_categorical + engineered["categorical"]
    )
    return expanded_numeric, expanded_categorical


def build_preprocessor(
    numeric_cols: list[str] = NUMERIC_COLS,
    categorical_cols: list[str] = CATEGORICAL_COLS,
    *,
    numeric_scaler: str = DEFAULT_SCALER_FOR_TREE,
    enable_feature_engineering: bool = True,
    enable_age_bucket: bool = True,
) -> Any:
    """Build sklearn ColumnTransformer with imputers + one-hot encoding."""
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn não disponível. Instale as dependências de requirements.txt."
        )

    numeric, categorical = _expand_model_feature_columns(
        numeric_cols,
        categorical_cols,
        enable_feature_engineering=enable_feature_engineering,
        enable_age_bucket=enable_age_bucket,
    )
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise ValueError(f"Colunas sobrepostas entre blocos num/cat: {overlap}")

    model_feature_cols = get_feature_columns_for_model(numeric, categorical)
    assert_no_pii_in_features(model_feature_cols)

    numeric_steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        )
    ]
    scaler_transformer = _build_numeric_scaler(numeric_scaler)
    if scaler_transformer is not None:
        numeric_steps.append(("scaler", scaler_transformer))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", add_indicator=True),
            ),
            ("onehot", _build_ohe()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        **_build_column_transformer_kwargs(),
    )


def build_preprocessing_bundle(
    numeric_scaler: str = DEFAULT_SCALER_FOR_TREE,
    numeric_cols: list[str] = NUMERIC_COLS,
    categorical_cols: list[str] = CATEGORICAL_COLS,
    *,
    enable_feature_engineering: bool = True,
    enable_age_bucket: bool = True,
) -> dict[str, Any]:
    """Build reusable preprocessing assets for both train and inference paths."""
    expected_raw_cols = get_expected_raw_feature_columns(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    expanded_numeric, expanded_categorical = _expand_model_feature_columns(
        numeric_cols,
        categorical_cols,
        enable_feature_engineering=enable_feature_engineering,
        enable_age_bucket=enable_age_bucket,
    )
    expected_model_cols = get_feature_columns_for_model(
        numeric_cols=expanded_numeric,
        categorical_cols=expanded_categorical,
    )
    excluded_cols = get_excluded_columns()
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_scaler=numeric_scaler,
        enable_feature_engineering=enable_feature_engineering,
        enable_age_bucket=enable_age_bucket,
    )

    def transform_raw_to_model_frame(
        X_raw: pd.DataFrame, *, strict: bool = False, context: str = "inference"
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        validate_inference_frame(
            X_raw,
            expected_cols=expected_raw_cols,
            context=context,
        )
        X_model = X_raw.loc[:, expected_raw_cols].copy()
        feature_report: dict[str, Any] = {
            "features_added": [],
            "base_columns_used": [],
            "enable_age_bucket": bool(enable_age_bucket),
        }
        if enable_feature_engineering:
            X_model, feature_report = add_engineered_features(
                X_model,
                enable_age_bucket=enable_age_bucket,
                strict=strict,
            )

        missing_model_cols = sorted(set(expected_model_cols) - set(X_model.columns))
        if missing_model_cols:
            raise ValueError(
                f"[{context}] Missing model columns after feature engineering: "
                f"{missing_model_cols}"
            )
        X_model = X_model.loc[:, expected_model_cols].copy()
        assert_no_leakage(
            X_model,
            year_t=None,
            year_t1=None,
            include_year_specific=False,
        )
        return X_model, feature_report

    return {
        "expected_cols": expected_raw_cols,
        "expected_raw_cols": expected_raw_cols,
        "expected_model_cols": expected_model_cols,
        "excluded_cols": excluded_cols,
        "numeric_scaler": numeric_scaler,
        "preprocessor": preprocessor,
        "enable_feature_engineering": enable_feature_engineering,
        "enable_age_bucket": enable_age_bucket,
        "transform_raw_to_model_frame": transform_raw_to_model_frame,
    }


if __name__ == "__main__":  # pragma: no cover - convenience smoke usage
    preprocessor = build_preprocessor()
    print(type(preprocessor).__name__)
