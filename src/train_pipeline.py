"""Factory helpers to build full sklearn pipelines for training/inference."""

from __future__ import annotations

from typing import Any

from src.pipeline_components import RawToModelFrameTransformer
from src.preprocessing import (
    DEFAULT_SCALER_FOR_LINEAR,
    build_preprocessing_bundle,
)

try:
    from sklearn.pipeline import Pipeline

    _SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    Pipeline = Any  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


def build_model_pipeline(
    model: Any,
    *,
    year_t: int,
    scaler_strategy: str = DEFAULT_SCALER_FOR_LINEAR,
    enable_feature_engineering: bool = True,
    feature_pruning_plan: dict[str, Any] | None = None,
    strict_raw: bool = True,
    enable_age_bucket: bool = True,
) -> Any:
    """Build full sklearn pipeline: raw->model frame -> preprocessor -> estimator."""
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn não disponível. Instale as dependências de requirements.txt."
        )

    bundle = build_preprocessing_bundle(
        numeric_scaler=scaler_strategy,
        enable_feature_engineering=enable_feature_engineering,
        enable_age_bucket=enable_age_bucket,
        feature_pruning_plan=feature_pruning_plan,
    )
    raw_to_model = RawToModelFrameTransformer(
        year_t=year_t,
        expected_raw_cols=list(bundle["expected_raw_cols"]),
        expected_model_cols=list(bundle["expected_model_cols"]),
        enable_feature_engineering=enable_feature_engineering,
        feature_pruning_plan=feature_pruning_plan,
        strict_raw=strict_raw,
        enable_age_bucket=enable_age_bucket,
    )
    pipeline = Pipeline(
        steps=[
            ("raw_to_model", raw_to_model),
            ("preprocessor", bundle["preprocessor"]),
            ("model", model),
        ]
    )
    return pipeline
