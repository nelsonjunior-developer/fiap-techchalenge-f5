"""Reusable sklearn-compatible components for end-to-end model pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing import transform_raw_to_model_frame

try:
    from sklearn.base import BaseEstimator, TransformerMixin

    _SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    class BaseEstimator:  # type: ignore[override]
        pass

    class TransformerMixin:  # type: ignore[override]
        pass

    _SKLEARN_AVAILABLE = False


class RawToModelFrameTransformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer that converts raw input frame into model-ready frame."""

    def __init__(
        self,
        *,
        year_t: int,
        expected_raw_cols: list[str],
        expected_model_cols: list[str],
        enable_feature_engineering: bool,
        feature_pruning_plan: dict[str, Any] | None,
        strict_raw: bool = True,
        enable_age_bucket: bool = True,
    ) -> None:
        self.year_t = int(year_t)
        self.expected_raw_cols = self._normalize_columns(expected_raw_cols)
        self.expected_model_cols = self._normalize_columns(expected_model_cols)
        self.enable_feature_engineering = bool(enable_feature_engineering)
        self.feature_pruning_plan = feature_pruning_plan
        self.strict_raw = bool(strict_raw)
        self.enable_age_bucket = bool(enable_age_bucket)

    @staticmethod
    def _normalize_columns(columns: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in columns:
            col = str(value).strip()
            if not col or col in seen:
                continue
            normalized.append(col)
            seen.add(col)
        return normalized

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RawToModelFrameTransformer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"[PIPELINE_FIT] Expected pandas.DataFrame, got {type(X)}"
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"[PIPELINE_TRANSFORM] Expected pandas.DataFrame, got {type(X)}"
            )
        transformed = transform_raw_to_model_frame(
            X_raw=X,
            year_t=self.year_t,
            expected_raw_cols=self.expected_raw_cols,
            expected_model_cols=self.expected_model_cols,
            enable_feature_engineering=self.enable_feature_engineering,
            enable_age_bucket=self.enable_age_bucket,
            feature_pruning_plan=self.feature_pruning_plan,
            strict_raw=self.strict_raw,
            include_report=False,
            context=f"pipeline_y{self.year_t}",
        )
        if isinstance(transformed, tuple):
            return transformed[0]
        return transformed
