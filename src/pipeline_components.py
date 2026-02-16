"""Reusable sklearn-compatible components for end-to-end model pipelines."""

from __future__ import annotations

from copy import deepcopy
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
        preprocessing_spec: dict[str, Any] | None = None,
        strict_raw: bool = True,
        enable_age_bucket: bool = True,
    ) -> None:
        self.year_t = int(year_t)
        self.expected_raw_cols = self._normalize_columns(expected_raw_cols)
        self.expected_model_cols = self._normalize_columns(expected_model_cols)
        self.enable_feature_engineering = bool(enable_feature_engineering)
        self.feature_pruning_plan = feature_pruning_plan
        self.preprocessing_spec = dict(preprocessing_spec or {})
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
        self.expected_raw_cols_ = list(self.expected_raw_cols)
        self.expected_model_cols_ = list(self.expected_model_cols)
        self.feature_pruning_plan_ = (
            deepcopy(self.feature_pruning_plan)
            if self.feature_pruning_plan is not None
            else None
        )
        self.preprocessing_spec_ = deepcopy(self.preprocessing_spec)
        dropped_by_pruning: list[str] = []
        if self.feature_pruning_plan_:
            for key in (
                "dropped_all_missing_cols",
                "dropped_constant_numeric_cols",
                "dropped_constant_categorical_cols",
                "dropped_high_cardinality_cols",
                "blocked_by_leakage_cols",
                "dropped_excluded_cols",
            ):
                dropped_by_pruning.extend(
                    list(self.feature_pruning_plan_.get(key, []))
                )
        self.consistency_report_ = {
            "n_expected_raw_cols": len(self.expected_raw_cols_),
            "n_expected_model_cols_final": len(self.expected_model_cols_),
            "dropped_by_pruning": sorted(set(dropped_by_pruning)),
            "engineered_features_enabled": bool(self.enable_feature_engineering),
            "age_bucket_enabled": bool(self.enable_age_bucket),
            "scaler_strategy": self.preprocessing_spec_.get("scaler_strategy"),
            "ohe_sparse_flag_used": self.preprocessing_spec_.get(
                "ohe_sparse_flag_used"
            ),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"[PIPELINE_TRANSFORM] Expected pandas.DataFrame, got {type(X)}"
            )
        expected_raw_cols = list(
            getattr(self, "expected_raw_cols_", self.expected_raw_cols)
        )
        expected_model_cols = list(
            getattr(self, "expected_model_cols_", self.expected_model_cols)
        )
        feature_pruning_plan = getattr(
            self, "feature_pruning_plan_", self.feature_pruning_plan
        )
        transformed = transform_raw_to_model_frame(
            X_raw=X,
            year_t=self.year_t,
            expected_raw_cols=expected_raw_cols,
            expected_model_cols=expected_model_cols,
            enable_feature_engineering=self.enable_feature_engineering,
            enable_age_bucket=self.enable_age_bucket,
            feature_pruning_plan=feature_pruning_plan,
            strict_raw=self.strict_raw,
            include_report=False,
            context=f"pipeline_y{self.year_t}",
        )
        if isinstance(transformed, tuple):
            transformed_df = transformed[0]
        else:
            transformed_df = transformed

        missing_internal = [
            col for col in expected_model_cols if col not in transformed_df.columns
        ]
        if missing_internal:
            raise ValueError(
                "[MODEL] missing in model frame: "
                f"{sorted(missing_internal)}"
            )
        return transformed_df.loc[:, expected_model_cols].copy()
