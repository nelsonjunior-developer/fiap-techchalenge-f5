"""Feature selection and dtype-based feature family split helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.contracts import PII_COLUMNS

# Keep exclusion policy aligned with the data contracts and avoid duplicating lists.
# RA is already removed before model features are built in temporal pairing.
DEFAULT_EXCLUDE_COLUMNS: set[str] = set(PII_COLUMNS) - {"RA"}

_GRADE_COLUMNS: list[str] = ["Mat", "Por", "Ing"]
_INDICATOR_COLUMNS: list[str] = ["IAA", "IAN", "IDA", "IEG", "IPS", "IPV", "INDE"]

ENGINEERED_NUMERIC_FEATURES: list[str] = [
    "avg_grades",
    "min_grade",
    "max_grade",
    "grade_std",
    "missing_grades_count",
    "missing_indicators_count",
    "defasagem_abs",
    "defasagem_neg_flag",
    "age_is_missing_flag",
]

ENGINEERED_CATEGORICAL_FEATURES: list[str] = ["age_bucket"]
ENGINEERED_ALL_FEATURES: list[str] = ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES


def get_feature_columns(
    X: pd.DataFrame,
    exclude_columns: set[str] | None = None,
) -> list[str]:
    """Return feature columns after removing excluded names."""
    excluded = set(DEFAULT_EXCLUDE_COLUMNS if exclude_columns is None else exclude_columns)
    feature_cols = [column for column in X.columns if column not in excluded]

    unexpected = [column for column in feature_cols if column in excluded]
    if unexpected:
        raise ValueError(f"Excluded columns leaked into feature list: {unexpected}")
    return feature_cols


def split_numeric_categorical_datetime(
    X: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """Split features into numeric, categorical and datetime families from pandas dtypes."""
    missing = [column for column in feature_cols if column not in X.columns]
    if missing:
        raise ValueError(f"Feature columns ausentes em X: {missing}")

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []

    for column in feature_cols:
        series = X[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(column)
        elif pd.api.types.is_bool_dtype(series):
            categorical_cols.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)

    excluded_cols = [column for column in X.columns if column not in feature_cols]
    all_missing_cols = [column for column in feature_cols if X[column].isna().all()]

    report = {
        "n_total_features": len(feature_cols),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "n_datetime": len(datetime_cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "excluded_cols": excluded_cols,
        "n_all_missing_cols_no_recorte": len(all_missing_cols),
        "all_missing_cols_no_recorte": all_missing_cols,
    }
    return numeric_cols, categorical_cols, datetime_cols, report


def persist_feature_split_report(
    report: dict[str, Any],
    path: str | Path = "artifacts/feature_split_report.json",
) -> None:
    """Persist aggregated feature split report as JSON."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_engineered_feature_names(enable_age_bucket: bool = True) -> dict[str, list[str]]:
    """Return engineered feature names grouped by family."""
    numeric = list(ENGINEERED_NUMERIC_FEATURES)
    categorical = list(ENGINEERED_CATEGORICAL_FEATURES) if enable_age_bucket else []
    return {
        "numeric": numeric,
        "categorical": categorical,
        "all": numeric + categorical,
    }


def _build_age_bucket(idade_numeric: pd.Series) -> pd.Series:
    bucket = pd.Series(pd.NA, index=idade_numeric.index, dtype="string")
    bucket.loc[((idade_numeric >= 7) & (idade_numeric <= 10)).fillna(False)] = "07_10"
    bucket.loc[((idade_numeric >= 11) & (idade_numeric <= 14)).fillna(False)] = "11_14"
    bucket.loc[((idade_numeric >= 15) & (idade_numeric <= 18)).fillna(False)] = "15_18"
    bucket.loc[(idade_numeric >= 19).fillna(False)] = "19_plus"
    return bucket


def add_engineered_features(
    X: pd.DataFrame,
    enable_age_bucket: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create low-leakage engineered features from year-t columns only."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(X)}")

    X_out = X.copy()

    if strict:
        essential_minimum = {"Defasagem", "Mat", "Por"}
        missing_essentials = sorted(essential_minimum - set(X_out.columns))
        if missing_essentials:
            raise ValueError(
                f"Colunas essenciais ausentes para strict=True: {missing_essentials}"
            )

    features_added: list[str] = []
    base_columns_used: set[str] = set()

    grade_cols = [col for col in _GRADE_COLUMNS if col in X_out.columns]
    if grade_cols:
        grades = X_out[grade_cols].apply(pd.to_numeric, errors="coerce")
        X_out["avg_grades"] = grades.mean(axis=1, skipna=True).astype("Float64")
        X_out["min_grade"] = grades.min(axis=1, skipna=True).astype("Float64")
        X_out["max_grade"] = grades.max(axis=1, skipna=True).astype("Float64")
        grade_non_null = grades.notna().sum(axis=1)
        grade_std = grades.std(axis=1, ddof=0, skipna=True).astype("Float64")
        grade_std = grade_std.where(grade_non_null >= 2, pd.NA)
        X_out["grade_std"] = grade_std.astype("Float64")
        X_out["missing_grades_count"] = grades.isna().sum(axis=1).astype("Int64")
        features_added.extend(
            ["avg_grades", "min_grade", "max_grade", "grade_std", "missing_grades_count"]
        )
        base_columns_used.update(grade_cols)

    indicator_cols = [col for col in _INDICATOR_COLUMNS if col in X_out.columns]
    if indicator_cols:
        indicators = X_out[indicator_cols].apply(pd.to_numeric, errors="coerce")
        X_out["missing_indicators_count"] = indicators.isna().sum(axis=1).astype("Int64")
        features_added.append("missing_indicators_count")
        base_columns_used.update(indicator_cols)

    if "Defasagem" in X_out.columns:
        defasagem_num = pd.to_numeric(X_out["Defasagem"], errors="coerce")
        X_out["defasagem_abs"] = defasagem_num.abs().astype("Float64")

        defasagem_neg_flag = pd.Series(pd.NA, index=X_out.index, dtype="Int64")
        valid_defasagem = defasagem_num.notna()
        defasagem_neg_flag.loc[valid_defasagem] = (
            defasagem_num.loc[valid_defasagem] < 0
        ).astype("Int64")
        X_out["defasagem_neg_flag"] = defasagem_neg_flag

        features_added.extend(["defasagem_abs", "defasagem_neg_flag"])
        base_columns_used.add("Defasagem")

    if "Idade" in X_out.columns:
        idade_num = pd.to_numeric(X_out["Idade"], errors="coerce")
        X_out["age_is_missing_flag"] = idade_num.isna().astype("Int64")
        features_added.append("age_is_missing_flag")
        base_columns_used.add("Idade")

        if enable_age_bucket:
            X_out["age_bucket"] = _build_age_bucket(idade_num)
            features_added.append("age_bucket")

    report = {
        "features_added": sorted(set(features_added)),
        "base_columns_used": sorted(base_columns_used),
        "enable_age_bucket": bool(enable_age_bucket),
    }
    return X_out, report
