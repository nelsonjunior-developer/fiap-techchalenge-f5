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
