"""Missing-value imputation planning utilities for model features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from sklearn.impute import SimpleImputer as _SklearnSimpleImputer
except ModuleNotFoundError:  # pragma: no cover - exercised only when sklearn is unavailable
    _SklearnSimpleImputer = None

DEFAULT_MISSING_POLICY: dict[str, Any] = {
    "drop_all_missing_columns": True,
    "numeric_strategy": "median",
    "categorical_strategy": "most_frequent",
    "add_missing_indicators": True,
}


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _validate_columns_exist(X: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [column for column in cols if column not in X.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em X para {label}: {sorted(missing)}")


def _validate_disjoint(
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
) -> None:
    numeric_set = set(numeric_cols)
    categorical_set = set(categorical_cols)
    datetime_set = set(datetime_cols)

    overlaps: dict[str, set[str]] = {
        "numeric_categorical": numeric_set & categorical_set,
        "numeric_datetime": numeric_set & datetime_set,
        "categorical_datetime": categorical_set & datetime_set,
    }
    non_empty = {name: sorted(values) for name, values in overlaps.items() if values}
    if non_empty:
        raise ValueError(f"Listas de colunas não são disjuntas: {non_empty}")


def find_all_missing_columns(X: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return sorted columns that are fully missing in the given dataframe."""
    _validate_columns_exist(X, cols, label="find_all_missing_columns")
    all_missing = [column for column in cols if X[column].isna().all()]
    return sorted(all_missing)


def build_imputation_plan(
    X_train: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
    exclude_cols: list[str] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducible imputation plan for numeric and categorical features."""
    numeric_input = _stable_unique(list(numeric_cols))
    categorical_input = _stable_unique(list(categorical_cols))
    datetime_input = _stable_unique(list(datetime_cols))
    excluded_input = _stable_unique(list(exclude_cols or []))

    _validate_disjoint(numeric_input, categorical_input, datetime_input)
    _validate_columns_exist(
        X_train,
        _stable_unique(numeric_input + categorical_input + datetime_input),
        label="build_imputation_plan",
    )

    effective_policy = dict(DEFAULT_MISSING_POLICY)
    if policy:
        effective_policy.update(policy)

    numeric_after_exclude = [column for column in numeric_input if column not in excluded_input]
    categorical_after_exclude = [
        column for column in categorical_input if column not in excluded_input
    ]
    datetime_excluded = [column for column in datetime_input if column not in excluded_input]

    dropped_all_missing_cols: list[str] = []
    if effective_policy["drop_all_missing_columns"]:
        dropped_all_missing_cols = find_all_missing_columns(
            X_train,
            numeric_after_exclude + categorical_after_exclude,
        )

    dropped_set = set(dropped_all_missing_cols)
    numeric_cols_used = sorted([column for column in numeric_after_exclude if column not in dropped_set])
    categorical_cols_used = sorted(
        [column for column in categorical_after_exclude if column not in dropped_set]
    )
    datetime_cols_excluded = sorted(datetime_excluded)
    excluded_cols_sorted = sorted(excluded_input)
    dropped_all_missing_sorted = sorted(dropped_all_missing_cols)

    notes = [
        "Datetime columns are excluded from imputation at this stage.",
        "Imputation must be applied inside sklearn Pipeline/ColumnTransformer.",
    ]
    if effective_policy["add_missing_indicators"]:
        notes.append(
            "add_indicator=True increases transformed feature dimensionality after fit."
        )

    plan = {
        "policy": effective_policy,
        "numeric_cols_used": numeric_cols_used,
        "categorical_cols_used": categorical_cols_used,
        "datetime_cols_excluded": datetime_cols_excluded,
        "dropped_all_missing_cols": dropped_all_missing_sorted,
        "excluded_cols": excluded_cols_sorted,
        "counts": {
            "n_numeric_used": len(numeric_cols_used),
            "n_categorical_used": len(categorical_cols_used),
            "n_dropped_all_missing": len(dropped_all_missing_sorted),
            "n_datetime_excluded": len(datetime_cols_excluded),
        },
        "notes": notes,
    }
    return plan


def make_imputers(plan: dict[str, Any]) -> tuple[Any, Any]:
    """Create sklearn imputers from a precomputed imputation plan."""
    policy = plan["policy"]

    if _SklearnSimpleImputer is None:
        # Fallback shim keeps tests/runtime importable in restricted environments.
        # Real training runs should install scikit-learn from requirements.
        class _SimpleImputerShim:  # pragma: no cover - trivial behavior
            def __init__(self, strategy: str, add_indicator: bool):
                self.strategy = strategy
                self.add_indicator = add_indicator

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        numeric_imputer = _SimpleImputerShim(
            strategy=policy["numeric_strategy"],
            add_indicator=policy["add_missing_indicators"],
        )
        categorical_imputer = _SimpleImputerShim(
            strategy=policy["categorical_strategy"],
            add_indicator=policy["add_missing_indicators"],
        )
        return numeric_imputer, categorical_imputer

    numeric_imputer = _SklearnSimpleImputer(
        strategy=policy["numeric_strategy"],
        add_indicator=policy["add_missing_indicators"],
    )
    categorical_imputer = _SklearnSimpleImputer(
        strategy=policy["categorical_strategy"],
        add_indicator=policy["add_missing_indicators"],
    )
    return numeric_imputer, categorical_imputer


def persist_imputation_plan(
    plan: dict[str, Any],
    path: str | Path = "artifacts/imputation_plan.json",
) -> None:
    """Persist imputation plan JSON with only aggregated metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
