"""Deterministic feature pruning plan for train/inference consistency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_PRUNING_CONFIG: dict[str, Any] = {
    "max_categorical_cardinality_abs": 50,
    "max_categorical_cardinality_rate": 0.20,
    "drop_constant_numeric": True,
    "drop_constant_categorical": True,
    "drop_all_missing": True,
    "exclude_pii": True,
}


def _stable_sorted(values: list[str]) -> list[str]:
    return sorted(set(values))


def _validate_disjoint_columns(
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
) -> None:
    numeric_set = set(numeric_cols)
    categorical_set = set(categorical_cols)
    datetime_set = set(datetime_cols)
    overlap = (
        (numeric_set & categorical_set)
        | (numeric_set & datetime_set)
        | (categorical_set & datetime_set)
    )
    if overlap:
        raise ValueError(
            f"Feature families com sobreposição: {sorted(overlap)}"
        )


def compute_feature_pruning_plan(
    X: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str] | None = None,
    excluded_cols: list[str] | None = None,
    pruning_config: dict[str, Any] | None = None,
    leakage_suspects: list[str] | None = None,
) -> dict[str, Any]:
    """Compute deterministic feature pruning plan from training frame only."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(X)}")

    config = dict(DEFAULT_PRUNING_CONFIG)
    if pruning_config:
        config.update(pruning_config)

    dt_cols = list(datetime_cols or [])
    _validate_disjoint_columns(numeric_cols, categorical_cols, dt_cols)

    numeric_in = [col for col in numeric_cols if col in X.columns]
    categorical_in = [col for col in categorical_cols if col in X.columns]
    datetime_in = [col for col in dt_cols if col in X.columns]
    candidates = list(dict.fromkeys(numeric_in + categorical_in))

    n_rows = int(len(X))
    cardinality_rate_threshold = float(config["max_categorical_cardinality_rate"]) * (
        float(n_rows) if n_rows > 0 else 0.0
    )

    dropped_all_missing_cols: list[str] = []
    dropped_constant_numeric_cols: list[str] = []
    dropped_constant_categorical_cols: list[str] = []
    dropped_high_cardinality_cols: list[str] = []
    dropped_excluded_cols: list[str] = []
    blocked_by_leakage_cols: list[str] = []

    if config["drop_all_missing"]:
        dropped_all_missing_cols = [
            col for col in candidates if X[col].notna().sum() == 0
        ]

    if config["drop_constant_numeric"]:
        dropped_constant_numeric_cols = [
            col
            for col in numeric_in
            if X[col].dropna().nunique() <= 1
        ]

    if config["drop_constant_categorical"]:
        dropped_constant_categorical_cols = [
            col
            for col in categorical_in
            if X[col].dropna().nunique() <= 1
        ]

    for col in categorical_in:
        non_null_unique = int(X[col].dropna().nunique())
        exceeds_abs = non_null_unique > int(config["max_categorical_cardinality_abs"])
        exceeds_rate = non_null_unique > cardinality_rate_threshold
        if exceeds_abs or exceeds_rate:
            dropped_high_cardinality_cols.append(col)

    if excluded_cols:
        dropped_excluded_cols = [col for col in candidates if col in set(excluded_cols)]

    if leakage_suspects:
        blocked_by_leakage_cols = [
            col for col in candidates if col in set(leakage_suspects)
        ]

    drop_set = set(
        dropped_all_missing_cols
        + dropped_constant_numeric_cols
        + dropped_constant_categorical_cols
        + dropped_high_cardinality_cols
        + dropped_excluded_cols
        + blocked_by_leakage_cols
    )

    kept_numeric_cols = [col for col in numeric_in if col not in drop_set]
    kept_categorical_cols = [col for col in categorical_in if col not in drop_set]

    plan = {
        "n_rows": n_rows,
        "config": config,
        "numeric_cols_in": _stable_sorted(numeric_in),
        "categorical_cols_in": _stable_sorted(categorical_in),
        "datetime_cols_in": _stable_sorted(datetime_in),
        "dropped_all_missing_cols": _stable_sorted(dropped_all_missing_cols),
        "dropped_constant_numeric_cols": _stable_sorted(dropped_constant_numeric_cols),
        "dropped_constant_categorical_cols": _stable_sorted(
            dropped_constant_categorical_cols
        ),
        "dropped_high_cardinality_cols": _stable_sorted(dropped_high_cardinality_cols),
        "blocked_by_leakage_cols": _stable_sorted(blocked_by_leakage_cols),
        "dropped_excluded_cols": _stable_sorted(dropped_excluded_cols),
        "kept_numeric_cols": _stable_sorted(kept_numeric_cols),
        "kept_categorical_cols": _stable_sorted(kept_categorical_cols),
        "kept_model_cols": _stable_sorted(kept_numeric_cols + kept_categorical_cols),
        "notes": [
            "cardinality thresholds apply on training fit only",
            "inference must apply this saved plan without recalculation",
        ],
    }
    return plan


def apply_feature_pruning_plan(X: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    """Apply precomputed pruning plan to any frame (inference uses apply-only)."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(X)}")

    kept_model_cols = list(plan.get("kept_model_cols", []))
    missing_cols = [col for col in kept_model_cols if col not in X.columns]
    if missing_cols:
        raise ValueError(
            f"Pruning plan expected columns missing in frame: {sorted(missing_cols)}"
        )

    return X.loc[:, kept_model_cols].copy()


def persist_feature_pruning_report(
    plan: dict[str, Any],
    path: str | Path = "artifacts/feature_pruning_report.json",
    markdown: bool = False,
) -> Path:
    """Persist pruning plan as JSON and optional markdown summary."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if markdown:
        md_path = report_path.with_suffix(".md")
        md_lines = [
            "# Feature Pruning Report",
            "",
            f"- n_rows: {plan.get('n_rows', 0)}",
            f"- kept_model_cols: {len(plan.get('kept_model_cols', []))}",
            f"- dropped_all_missing_cols: {len(plan.get('dropped_all_missing_cols', []))}",
            f"- dropped_constant_numeric_cols: {len(plan.get('dropped_constant_numeric_cols', []))}",
            f"- dropped_constant_categorical_cols: {len(plan.get('dropped_constant_categorical_cols', []))}",
            f"- dropped_high_cardinality_cols: {len(plan.get('dropped_high_cardinality_cols', []))}",
            f"- blocked_by_leakage_cols: {len(plan.get('blocked_by_leakage_cols', []))}",
            f"- dropped_excluded_cols: {len(plan.get('dropped_excluded_cols', []))}",
        ]
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report_path

