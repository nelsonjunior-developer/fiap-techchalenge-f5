import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_pruning import (
    apply_feature_pruning_plan,
    compute_feature_pruning_plan,
    persist_feature_pruning_report,
)
from src.preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    _SKLEARN_AVAILABLE,
    build_preprocessing_bundle,
    build_pruning_plan_from_training_frame,
    get_expected_raw_feature_columns,
)


def _build_raw_frame(n_rows: int = 10) -> pd.DataFrame:
    raw_cols = get_expected_raw_feature_columns()
    data: dict[str, object] = {}
    for idx, col in enumerate(NUMERIC_COLS):
        values = np.linspace(1, n_rows, n_rows, dtype=float) + idx
        data[col] = pd.Series(values, dtype="Float64")
    for col in CATEGORICAL_COLS:
        values = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"][:n_rows]
        data[col] = pd.Series(values, dtype="string")
    return pd.DataFrame(data).loc[:, raw_cols]


def test_compute_plan_drops_all_missing_and_constant_columns() -> None:
    X = pd.DataFrame(
        {
            "num_missing": pd.Series([pd.NA, pd.NA], dtype="Float64"),
            "num_const": pd.Series([1.0, 1.0], dtype="Float64"),
            "num_var": pd.Series([1.0, 2.0], dtype="Float64"),
            "cat_const": pd.Series(["A", "A"], dtype="string"),
            "cat_var": pd.Series(["A", "B"], dtype="string"),
        }
    )
    plan = compute_feature_pruning_plan(
        X,
        numeric_cols=["num_missing", "num_const", "num_var"],
        categorical_cols=["cat_const", "cat_var"],
        pruning_config={
            "max_categorical_cardinality_abs": 50,
            "max_categorical_cardinality_rate": 1.0,
        },
    )
    assert "num_missing" in plan["dropped_all_missing_cols"]
    assert "num_const" in plan["dropped_constant_numeric_cols"]
    assert "cat_const" in plan["dropped_constant_categorical_cols"]
    assert "num_var" in plan["kept_numeric_cols"]
    assert "cat_var" in plan["kept_categorical_cols"]


def test_compute_plan_drops_high_cardinality_by_abs_and_rate() -> None:
    X = pd.DataFrame(
        {
            "cat_hi": pd.Series([f"v{i}" for i in range(10)], dtype="string"),
            "cat_ok": pd.Series(["A", "A", "B", "B", "A", "B", "A", "A", "B", "A"], dtype="string"),
        }
    )
    plan = compute_feature_pruning_plan(
        X,
        numeric_cols=[],
        categorical_cols=["cat_hi", "cat_ok"],
        pruning_config={
            "max_categorical_cardinality_abs": 50,
            "max_categorical_cardinality_rate": 0.2,
        },
    )
    assert "cat_hi" in plan["dropped_high_cardinality_cols"]
    assert "cat_ok" not in plan["dropped_high_cardinality_cols"]


def test_plan_lists_are_sorted_and_apply_preserves_order() -> None:
    X = pd.DataFrame(
        {
            "b": pd.Series([1.0, 2.0], dtype="Float64"),
            "a": pd.Series([2.0, 3.0], dtype="Float64"),
            "c": pd.Series(["A", "B"], dtype="string"),
        }
    )
    plan = compute_feature_pruning_plan(
        X,
        numeric_cols=["b", "a"],
        categorical_cols=["c"],
    )
    assert plan["kept_model_cols"] == sorted(plan["kept_model_cols"])
    X_kept = apply_feature_pruning_plan(X, plan)
    assert list(X_kept.columns) == plan["kept_model_cols"]


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn não disponível no ambiente")
def test_inference_applies_fixed_plan_without_recalculation() -> None:
    X_train_raw = _build_raw_frame(n_rows=10)
    # Force Turma drop on train by low threshold.
    X_train_raw["Turma"] = pd.Series([f"T{i}" for i in range(10)], dtype="string")
    plan = build_pruning_plan_from_training_frame(
        X_train_raw,
        enable_feature_engineering=True,
        pruning_config={
            "max_categorical_cardinality_abs": 5,
            "max_categorical_cardinality_rate": 0.5,
        },
    )
    assert "Turma" in plan["dropped_high_cardinality_cols"]

    bundle = build_preprocessing_bundle(
        numeric_scaler="none",
        enable_feature_engineering=True,
        feature_pruning_plan=plan,
    )
    preprocessor = bundle["preprocessor"]
    assert list(preprocessor.transformers[0][2]) == plan["kept_numeric_cols"]
    assert list(preprocessor.transformers[1][2]) == plan["kept_categorical_cols"]

    X_inf_raw = _build_raw_frame(n_rows=5)
    # Lower cardinality at inference should not change plan outcome.
    X_inf_raw["Turma"] = pd.Series(["A", "A", "B", "B", "A"], dtype="string")
    X_model, _ = bundle["transform_raw_to_model_frame"](X_inf_raw, context="inference")
    assert "Turma" not in X_model.columns
    assert list(X_model.columns) == plan["kept_model_cols"]


def test_persist_pruning_report_no_sensitive_payload(tmp_path: Path) -> None:
    X = pd.DataFrame(
        {
            "num": pd.Series([1.0, 2.0], dtype="Float64"),
            "cat": pd.Series(["A", "B"], dtype="string"),
        }
    )
    plan = compute_feature_pruning_plan(X, numeric_cols=["num"], categorical_cols=["cat"])
    report_path = tmp_path / "feature_pruning_report.json"
    persist_feature_pruning_report(plan, path=report_path, markdown=True)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "values" not in payload
    assert "student_ids" not in payload
    assert "ids" not in payload
