from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.temporal_shift import (
    DEFAULT_SHIFT_THRESHOLDS,
    aggregate_shift_status,
    compute_feature_shift,
    compute_target_shift,
)


def _collect_keys(payload: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys |= _collect_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_keys(item)
    return keys


def test_compute_target_shift_returns_expected_values() -> None:
    y_train = pd.Series([1, 0, 1, 1, 0], dtype="Int64")
    y_holdout = pd.Series([1, 0, 0, 0, 0], dtype="Int64")

    shift = compute_target_shift(y_train, y_holdout)

    assert shift["train"]["n"] == 5
    assert shift["train"]["n_pos"] == 3
    assert shift["train"]["prevalence"] == pytest.approx(0.6)
    assert shift["holdout"]["prevalence"] == pytest.approx(0.2)
    assert shift["delta_prevalence_abs"] == pytest.approx(-0.4)
    assert shift["delta_prevalence_rel"] == pytest.approx((0.2 / 0.6) - 1.0)


def test_compute_feature_shift_equal_distributions_return_zero_scores() -> None:
    X_train = pd.DataFrame(
        {
            "num": [0.0, 1.0, 2.0, 3.0, 4.0],
            "cat": ["A", "B", "A", "B", "A"],
            "bin": [0, 1, 0, 1, 0],
        }
    )
    X_holdout = X_train.copy()

    shift = compute_feature_shift(X_train, X_holdout)
    assert shift["n_features"] == 3
    assert shift["counts_by_severity"]["fail"] == 0
    assert shift["counts_by_severity"]["warning"] == 0

    for item in shift["features"]:
        assert float(item["drift_score"]) == pytest.approx(0.0, abs=1e-9)


def test_extreme_feature_shift_generates_fail_status() -> None:
    n_rows = 200
    X_train = pd.DataFrame(
        {
            "num_a": np.linspace(0.0, 1.0, n_rows),
            "num_b": np.linspace(-1.0, 1.0, n_rows),
            "cat_a": ["A"] * 190 + ["B"] * 10,
        }
    )
    X_holdout = pd.DataFrame(
        {
            "num_a": np.linspace(5.0, 6.0, n_rows),
            "num_b": np.linspace(10.0, 11.0, n_rows),
            "cat_a": ["Z"] * n_rows,
        }
    )
    y_train = pd.Series(([0, 1] * (n_rows // 2)), dtype="Int64")
    y_holdout = pd.Series(([0, 1] * (n_rows // 2)), dtype="Int64")

    target_shift = compute_target_shift(y_train, y_holdout)
    feature_shift = compute_feature_shift(X_train, X_holdout)
    summary = aggregate_shift_status(
        target_shift=target_shift,
        feature_shift_summary=feature_shift,
        thresholds=DEFAULT_SHIFT_THRESHOLDS,
    )

    assert feature_shift["counts_by_severity"]["fail"] >= 3
    assert summary["status"] == "FAIL"


def test_shift_payload_is_privacy_safe_and_has_no_id_like_keys() -> None:
    X_train = pd.DataFrame(
        {
            "num": [0.1, 0.2, 0.3, 0.4],
            "cat": ["A", "A", "B", "B"],
        }
    )
    X_holdout = pd.DataFrame(
        {
            "num": [0.1, 0.2, 0.3, 0.4],
            "cat": ["A", "A", "B", "B"],
        }
    )
    y_train = pd.Series([0, 1, 0, 1], dtype="Int64")
    y_holdout = pd.Series([0, 1, 0, 1], dtype="Int64")

    target_shift = compute_target_shift(y_train, y_holdout)
    feature_shift = compute_feature_shift(X_train, X_holdout)
    summary = aggregate_shift_status(target_shift, feature_shift)
    payload = {
        "target_shift": target_shift,
        "feature_shift": feature_shift,
        "status_summary": summary,
    }

    keys = _collect_keys(payload)
    forbidden = {"ra", "ra_list", "ids", "student_ids", "students", "records"}
    assert forbidden.isdisjoint(keys)

    serialized = json.dumps(payload).lower()
    assert '"ids"' not in serialized
    assert "ra_list" not in serialized

    for item in feature_shift["features"]:
        if item["dtype_kind"] in {"categorical", "binary"}:
            assert len(item["top_categories_train"]) <= 5
            assert len(item["top_categories_holdout"]) <= 5
