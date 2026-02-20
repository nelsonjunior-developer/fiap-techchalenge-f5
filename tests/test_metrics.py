from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    build_default_prediction_policy,
    compute_classification_metrics_at_threshold,
    compute_confusion_counts_at_threshold,
    compute_metrics_threshold,
    compute_metrics_topk,
    compute_prevalence,
    select_threshold_for_target_recall,
    summarize_proba,
)


def test_compute_metrics_threshold_deterministic() -> None:
    y_true = pd.Series([1, 1, 0, 0], dtype="Int64")
    y_score = np.array([0.9, 0.8, 0.7, 0.1], dtype=float)
    metrics = compute_metrics_threshold(y_true, y_score, threshold=0.7)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["positive_rate_at_threshold"] == pytest.approx(0.75)
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None


def test_compute_classification_metrics_at_threshold_contains_counts() -> None:
    y_true = pd.Series([1, 1, 0, 0], dtype="Int64")
    y_score = np.array([0.9, 0.8, 0.7, 0.1], dtype=float)
    metrics = compute_classification_metrics_at_threshold(y_true, y_score, threshold=0.7)
    assert metrics["n"] == 4
    assert metrics["n_pos"] == 2
    assert metrics["prevalence"] == pytest.approx(0.5)
    assert metrics["positive_rate"] == pytest.approx(0.75)
    assert metrics["notes"] == []
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}


def test_compute_classification_metrics_single_class_sets_auc_none() -> None:
    y_true = pd.Series([1, 1, 1], dtype="Int64")
    y_score = np.array([0.2, 0.7, 0.9], dtype=float)
    metrics = compute_classification_metrics_at_threshold(y_true, y_score, threshold=0.5)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert "single_class_target" in metrics["notes"]
    assert metrics["confusion_matrix"] == {"tn": 0, "fp": 0, "fn": 1, "tp": 2}


def test_compute_confusion_counts_at_threshold_basic() -> None:
    y_true = pd.Series([0, 0, 1, 1], dtype="Int64")
    y_score = np.array([0.1, 0.6, 0.4, 0.9], dtype=float)
    cm = compute_confusion_counts_at_threshold(y_true, y_score, threshold=0.5)
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert sum(cm.values()) == 4


def test_compute_confusion_counts_at_threshold_single_class() -> None:
    y_true = pd.Series([1, 1, 1], dtype="Int64")
    y_score = np.array([0.2, 0.7, 0.9], dtype=float)
    cm = compute_confusion_counts_at_threshold(y_true, y_score, threshold=0.5)
    assert cm == {"tn": 0, "fp": 0, "fn": 1, "tp": 2}
    assert all(isinstance(value, int) for value in cm.values())


def test_compute_metrics_topk_frac_and_abs() -> None:
    y_true = pd.Series([1, 0, 1, 0, 0], dtype="Int64")
    y_score = np.array([0.95, 0.8, 0.7, 0.4, 0.3], dtype=float)
    topk_frac = compute_metrics_topk(y_true, y_score, k_frac=0.4)
    topk_abs = compute_metrics_topk(y_true, y_score, k_abs=2)
    assert topk_frac["k"] == 2
    assert topk_abs["k"] == 2
    assert topk_frac["recall"] == topk_abs["recall"]
    assert topk_frac["precision"] == topk_abs["precision"]


def test_select_threshold_for_target_recall() -> None:
    y_true = pd.Series([1, 1, 0, 0, 1], dtype="Int64")
    y_score = np.array([0.9, 0.7, 0.6, 0.2, 0.4], dtype=float)
    thr = select_threshold_for_target_recall(y_true, y_score, target_recall=2 / 3)
    metrics = compute_metrics_threshold(y_true, y_score, threshold=thr)
    assert metrics["recall"] >= (2 / 3) - 1e-12


def test_privacy_aggregated_only() -> None:
    y_true = pd.Series([1, 0, 1, 0], dtype="Int64")
    y_score = np.array([0.9, 0.2, 0.8, 0.1], dtype=float)
    payload = {
        "prevalence": compute_prevalence(y_true),
        "threshold": compute_metrics_threshold(y_true, y_score),
        "topk": compute_metrics_topk(y_true, y_score, k_frac=0.5),
        "policy": build_default_prediction_policy(),
    }
    forbidden = {"ids", "ra", "ra_list", "students", "rows"}
    assert forbidden.isdisjoint({str(key).lower() for key in payload.keys()})
    for section in payload.values():
        if isinstance(section, dict):
            for value in section.values():
                assert not isinstance(value, list)


def test_summarize_proba_expected_keys() -> None:
    summary = summarize_proba(np.array([0.1, 0.3, 0.7, 0.9], dtype=float))
    assert set(summary.keys()) == {"min", "mean", "p50", "p95", "max"}


def test_build_default_prediction_policy_defaults_to_threshold() -> None:
    policy = build_default_prediction_policy()
    assert policy["kind"] == "threshold"
    assert policy["threshold"] == pytest.approx(0.30)
    assert policy["k_frac"] == pytest.approx(0.20)


def test_build_default_prediction_policy_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        build_default_prediction_policy(kind="invalid")
