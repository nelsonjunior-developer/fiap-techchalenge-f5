from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.thresholding import evaluate_at_threshold, select_threshold_by_recall


def test_select_threshold_by_recall_is_deterministic() -> None:
    y_true = pd.Series([1, 1, 0, 0], dtype="Int64")
    y_proba = np.array([0.9, 0.8, 0.7, 0.1], dtype=float)
    result = select_threshold_by_recall(
        y_true=y_true,
        y_proba=y_proba,
        recall_target=1.0,
        grid_size=11,
    )
    assert result["threshold_selected"] == pytest.approx(0.8)
    assert result["achieved_recall"] == pytest.approx(1.0)
    assert result["achieved_precision"] == pytest.approx(1.0)
    assert result["notes"] == []


def test_select_threshold_by_recall_impossible_target_marks_note() -> None:
    y_true = pd.Series([0, 0, 0], dtype="Int64")
    y_proba = np.array([0.2, 0.5, 0.8], dtype=float)
    result = select_threshold_by_recall(
        y_true=y_true,
        y_proba=y_proba,
        recall_target=0.9,
        grid_size=21,
    )
    assert result["threshold_selected"] == pytest.approx(0.0)
    assert "recall_target_not_met" in result["notes"]


def test_evaluate_at_threshold_returns_flat_confusion_matrix() -> None:
    y_true = pd.Series([0, 0, 1, 1], dtype="Int64")
    y_proba = np.array([0.1, 0.7, 0.2, 0.8], dtype=float)
    result = evaluate_at_threshold(y_true=y_true, y_proba=y_proba, threshold=0.5)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert set(result["confusion_matrix"].keys()) == {"tn", "fp", "fn", "tp"}
