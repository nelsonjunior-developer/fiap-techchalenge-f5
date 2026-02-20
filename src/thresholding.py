"""Threshold policy helpers for recall-focused operational decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics import compute_classification_metrics_at_threshold

_SELECTION_RULE = "max_precision_subject_to_recall>=0.90"


def evaluate_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate aggregate metrics and confusion matrix at a fixed threshold."""
    payload = compute_classification_metrics_at_threshold(
        y_true=y_true,
        y_proba=y_proba,
        threshold=float(threshold),
    )
    return {
        "threshold": float(payload["threshold"]),
        "n": int(payload["n"]),
        "n_pos": int(payload["n_pos"]),
        "prevalence": float(payload["prevalence"]),
        "metrics": {
            "recall": float(payload["recall"]),
            "precision": float(payload["precision"]),
            "f1": float(payload["f1"]),
            "roc_auc": payload["roc_auc"],
            "pr_auc": payload["pr_auc"],
            "positive_rate": float(payload["positive_rate"]),
        },
        "confusion_matrix": dict(payload["confusion_matrix"]),
        "notes": list(payload.get("notes", [])),
    }


def _build_grid(grid_size: int) -> np.ndarray:
    resolved = int(grid_size)
    if resolved < 2:
        raise ValueError("grid_size must be >= 2.")
    return np.linspace(0.0, 1.0, resolved, dtype=float)


def select_threshold_by_recall(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    recall_target: float = 0.90,
    grid_size: int = 2001,
) -> dict[str, Any]:
    """Select threshold by maximizing precision subject to recall target on train."""
    target = float(recall_target)
    grid = _build_grid(grid_size)

    best_eval: dict[str, Any] | None = None
    best_precision = -1.0
    best_threshold = -1.0
    for threshold in grid:
        evaluated = evaluate_at_threshold(y_true=y_true, y_proba=y_proba, threshold=float(threshold))
        recall_value = float(evaluated["metrics"]["recall"])
        if recall_value < target:
            continue
        precision_value = float(evaluated["metrics"]["precision"])
        if precision_value > best_precision or (
            np.isclose(precision_value, best_precision) and float(threshold) > best_threshold
        ):
            best_precision = precision_value
            best_threshold = float(threshold)
            best_eval = evaluated

    notes: list[str] = []
    if best_eval is None:
        threshold_selected = float(np.min(grid))
        best_eval = evaluate_at_threshold(
            y_true=y_true,
            y_proba=y_proba,
            threshold=threshold_selected,
        )
        notes.append("recall_target_not_met")
    else:
        threshold_selected = float(best_eval["threshold"])

    for note in list(best_eval.get("notes", [])):
        if note not in notes:
            notes.append(str(note))

    selection_rule = _SELECTION_RULE.replace("0.90", f"{target:.2f}")
    return {
        "threshold_selected": threshold_selected,
        "recall_target": target,
        "selection_rule": selection_rule,
        "grid_size": int(grid_size),
        "achieved_recall": float(best_eval["metrics"]["recall"]),
        "achieved_precision": float(best_eval["metrics"]["precision"]),
        "achieved_positive_rate": float(best_eval["metrics"]["positive_rate"]),
        "confusion_matrix_at_selected": dict(best_eval["confusion_matrix"]),
        "notes": notes,
    }
