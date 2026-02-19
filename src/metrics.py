"""Metric helpers for threshold and top-k decision policies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_prevalence(y_true: pd.Series | np.ndarray) -> dict[str, float]:
    """Return aggregate prevalence stats without exposing any row-level data."""
    y = pd.Series(y_true).astype(int).to_numpy()
    n = int(len(y))
    n_pos = int(np.sum(y == 1))
    prevalence = float(n_pos / n) if n else 0.0
    return {"n": n, "n_pos": n_pos, "prevalence": prevalence}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _binary_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    u = sum_ranks_pos - positives * (positives + 1) / 2.0
    auc = u / (positives * negatives)
    return float(auc)


def _pr_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return None

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = 0.0
    fp = 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    for label in y_sorted:
        if int(label) == 1:
            tp += 1.0
        else:
            fp += 1.0
        precisions.append(_safe_div(tp, tp + fp))
        recalls.append(_safe_div(tp, positives))

    # Average precision approximation by step-wise integration over recall.
    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        if recall > prev_recall:
            ap += precision * (recall - prev_recall)
            prev_recall = recall
    return float(ap)


def summarize_proba(y_proba: pd.Series | np.ndarray) -> dict[str, float]:
    """Return privacy-safe aggregated score summary."""
    scores = np.asarray(y_proba, dtype=float)
    if scores.size == 0:
        return {
            "min": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(np.min(scores)),
        "mean": float(np.mean(scores)),
        "p50": float(np.quantile(scores, 0.50)),
        "p95": float(np.quantile(scores, 0.95)),
        "max": float(np.max(scores)),
    }


def compute_classification_metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute threshold metrics and dataset prevalence in a single payload.

    Returns only aggregate values (privacy-safe).
    """
    y = pd.Series(y_true).astype("Int64").to_numpy()
    scores = np.asarray(y_proba, dtype=float)
    if len(y) != len(scores):
        raise ValueError("y_true and y_proba must have same length.")

    n = int(len(y))
    notes: list[str] = []
    if n == 0:
        notes.append("empty_input")
        return {
            "threshold": float(threshold),
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "pr_auc": None,
            "positive_rate": 0.0,
            "n": 0,
            "n_pos": 0,
            "prevalence": 0.0,
            "notes": notes,
        }

    unique_values = set(pd.Series(y).dropna().astype(int).unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"y_true must be binary (0/1). Got values: {sorted(unique_values)}")

    y_pred = (scores >= float(threshold)).astype(int)
    precision, recall, f1 = _binary_precision_recall_f1(y.astype(int), y_pred)
    prevalence_info = compute_prevalence(y.astype(int))

    roc_auc = _roc_auc_safe(y.astype(int), scores)
    pr_auc = _pr_auc_safe(y.astype(int), scores)
    if len(unique_values) <= 1:
        roc_auc = None
        pr_auc = None
        notes.append("single_class_target")

    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "pr_auc": None if pr_auc is None else float(pr_auc),
        "positive_rate": float(np.mean(y_pred)),
        "n": int(prevalence_info["n"]),
        "n_pos": int(prevalence_info["n_pos"]),
        "prevalence": float(prevalence_info["prevalence"]),
        "notes": notes,
    }


def compute_metrics_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Backward-compatible threshold metrics payload used across project."""
    payload = compute_classification_metrics_at_threshold(
        y_true,
        y_score,
        threshold=threshold,
    )
    return {
        "threshold": float(payload["threshold"]),
        "recall": float(payload["recall"]),
        "precision": float(payload["precision"]),
        "f1": float(payload["f1"]),
        "roc_auc": payload["roc_auc"],
        "pr_auc": payload["pr_auc"],
        "positive_rate_at_threshold": float(payload["positive_rate"]),
    }


def compute_metrics_topk(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    k_frac: float | None = None,
    k_abs: int | None = None,
) -> dict[str, float | int]:
    """Compute aggregated recall/precision for top-k ranking policy."""
    if (k_frac is None) == (k_abs is None):
        raise ValueError("Provide exactly one of k_frac or k_abs.")

    y = pd.Series(y_true).astype(int).to_numpy()
    scores = np.asarray(y_score, dtype=float)
    if len(y) != len(scores):
        raise ValueError("y_true and y_score must have same length.")
    n = len(y)
    if n == 0:
        raise ValueError("Empty inputs are not allowed.")

    if k_abs is None:
        k_frac_value = float(k_frac)  # type: ignore[arg-type]
        if not (0.0 < k_frac_value <= 1.0):
            raise ValueError("k_frac must be in (0, 1].")
        k = max(1, int(np.ceil(k_frac_value * n)))
    else:
        k = int(k_abs)
        if k <= 0 or k > n:
            raise ValueError("k_abs must be in [1, n_samples].")
        k_frac_value = float(k / n)

    idx = np.argsort(-scores)[:k]
    y_pred = np.zeros(n, dtype=int)
    y_pred[idx] = 1
    precision, recall, _ = _binary_precision_recall_f1(y, y_pred)
    return {
        "k_frac": float(k_frac_value),
        "k": int(k),
        "recall": float(recall),
        "precision": float(precision),
        "positive_rate": float(np.mean(y_pred)),
    }


def select_threshold_for_target_recall(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    target_recall: float = 0.85,
) -> float:
    """Pick threshold maximizing precision under recall constraint on train."""
    y = pd.Series(y_true).astype(int).to_numpy()
    scores = np.asarray(y_score, dtype=float)
    if len(y) != len(scores):
        raise ValueError("y_true and y_score must have same length.")
    if len(y) == 0:
        raise ValueError("Empty inputs are not allowed.")

    candidates = np.unique(scores)
    best_precision = -1.0
    best_threshold: float | None = None
    for threshold in candidates:
        y_pred = (scores >= threshold).astype(int)
        precision, recall, _ = _binary_precision_recall_f1(y, y_pred)
        if recall < float(target_recall):
            continue
        if precision > best_precision or (
            np.isclose(precision, best_precision) and best_threshold is not None and threshold > best_threshold
        ):
            best_precision = float(precision)
            best_threshold = float(threshold)

    if best_threshold is not None:
        return float(best_threshold)

    # Fallback when target recall is unattainable: maximize recall then precision.
    best_key: tuple[float, float, float] | None = None
    best_threshold = float(candidates[0])
    for threshold in candidates:
        y_pred = (scores >= threshold).astype(int)
        precision, recall, _ = _binary_precision_recall_f1(y, y_pred)
        key = (float(recall), float(precision), -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return float(best_threshold)


def build_default_prediction_policy(
    *,
    k_frac: float = 0.10,
    threshold: float = 0.50,
    score_name: str = "risk_proba",
) -> dict[str, Any]:
    """Return default decision policy config saved with model metadata."""
    return {
        "kind": "top_k",
        "k_frac": float(k_frac),
        "threshold": float(threshold),
        "score_name": str(score_name),
    }
