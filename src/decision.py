"""Decision helpers for probability-to-class conversion in serving paths."""

from __future__ import annotations


def decide_risk_class(risk_proba: float, threshold: float) -> int:
    """Return binary class from probability and threshold (both in [0,1])."""
    probability = float(risk_proba)
    threshold_value = float(threshold)
    if probability < 0.0 or probability > 1.0:
        raise ValueError("risk_proba must be within [0,1].")
    if threshold_value < 0.0 or threshold_value > 1.0:
        raise ValueError("threshold must be within [0,1].")
    return 1 if probability >= threshold_value else 0


__all__ = ["decide_risk_class"]

