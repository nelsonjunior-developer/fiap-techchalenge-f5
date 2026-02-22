"""Thresholds for model non-regression checks.

Source of truth:
- Minimum holdout gates reuse the same defaults from `src.promotion_policy`
  to avoid drift between champion selection/promotion and regression tests.
"""

from __future__ import annotations

from src.promotion_policy import DEFAULT_MIN_PRAUC_HOLDOUT, DEFAULT_MIN_RECALL_HOLDOUT

MIN_RECALL_HOLDOUT_AT_030 = float(DEFAULT_MIN_RECALL_HOLDOUT)
MIN_PRAUC_HOLDOUT = float(DEFAULT_MIN_PRAUC_HOLDOUT)
THRESHOLD_PREFERRED = 0.30
FALLBACK_THRESHOLD = 0.50
ALLOW_FALLBACK_05 = True


__all__ = [
    "ALLOW_FALLBACK_05",
    "FALLBACK_THRESHOLD",
    "MIN_PRAUC_HOLDOUT",
    "MIN_RECALL_HOLDOUT_AT_030",
    "THRESHOLD_PREFERRED",
]

