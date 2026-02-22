"""Privacy-safe aggregated online metrics for inference requests/batches."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils import get_logger

_logger = get_logger(__name__)
DEFAULT_ONLINE_METRICS_PATH = "logs/online_metrics.jsonl"
DEFAULT_HISTOGRAM_BINS: tuple[float, ...] = tuple(i / 10 for i in range(11))


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_family(status_code: int) -> str:
    code = int(status_code)
    if 200 <= code < 300:
        return "2xx"
    if 400 <= code < 500:
        return "4xx"
    if 500 <= code < 600:
        return "5xx"
    return "other"


def _normalize_bins(bins: Sequence[float] | None) -> list[float]:
    raw = list(DEFAULT_HISTOGRAM_BINS if bins is None else bins)
    if len(raw) < 2:
        raise ValueError("bins must contain at least 2 edges.")
    normalized = [float(value) for value in raw]
    if any(np.isnan(normalized)):
        raise ValueError("bins must not contain NaN.")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("bins must be strictly increasing.")
    return normalized


def build_score_histogram(probas: list[float], bins: list[float]) -> dict[str, Any]:
    """Return histogram counts for risk scores without storing individual probabilities."""
    edges = _normalize_bins(bins)
    scores = np.asarray(list(probas), dtype=float)
    if scores.ndim != 1:
        raise ValueError("probas must be a 1-D list of floats.")
    if np.isnan(scores).any():
        raise ValueError("probas must not contain NaN.")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("probas values must be in [0, 1].")

    counts, out_edges = np.histogram(scores, bins=np.asarray(edges, dtype=float))
    return {
        "bin_edges": [float(value) for value in out_edges.tolist()],
        "bin_counts": [int(value) for value in counts.tolist()],
        "n_bins": int(len(counts)),
        "total": int(len(scores)),
    }


def summarize_online_batch(
    probas: Sequence[float] | None,
    threshold: float | None,
    missing_stats: Mapping[str, Any] | None,
    status_code: int,
    model_version: str,
    *,
    model_family: str | None = None,
    variant: str | None = None,
    n_records: int | None = None,
    reason_code: str | None = None,
    bins: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build one aggregated online monitoring event for a request/batch.

    Error requests may omit `probas`. In that case, score-related aggregates are null.
    `error_rate` must be derived downstream by aggregating `status_family` over many events.
    """
    threshold_value = _safe_float(threshold)
    scores: list[float] | None = None if probas is None else [float(p) for p in probas]
    n_records_value = int(len(scores)) if scores is not None else int(n_records or 0)

    histogram: dict[str, Any] | None = None
    positive_rate_at_threshold: float | None = None
    n_positive_at_threshold: int | None = None
    if scores is not None:
        histogram = build_score_histogram(list(scores), list(_normalize_bins(bins)))
        if threshold_value is not None:
            n_positive_at_threshold = int(sum(1 for score in scores if float(score) >= threshold_value))
            positive_rate_at_threshold = (
                float(n_positive_at_threshold / n_records_value) if n_records_value else 0.0
            )

    missing = dict(missing_stats or {})
    event = {
        "event_type": "online_inference_aggregate",
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_code": int(status_code),
        "status_family": _status_family(int(status_code)),
        "reason_code": None if not reason_code else str(reason_code),
        "model_version": str(model_version or "unknown"),
        "model_family": str(model_family or "unknown"),
        "variant": str(variant or "unknown"),
        "threshold": threshold_value,
        "n_records": int(n_records_value),
        "n_positive_at_threshold": n_positive_at_threshold,
        "positive_rate_at_threshold": positive_rate_at_threshold,
        "score_histogram": histogram,
        "missing_cols_rate": _safe_float(missing.get("missing_cols_rate")),
        "missing_values_rate": _safe_float(missing.get("missing_values_rate")),
        "missing_non_structural_cols_rate": _safe_float(
            missing.get("missing_non_structural_cols_rate")
        ),
        "missing_cols_count": _safe_int(missing.get("missing_cols_count")),
        "expected_cols_count": _safe_int(missing.get("expected_cols_count")),
        "extra_cols_count": _safe_int(missing.get("extra_cols_count")),
        "allow_partial_enabled": bool(missing.get("allow_partial_enabled", False))
        if missing
        else None,
        "allow_partial_used": bool(missing.get("allow_partial_used", False)) if missing else None,
    }
    return event


def append_online_event(event: dict, path: str = DEFAULT_ONLINE_METRICS_PATH) -> None:
    """Append one aggregated event as JSONL. Best-effort: never break the API on I/O failures."""
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - environment/fs dependent
        _logger.warning(
            "online metrics append failed | basename=%s | exc=%s",
            output_path.name,
            exc.__class__.__name__,
        )


__all__ = [
    "DEFAULT_HISTOGRAM_BINS",
    "DEFAULT_ONLINE_METRICS_PATH",
    "append_online_event",
    "build_score_histogram",
    "summarize_online_batch",
]
