"""Prometheus metrics for API observability (local, privacy-safe)."""

from __future__ import annotations

from time import perf_counter
from typing import Final

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

HTTP_LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the API.",
    labelnames=("method", "path", "status"),
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "path"),
    buckets=HTTP_LATENCY_BUCKETS,
)
INFERENCE_RECORDS_TOTAL = Counter(
    "inference_records_total",
    "Total records processed by inference endpoints.",
    labelnames=("endpoint",),
)
INFERENCE_POSITIVE_TOTAL = Counter(
    "inference_positive_total",
    "Total positive predictions at the operational threshold.",
    labelnames=("endpoint", "threshold"),
)
MODEL_LOADED = Gauge(
    "model_loaded",
    "Model availability gauge (1=loaded, 0=not loaded).",
)
METADATA_LOADED = Gauge(
    "metadata_loaded",
    "Serving metadata availability gauge (1=loaded, 0=not loaded).",
)


def now_perf_counter() -> float:
    """Return monotonic high-resolution timer value."""
    return perf_counter()


def route_path_label(request: Request) -> str:
    """
    Resolve a low-cardinality path label for metrics.

    Prefer route template (`/predict`) and fall back to raw URL path if route is not resolved.
    """
    route = request.scope.get("route")
    path_template = getattr(route, "path", None) if route is not None else None
    if isinstance(path_template, str) and path_template.strip():
        return path_template
    return "/__unmatched__"


def observe_http_request(
    *,
    method: str,
    path: str,
    status_code: int,
    latency_seconds: float,
) -> None:
    """Record request count + latency histogram."""
    status_label = str(int(status_code))
    safe_method = str(method or "UNKNOWN").upper()
    safe_path = str(path or "/")
    safe_latency = max(float(latency_seconds), 0.0)
    HTTP_REQUESTS_TOTAL.labels(
        method=safe_method,
        path=safe_path,
        status=status_label,
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=safe_method,
        path=safe_path,
    ).observe(safe_latency)


def observe_inference_batch(
    *,
    endpoint: str,
    n_records: int,
    n_positives: int,
    threshold: float | None,
) -> None:
    """Record aggregate inference counters for successful predictions."""
    safe_endpoint = str(endpoint or "/predict")
    records = max(int(n_records), 0)
    positives = max(int(n_positives), 0)
    if records > 0:
        INFERENCE_RECORDS_TOTAL.labels(endpoint=safe_endpoint).inc(records)
    if positives > 0:
        threshold_label = (
            f"{float(threshold):.2f}" if threshold is not None else "unknown"
        )
        INFERENCE_POSITIVE_TOTAL.labels(
            endpoint=safe_endpoint,
            threshold=threshold_label,
        ).inc(positives)


def set_model_metadata_gauges(
    *,
    model_loaded: bool | None,
    metadata_loaded: bool | None,
) -> None:
    """Update model/metadata availability gauges."""
    if model_loaded is not None:
        MODEL_LOADED.set(1 if bool(model_loaded) else 0)
    if metadata_loaded is not None:
        METADATA_LOADED.set(1 if bool(metadata_loaded) else 0)


def build_metrics_response() -> Response:
    """Expose current registry in Prometheus text format."""
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


__all__ = [
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION_SECONDS",
    "INFERENCE_RECORDS_TOTAL",
    "INFERENCE_POSITIVE_TOTAL",
    "MODEL_LOADED",
    "METADATA_LOADED",
    "build_metrics_response",
    "now_perf_counter",
    "observe_http_request",
    "observe_inference_batch",
    "route_path_label",
    "set_model_metadata_gauges",
]
