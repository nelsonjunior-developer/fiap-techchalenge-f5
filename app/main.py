"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from app.deps import METADATA_PATH, get_prediction_context, get_serving_metadata
from app.routes import router
from src.online_metrics import append_online_event, summarize_online_batch
from src.utils import (
    get_logger,
    log_event,
    reset_request_id_context,
    set_request_id_context,
)

_logger = get_logger(__name__)


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize lightweight serving diagnostics without deprecated on_event hooks."""
    metadata, _ = get_serving_metadata()
    log_event(
        _logger,
        "api_started",
        context={"component": "fastapi", "service": "pede-defasagem-api"},
    )
    log_event(
        _logger,
        "metadata_status",
        message=f"metadata_loaded {bool(metadata)} | basename={METADATA_PATH.name}",
        context={"metadata_loaded": bool(metadata), "basename": METADATA_PATH.name},
    )
    yield


app = FastAPI(
    title="PEDE Defasagem API",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    request_id = uuid4().hex[:12]
    request.state.request_id = request_id
    token = set_request_id_context(request_id)
    response = None
    try:
        response = await call_next(request)
        try:
            response.headers["X-Request-ID"] = request_id
        except Exception:
            # Keep request processing resilient even if a custom response object misbehaves.
            pass
        return response
    finally:
        reset_request_id_context(token)


@app.exception_handler(RequestValidationError)
async def _request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    path = request.url.path
    error_count = len(exc.errors()) if isinstance(exc.errors(), list) else 0
    is_predict_route = path == "/predict"
    allow_partial_enabled = (
        _env_flag_enabled("ALLOW_PARTIAL_PAYLOAD", default=False)
        if is_predict_route
        else None
    )
    allow_partial_label = (
        str(bool(allow_partial_enabled)) if allow_partial_enabled is not None else "na"
    )
    log_event(
        _logger,
        "request_validation_422",
        message=(
            "request_validation_summary | status_code=422 | method=%s | path=%s "
            "| error_count=%s | predict_route=%s | allow_partial_enabled=%s"
        )
        % (
            request.method,
            path,
            int(error_count),
            bool(is_predict_route),
            allow_partial_label,
        ),
        level=logging.INFO,
        context={
            "status_code": 422,
            "method": request.method,
            "path": path,
            "error_count": int(error_count),
            "predict_route": bool(is_predict_route),
            "allow_partial_enabled": allow_partial_enabled,
        },
    )
    if is_predict_route:
        try:
            context = get_prediction_context()
            identity = dict(context.get("identity", {}))
            event = summarize_online_batch(
                None,
                context.get("threshold"),
                None,
                422,
                str(identity.get("model_version", "unknown")),
                model_family=str(identity.get("model_family", "unknown")),
                variant=str(identity.get("variant", "unknown")),
                n_records=0,
                reason_code="request_validation_422",
            )
            append_online_event(
                event,
                path=os.getenv("ONLINE_METRICS_PATH", "logs/online_metrics.jsonl"),
            )
        except Exception as inner_exc:  # pragma: no cover - defensive
            log_event(
                _logger,
                "online_metrics_emit_failed",
                level=logging.WARNING,
                message=(
                    "request_validation online metrics emit failed | exc=%s"
                    % inner_exc.__class__.__name__
                ),
                context={
                    "status_code": 422,
                    "path": path,
                    "exc_type": inner_exc.__class__.__name__,
                },
            )
    return await request_validation_exception_handler(request, exc)


app.include_router(router)


__all__ = ["app"]
