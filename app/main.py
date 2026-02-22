"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from app.deps import METADATA_PATH, get_prediction_context, get_serving_metadata
from app.routes import router
from src.online_metrics import append_online_event, summarize_online_batch
from src.utils import get_logger

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
    _logger.info("API started")
    _logger.info(
        "metadata_loaded %s | basename=%s",
        bool(metadata),
        METADATA_PATH.name,
    )
    yield


app = FastAPI(
    title="PEDE Defasagem API",
    version="0.1.0",
    lifespan=_lifespan,
)


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
    _logger.info(
        (
            "request_validation_summary | status_code=422 | method=%s | path=%s "
            "| error_count=%s | predict_route=%s | allow_partial_enabled=%s"
        ),
        request.method,
        path,
        int(error_count),
        bool(is_predict_route),
        (
            str(bool(allow_partial_enabled))
            if allow_partial_enabled is not None
            else "na"
        ),
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
            _logger.warning(
                "request_validation online metrics emit failed | exc=%s",
                inner_exc.__class__.__name__,
            )
    return await request_validation_exception_handler(request, exc)


app.include_router(router)


__all__ = ["app"]
