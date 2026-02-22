"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.deps import METADATA_PATH, get_serving_metadata
from app.routes import router
from src.utils import get_logger

_logger = get_logger(__name__)


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
app.include_router(router)


__all__ = ["app"]
