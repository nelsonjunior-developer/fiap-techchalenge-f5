"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from app.deps import METADATA_PATH, get_serving_metadata
from app.routes import router
from src.utils import get_logger

_logger = get_logger(__name__)

app = FastAPI(
    title="PEDE Defasagem API",
    version="0.1.0",
)
app.include_router(router)


@app.on_event("startup")
def _on_startup() -> None:
    metadata, _ = get_serving_metadata()
    _logger.info("API started")
    _logger.info(
        "metadata_loaded %s | basename=%s",
        bool(metadata),
        METADATA_PATH.name,
    )


__all__ = ["app"]

