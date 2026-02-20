"""FastAPI dependencies for serving metadata/model status."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.serving_context import load_serving_metadata
from src.utils import get_logger

_logger = get_logger(__name__)

MODEL_DIR = Path("app/model")
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


def _dedupe_notes(notes: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for note in notes:
        normalized = str(note).strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


@lru_cache(maxsize=1)
def get_serving_metadata() -> tuple[dict[str, Any], list[str]]:
    """Return serving metadata + loading notes with simple process-local cache."""
    metadata = load_serving_metadata(METADATA_PATH)
    notes: list[str] = []
    if metadata:
        notes.append("metadata_loaded")
    else:
        notes.append("metadata_missing_or_invalid")
    _logger.info(
        "metadata_loaded %s | basename=%s",
        bool(metadata),
        METADATA_PATH.name,
    )
    return metadata, _dedupe_notes(notes)


@lru_cache(maxsize=1)
def get_model_loader_status() -> dict[str, Any]:
    """Return model loading status without loading joblib yet (stub for next task)."""
    metadata, _ = get_serving_metadata()
    status = {
        "model_exists": bool(MODEL_PATH.exists()),
        "metadata_exists": bool(METADATA_PATH.exists()),
        "metadata_loaded": bool(metadata),
        "model_loaded": False,
        "model_basename": MODEL_PATH.name,
        "metadata_basename": METADATA_PATH.name,
        "notes": [],
    }
    notes: list[str] = []
    if not status["model_exists"]:
        notes.append("model_joblib_not_found")
    if not status["metadata_exists"]:
        notes.append("metadata_json_not_found")
    notes.append("model_loading_stub_only")
    status["notes"] = _dedupe_notes(notes)
    return status


__all__ = [
    "METADATA_PATH",
    "MODEL_DIR",
    "MODEL_PATH",
    "get_model_loader_status",
    "get_serving_metadata",
]

