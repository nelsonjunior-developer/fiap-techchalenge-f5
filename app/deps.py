"""FastAPI dependencies for serving metadata/model status."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.serving_context import (
    extract_model_identity,
    extract_operational_threshold,
    load_serving_metadata,
)
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
def get_model() -> tuple[Any | None, dict[str, Any]]:
    """Lazy load promoted model.joblib for serving."""
    status: dict[str, Any] = {
        "model_loaded": False,
        "model_exists": bool(MODEL_PATH.exists()),
        "model_basename": MODEL_PATH.name,
        "notes": [],
    }
    notes: list[str] = []
    if not MODEL_PATH.exists():
        notes.append("model_joblib_not_found")
        status["notes"] = _dedupe_notes(notes)
        return None, status

    try:
        import joblib
    except ModuleNotFoundError:
        notes.append("joblib_not_available")
        status["notes"] = _dedupe_notes(notes)
        return None, status

    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        notes.append("model_load_failed")
        status["notes"] = _dedupe_notes(notes)
        return None, status

    status["model_loaded"] = True
    notes.append("model_loaded")
    status["notes"] = _dedupe_notes(notes)
    return model, status


@lru_cache(maxsize=1)
def get_prediction_context() -> dict[str, Any]:
    """Resolve metadata-backed prediction context (threshold/identity/contract)."""
    metadata, metadata_notes = get_serving_metadata()
    threshold, threshold_notes = extract_operational_threshold(metadata)
    identity, identity_notes = extract_model_identity(metadata)

    raw_cols = metadata.get("expected_raw_cols")
    expected_raw_cols: list[str] = []
    notes: list[str] = []
    notes.extend(metadata_notes)
    notes.extend(identity_notes)
    notes.extend(threshold_notes)

    if isinstance(raw_cols, list):
        expected_raw_cols = [str(col).strip() for col in raw_cols if str(col).strip()]
    if not expected_raw_cols:
        notes.append("expected_raw_cols_missing")

    return {
        "expected_raw_cols": expected_raw_cols,
        "threshold": float(threshold),
        "identity": identity,
        "metadata_loaded": bool(metadata),
        "metadata": metadata,
        "notes": _dedupe_notes(notes),
    }


@lru_cache(maxsize=1)
def get_model_loader_status() -> dict[str, Any]:
    """Return model + metadata loading status for diagnostics endpoints."""
    metadata, _ = get_serving_metadata()
    _, model_status = get_model()
    status = {
        "model_exists": bool(model_status.get("model_exists", False)),
        "metadata_exists": bool(METADATA_PATH.exists()),
        "metadata_loaded": bool(metadata),
        "model_loaded": bool(model_status.get("model_loaded", False)),
        "model_basename": MODEL_PATH.name,
        "metadata_basename": METADATA_PATH.name,
        "notes": [],
    }
    notes: list[str] = []
    notes.extend(list(model_status.get("notes", [])))
    if not status["metadata_exists"]:
        notes.append("metadata_json_not_found")
    status["notes"] = _dedupe_notes(notes)
    return status


__all__ = [
    "METADATA_PATH",
    "MODEL_DIR",
    "MODEL_PATH",
    "get_model",
    "get_model_loader_status",
    "get_prediction_context",
    "get_serving_metadata",
]
