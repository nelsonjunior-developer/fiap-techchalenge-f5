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


def _model_paths(model_dir: str | Path | None = None) -> dict[str, Path]:
    """Return canonical serving paths for model + metadata."""
    base_dir = Path(model_dir) if model_dir is not None else Path(MODEL_DIR)
    return {
        "model_path": base_dir / "model.joblib",
        "metadata_path": base_dir / "metadata.json",
    }


def load_model_joblib(model_path: str | Path) -> tuple[Any | None, list[str]]:
    """Load promoted joblib model with safe diagnostics (no stacktrace in responses)."""
    path = Path(model_path)

    try:
        import joblib
    except ModuleNotFoundError:
        _logger.warning("joblib unavailable | basename=%s", path.name)
        return None, ["joblib_not_available"]

    try:
        model = joblib.load(path)
    except FileNotFoundError:
        _logger.info("model file missing | basename=%s", path.name)
        return None, ["model_file_missing"]
    except Exception as exc:  # pragma: no cover - exception types vary by environment
        exc_name = exc.__class__.__name__
        _logger.warning(
            "model load failed | basename=%s | exc=%s",
            path.name,
            exc_name,
        )
        return None, [f"model_load_failed:{exc_name}"]

    _logger.info("model_loaded %s | basename=%s", True, path.name)
    return model, ["model_loaded_ok"]


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
def get_model() -> dict[str, Any]:
    """Lazy load promoted model.joblib for serving (cached process-local state)."""
    paths = _model_paths(MODEL_DIR)
    model_path = paths["model_path"]
    model, notes = load_model_joblib(model_path)
    model_exists = bool(model_path.exists())
    return {
        "model": model,
        "model_loaded": bool(model is not None),
        "model_exists": model_exists,  # backwards-compatible alias
        "model_joblib_exists": model_exists,
        "model_basename": model_path.name,  # backwards-compatible alias
        "model_path_basename": model_path.name,
        "notes": _dedupe_notes(notes),
    }


def invalidate_model_cache() -> None:
    """Clear cached model loader state (useful for tests and future hot-reload)."""
    get_model.cache_clear()
    get_model_loader_status.cache_clear()


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
        if metadata:
            notes.append("metadata_invalid_missing_expected_raw_cols")

    return {
        "expected_raw_cols": expected_raw_cols,
        "threshold": float(threshold),
        "identity": identity,
        "metadata_loaded": bool(metadata),
        "metadata_contract_valid": bool(expected_raw_cols),
        "metadata": metadata,
        "notes": _dedupe_notes(notes),
    }


@lru_cache(maxsize=1)
def get_model_loader_status() -> dict[str, Any]:
    """Return model + metadata loading status for diagnostics endpoints."""
    metadata, _ = get_serving_metadata()
    model_status = get_model()
    status = {
        "model_exists": bool(model_status.get("model_exists", False)),
        "model_joblib_exists": bool(model_status.get("model_joblib_exists", False)),
        "metadata_exists": bool(METADATA_PATH.exists()),
        "metadata_loaded": bool(metadata),
        "model_loaded": bool(model_status.get("model_loaded", False)),
        "model_basename": MODEL_PATH.name,
        "model_path_basename": str(model_status.get("model_path_basename", MODEL_PATH.name)),
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
    "invalidate_model_cache",
    "get_model",
    "get_model_loader_status",
    "get_prediction_context",
    "get_serving_metadata",
    "load_model_joblib",
]
