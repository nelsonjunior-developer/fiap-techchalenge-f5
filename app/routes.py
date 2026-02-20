"""Base API routes for health and version endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import get_model_loader_status, get_serving_metadata
from src.serving_context import extract_model_identity, extract_operational_threshold

router = APIRouter()


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


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/version")
def version() -> dict[str, object]:
    metadata, metadata_notes = get_serving_metadata()
    identity, identity_notes = extract_model_identity(metadata)
    threshold, threshold_notes = extract_operational_threshold(metadata)
    status = get_model_loader_status()

    notes: list[str] = []
    notes.extend(metadata_notes)
    notes.extend(identity_notes)
    notes.extend(threshold_notes)
    notes.extend(status.get("notes", []))

    return {
        "model_version": identity["model_version"],
        "model_family": identity["model_family"],
        "variant": identity["variant"],
        "threshold_operational": float(threshold),
        "metadata_loaded": bool(status.get("metadata_loaded", False)),
        "notes": _dedupe_notes(notes),
    }


__all__ = ["router"]

