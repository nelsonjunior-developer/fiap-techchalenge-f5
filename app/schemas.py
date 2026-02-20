"""Formal output schema for prediction responses used by the future API."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError, confloat, root_validator, validator
from src.decision import decide_risk_class
from src.serving_context import (
    extract_model_identity,
    extract_operational_threshold,
    load_serving_metadata as _load_serving_metadata,
)

DEFAULT_OPERATIONAL_THRESHOLD = 0.30
DEFAULT_MODEL_VERSION = "unknown"
DEFAULT_MODEL_FAMILY = "unknown"
DEFAULT_VARIANT = "unknown"


def _parse_iso8601(value: str) -> datetime:
    normalized = str(value).strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return datetime.fromisoformat(normalized)


def _dedupe_notes(notes: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for note in notes:
        text = str(note).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def load_serving_metadata(path: str | Path = "app/model/metadata.json") -> dict[str, Any]:
    """Load serving metadata proxy for backward-compatible imports."""
    return _load_serving_metadata(path)


def resolve_prediction_context(metadata: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Resolve threshold/model identity with defaults for legacy metadata payloads."""
    meta = dict(metadata) if isinstance(metadata, Mapping) else {}
    threshold_value, threshold_notes = extract_operational_threshold(
        meta,
        default_threshold=DEFAULT_OPERATIONAL_THRESHOLD,
    )
    identity, identity_notes = extract_model_identity(meta)
    notes = list(threshold_notes) + list(identity_notes)

    return (
        {
            "threshold_applied": float(threshold_value),
            "model_version": str(identity["model_version"] or DEFAULT_MODEL_VERSION),
            "model_family": str(identity["model_family"] or DEFAULT_MODEL_FAMILY),
            "variant": str(identity["variant"] or DEFAULT_VARIANT),
        },
        _dedupe_notes(notes),
    )


def derive_risk_class(*, risk_proba: float, threshold_applied: float) -> int:
    """Backward-compatible wrapper over src.decision.decide_risk_class."""
    return decide_risk_class(risk_proba=risk_proba, threshold=threshold_applied)


class PredictionResult(BaseModel):
    """Single prediction output row contract."""

    risk_proba: confloat(ge=0.0, le=1.0) = Field(..., description="Predicted risk probability [0,1].")
    risk_class: int = Field(..., description="Derived class (0|1) from risk_proba >= threshold_applied.")
    threshold_applied: confloat(ge=0.0, le=1.0) = Field(..., description="Operational threshold used.")
    model_version: str
    model_family: str
    variant: str
    decision_policy: Literal["fixed_threshold", "topk"] = "fixed_threshold"
    notes: Optional[list[str]] = None

    @validator("risk_class")
    def _validate_risk_class(cls, value: int) -> int:
        if value not in {0, 1}:
            raise ValueError("risk_class must be 0 or 1.")
        return int(value)

    @validator("model_version", "model_family", "variant")
    def _validate_non_empty_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("must be a non-empty string.")
        return normalized

    @root_validator
    def _validate_derived_class(cls, values: dict[str, Any]) -> dict[str, Any]:
        risk_proba = values.get("risk_proba")
        threshold_applied = values.get("threshold_applied")
        risk_class = values.get("risk_class")
        if risk_proba is None or threshold_applied is None or risk_class is None:
            return values
        derived = derive_risk_class(
            risk_proba=float(risk_proba),
            threshold_applied=float(threshold_applied),
        )
        if int(risk_class) != int(derived):
            raise ValueError(
                "risk_class must be derived from risk_proba and threshold_applied."
            )
        return values


class PredictResponse(BaseModel):
    """Batch prediction output contract."""

    predictions: list[PredictionResult]
    count: int
    generated_at: str

    @validator("generated_at")
    def _validate_generated_at(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("generated_at must be a non-empty ISO8601 string.")
        _parse_iso8601(value)
        return value

    @root_validator
    def _validate_count(cls, values: dict[str, Any]) -> dict[str, Any]:
        predictions = values.get("predictions")
        count = values.get("count")
        if isinstance(predictions, list) and isinstance(count, int):
            if count != len(predictions):
                raise ValueError("count must match len(predictions).")
        return values


def build_prediction_result(
    *,
    risk_proba: float,
    metadata: Mapping[str, Any] | None = None,
    decision_policy: Literal["fixed_threshold", "topk"] = "fixed_threshold",
    notes: Sequence[str] | None = None,
) -> PredictionResult:
    """Build one validated prediction result from score + metadata context."""
    context, fallback_notes = resolve_prediction_context(metadata)
    threshold_applied = float(context["threshold_applied"])
    probability = float(risk_proba)
    risk_class = derive_risk_class(
        risk_proba=probability,
        threshold_applied=threshold_applied,
    )

    merged_notes: list[str] = []
    merged_notes.extend(fallback_notes)
    if notes:
        merged_notes.extend(str(note) for note in notes)

    payload = {
        "risk_proba": probability,
        "risk_class": int(risk_class),
        "threshold_applied": threshold_applied,
        "model_version": str(context["model_version"]),
        "model_family": str(context["model_family"]),
        "variant": str(context["variant"]),
        "decision_policy": decision_policy,
        "notes": _dedupe_notes(merged_notes) or None,
    }
    return PredictionResult(**payload)


def build_predict_response(
    *,
    risk_probas: Sequence[float],
    metadata: Mapping[str, Any] | None = None,
    decision_policy: Literal["fixed_threshold", "topk"] = "fixed_threshold",
    notes: Sequence[str] | None = None,
    generated_at: str | None = None,
) -> PredictResponse:
    """Build validated batch response with deterministic schema."""
    predictions = [
        build_prediction_result(
            risk_proba=float(probability),
            metadata=metadata,
            decision_policy=decision_policy,
            notes=notes,
        )
        for probability in risk_probas
    ]
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    return PredictResponse(
        predictions=predictions,
        count=len(predictions),
        generated_at=timestamp,
    )


__all__ = [
    "PredictionResult",
    "PredictResponse",
    "build_prediction_result",
    "build_predict_response",
    "derive_risk_class",
    "load_serving_metadata",
    "resolve_prediction_context",
    "ValidationError",
]
