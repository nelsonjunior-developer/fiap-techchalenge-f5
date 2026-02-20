"""Base API routes for health and version endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Body, HTTPException

import app.deps as deps
from app.predict_utils import (
    apply_leakage_gate_on_extras,
    build_raw_dataframe,
    normalize_records,
    validate_required_columns,
)
from app.request_schemas import PredictRequest
from app.schemas import PredictResponse, PredictionResult
from src.decision import decide_risk_class

router = APIRouter()
MAX_BATCH_SIZE = 500


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
    context = deps.get_prediction_context()
    status = deps.get_model_loader_status()

    notes: list[str] = []
    notes.extend(context.get("notes", []))
    notes.extend(status.get("notes", []))

    return {
        "model_version": context["identity"]["model_version"],
        "model_family": context["identity"]["model_family"],
        "variant": context["identity"]["variant"],
        "threshold_operational": float(context["threshold"]),
        "metadata_loaded": bool(status.get("metadata_loaded", False)),
        "notes": _dedupe_notes(notes),
    }


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest = Body(...)) -> PredictResponse:
    try:
        records = normalize_records(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if len(records) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "batch too large",
                "max_batch_size": int(MAX_BATCH_SIZE),
                "received": int(len(records)),
            },
        )

    context = deps.get_prediction_context()
    model, model_status = deps.get_model()

    expected_raw_cols = list(context.get("expected_raw_cols", []))
    metadata_loaded = bool(context.get("metadata_loaded", False))
    if not expected_raw_cols:
        raise HTTPException(
            status_code=503,
            detail={
                "detail": "metadata not available",
                "model_loaded": bool(model_status.get("model_loaded", False)),
                "metadata_loaded": metadata_loaded,
            },
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "detail": "model not available",
                "model_loaded": False,
                "metadata_loaded": metadata_loaded,
            },
        )

    df = build_raw_dataframe(records)
    try:
        apply_leakage_gate_on_extras(df, expected_raw_cols)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ok, missing_columns = validate_required_columns(df, expected_raw_cols)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Missing required columns",
                "missing_columns": missing_columns,
                "model_loaded": bool(model_status.get("model_loaded", False)),
                "metadata_loaded": metadata_loaded,
            },
        )

    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=500,
            detail="model does not expose predict_proba; probability output is required",
        )

    X_raw = df.reindex(columns=expected_raw_cols)
    try:
        proba_matrix = np.asarray(model.predict_proba(X_raw), dtype=float)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    if proba_matrix.ndim != 2 or proba_matrix.shape[1] < 2:
        raise HTTPException(
            status_code=500,
            detail="model predict_proba output must have shape (n, 2)",
        )
    risk_probas = proba_matrix[:, 1].tolist()

    threshold = float(context["threshold"])
    identity = dict(context.get("identity", {}))
    notes: list[str] = []
    notes.extend(list(context.get("notes", [])))
    notes.extend(list(model_status.get("notes", [])))
    deduped_notes = _dedupe_notes(notes)

    predictions: list[PredictionResult] = []
    for risk_proba in risk_probas:
        try:
            risk_class = decide_risk_class(float(risk_proba), threshold)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        predictions.append(
            PredictionResult(
                risk_proba=float(risk_proba),
                risk_class=int(risk_class),
                threshold_applied=threshold,
                model_version=str(identity.get("model_version", "unknown")),
                model_family=str(identity.get("model_family", "unknown")),
                variant=str(identity.get("variant", "unknown")),
                decision_policy="fixed_threshold",
                notes=deduped_notes or None,
            )
        )

    return PredictResponse(
        predictions=predictions,
        count=len(predictions),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


__all__ = ["router"]
