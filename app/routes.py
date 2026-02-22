"""Base API routes for health and version endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import numpy as np
from fastapi import APIRouter, Body, HTTPException

import app.deps as deps
from app.predict_utils import (
    build_raw_dataframe,
    build_model_input_frame,
    MissingColumnsError,
    normalize_records,
)
from app.request_schemas import PredictRequest
from app.schemas import PredictResponse, PredictionResult
from src.decision import decide_risk_class
from src.online_metrics import append_online_event, summarize_online_batch
from src.utils import get_logger

router = APIRouter()
MAX_BATCH_SIZE = 500
_logger = get_logger(__name__)


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


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _format_rate(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "na"


def _build_missing_stats_notes(missing_stats: dict[str, Any] | None) -> list[str]:
    if not isinstance(missing_stats, dict) or not missing_stats:
        return []

    notes = [
        f"missing_cols_count={int(missing_stats.get('missing_cols_count', 0))}",
        f"missing_cols_rate={float(missing_stats.get('missing_cols_rate', 0.0)):.2f}",
        f"missing_values_rate={float(missing_stats.get('missing_values_rate', 0.0)):.2f}",
    ]
    if "missing_non_structural_cols_rate" in missing_stats:
        notes.append(
            "missing_non_structural_cols_rate="
            f"{float(missing_stats.get('missing_non_structural_cols_rate', 0.0)):.2f}"
        )
    if bool(missing_stats.get("allow_partial_used", False)):
        notes.append("allow_partial_payload=1")

    missing_cols_count = int(missing_stats.get("missing_cols_count", 0))
    top_missing_cols = list(missing_stats.get("top_missing_cols", []))
    if 0 < missing_cols_count <= 10 and top_missing_cols:
        notes.append(
            "top_missing_cols=" + ",".join(str(col) for col in top_missing_cols[:10])
        )
    return notes


def _log_predict_request_summary(
    *,
    status_code: int,
    count_records: int,
    allow_partial_enabled: bool,
    missing_stats: dict[str, Any] | None,
) -> None:
    stats = dict(missing_stats or {})
    _logger.info(
        (
            "predict_request_summary | status_code=%s | count_records=%s "
            "| allow_partial_enabled=%s | allow_partial_used=%s "
            "| expected_cols_count=%s | present_cols_count=%s | missing_cols_count=%s "
            "| missing_cols_rate=%s | missing_non_structural_cols_rate=%s "
            "| missing_values_rate=%s | extra_cols_count=%s"
        ),
        int(status_code),
        int(count_records),
        bool(allow_partial_enabled),
        bool(stats.get("allow_partial_used", False)),
        stats.get("expected_cols_count"),
        stats.get("present_cols_count"),
        stats.get("missing_cols_count"),
        _format_rate(stats.get("missing_cols_rate")),
        _format_rate(stats.get("missing_non_structural_cols_rate")),
        _format_rate(stats.get("missing_values_rate")),
        stats.get("extra_cols_count"),
    )


def _append_predict_online_metrics_event(
    *,
    status_code: int,
    count_records: int,
    risk_probas: list[float] | None,
    threshold: float | None,
    missing_stats: dict[str, Any] | None,
    model_version: str,
    model_family: str,
    variant: str,
    reason_code: str | None,
) -> None:
    try:
        event = summarize_online_batch(
            risk_probas,
            threshold,
            missing_stats,
            int(status_code),
            model_version,
            model_family=model_family,
            variant=variant,
            n_records=int(count_records),
            reason_code=reason_code,
        )
        append_online_event(
            event,
            path=os.getenv("ONLINE_METRICS_PATH", "logs/online_metrics.jsonl"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning(
            "predict online metrics emit failed | status_code=%s | exc=%s",
            int(status_code),
            exc.__class__.__name__,
        )


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
        "model_loaded": bool(status.get("model_loaded", False)),
        "model_joblib_exists": bool(status.get("model_joblib_exists", False)),
        "model_notes": list(status.get("notes", [])),
        "notes": _dedupe_notes(notes),
    }


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest = Body(...)) -> PredictResponse:
    allow_partial_enabled = _env_flag_enabled("ALLOW_PARTIAL_PAYLOAD", default=False)
    status_code_for_log = 500
    count_records = 0
    missing_stats: dict[str, Any] | None = None
    risk_probas_for_online: list[float] | None = None
    reason_code_for_online: str | None = None
    model_version_for_online = "unknown"
    model_family_for_online = "unknown"
    variant_for_online = "unknown"
    threshold_for_online: float | None = None

    try:
        try:
            records = normalize_records(payload)
        except ValueError as exc:
            reason_code_for_online = "payload_normalization_error"
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        count_records = len(records)
        if len(records) > MAX_BATCH_SIZE:
            reason_code_for_online = "batch_too_large"
            raise HTTPException(
                status_code=400,
                detail={
                    "detail": "batch too large",
                    "max_batch_size": int(MAX_BATCH_SIZE),
                    "received": int(len(records)),
                },
            )

        context = deps.get_prediction_context()
        model_state = deps.get_model()
        model = model_state.get("model")

        expected_raw_cols = list(context.get("expected_raw_cols", []))
        metadata_loaded = bool(context.get("metadata_loaded", False))
        identity_context = dict(context.get("identity", {}))
        model_version_for_online = str(identity_context.get("model_version", "unknown"))
        model_family_for_online = str(identity_context.get("model_family", "unknown"))
        variant_for_online = str(identity_context.get("variant", "unknown"))
        threshold_for_online = float(context.get("threshold", 0.30))
        if not expected_raw_cols:
            reason_code_for_online = "metadata_unavailable"
            raise HTTPException(
                status_code=503,
                detail={
                    "detail": "metadata not available",
                    "model_loaded": bool(model_state.get("model_loaded", False)),
                    "metadata_loaded": metadata_loaded,
                    "notes": _dedupe_notes(
                        list(context.get("notes", [])) + list(model_state.get("notes", []))
                    ),
                },
            )

        if model is None:
            reason_code_for_online = "model_unavailable"
            raise HTTPException(
                status_code=503,
                detail={
                    "detail": "model not available",
                    "model_loaded": False,
                    "metadata_loaded": metadata_loaded,
                    "notes": list(model_state.get("notes", [])),
                },
            )

        df = build_raw_dataframe(records)
        try:
            X_raw, missing_stats = build_model_input_frame(
                df_payload=df,
                expected_raw_cols=expected_raw_cols,
                allow_partial=allow_partial_enabled,
            )
        except MissingColumnsError as exc:
            missing_stats = dict(exc.stats or {})
            reason_code_for_online = "missing_required_columns"
            raise HTTPException(
                status_code=400,
                detail={
                    "detail": "Missing required columns",
                    "missing_columns": list(exc.missing_columns),
                    "model_loaded": bool(model_state.get("model_loaded", False)),
                    "metadata_loaded": metadata_loaded,
                },
            ) from exc
        except ValueError as exc:
            if "leakage-like extra columns" in str(exc):
                reason_code_for_online = "leakage_like_extra_columns"
            else:
                reason_code_for_online = "payload_validation_error"
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not hasattr(model, "predict_proba"):
            reason_code_for_online = "model_predict_proba_missing"
            raise HTTPException(
                status_code=500,
                detail="model does not expose predict_proba; probability output is required",
            )

        try:
            proba_matrix = np.asarray(model.predict_proba(X_raw), dtype=float)
        except HTTPException:
            raise
        except Exception as exc:
            reason_code_for_online = "inference_failed"
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

        if proba_matrix.ndim != 2 or proba_matrix.shape[1] < 2:
            reason_code_for_online = "invalid_predict_proba_shape"
            raise HTTPException(
                status_code=500,
                detail="model predict_proba output must have shape (n, 2)",
            )
        risk_probas = proba_matrix[:, 1].tolist()
        risk_probas_for_online = list(risk_probas)

        threshold = float(context["threshold"])
        threshold_for_online = float(threshold)
        identity = dict(context.get("identity", {}))
        model_version_for_online = str(identity.get("model_version", "unknown"))
        model_family_for_online = str(identity.get("model_family", "unknown"))
        variant_for_online = str(identity.get("variant", "unknown"))
        notes: list[str] = []
        notes.extend(list(context.get("notes", [])))
        notes.extend(list(model_state.get("notes", [])))
        if missing_stats and (
            bool(missing_stats.get("allow_partial_enabled", False))
            or int(missing_stats.get("missing_cols_count", 0)) > 0
        ):
            notes.extend(_build_missing_stats_notes(missing_stats))
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

        status_code_for_log = 200
        reason_code_for_online = "predict_success"
        return PredictResponse(
            predictions=predictions,
            count=len(predictions),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException as exc:
        status_code_for_log = int(exc.status_code)
        if reason_code_for_online is None:
            if int(exc.status_code) == 400:
                reason_code_for_online = "http_400"
            elif int(exc.status_code) == 503:
                reason_code_for_online = "http_503"
            else:
                reason_code_for_online = f"http_{int(exc.status_code)}"
        raise
    finally:
        _log_predict_request_summary(
            status_code=status_code_for_log,
            count_records=count_records,
            allow_partial_enabled=allow_partial_enabled,
            missing_stats=missing_stats,
        )
        _append_predict_online_metrics_event(
            status_code=status_code_for_log,
            count_records=count_records,
            risk_probas=risk_probas_for_online,
            threshold=threshold_for_online,
            missing_stats=missing_stats,
            model_version=model_version_for_online,
            model_family=model_family_for_online,
            variant=variant_for_online,
            reason_code=reason_code_for_online,
        )


__all__ = ["router"]
