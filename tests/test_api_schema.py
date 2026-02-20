from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas import (
    PredictResponse,
    PredictionResult,
    build_predict_response,
    build_prediction_result,
    derive_risk_class,
)


def _valid_metadata() -> dict[str, object]:
    return {
        "model_version": "2026-02-20T12-00-00Z",
        "model_family": "nonlinear_hgb",
        "variant": "default",
        "threshold_policy": {
            "operational_fixed_threshold": 0.30,
        },
    }


def test_prediction_result_and_predict_response_serialization() -> None:
    metadata = _valid_metadata()

    single = build_prediction_result(risk_proba=0.82, metadata=metadata)
    assert single.risk_class == 1
    assert single.threshold_applied == pytest.approx(0.30)
    assert single.model_version == "2026-02-20T12-00-00Z"

    response = build_predict_response(
        risk_probas=[0.82, 0.12],
        metadata=metadata,
    )
    payload = response.dict()
    assert isinstance(payload["predictions"], list)
    assert payload["count"] == 2
    assert len(payload["predictions"]) == 2
    datetime.fromisoformat(payload["generated_at"])

    serialized = response.json()
    assert "risk_proba" in serialized
    assert "threshold_applied" in serialized
    assert "model_version" in serialized


def test_risk_class_is_derived_from_probability_and_threshold() -> None:
    assert derive_risk_class(risk_proba=0.29, threshold_applied=0.30) == 0
    assert derive_risk_class(risk_proba=0.30, threshold_applied=0.30) == 1
    assert derive_risk_class(risk_proba=0.91, threshold_applied=0.30) == 1

    with pytest.raises(ValidationError, match="risk_class must be derived"):
        PredictionResult(
            risk_proba=0.9,
            risk_class=0,
            threshold_applied=0.30,
            model_version="v1",
            model_family="baseline_logreg",
            variant="none",
            decision_policy="fixed_threshold",
        )

    with pytest.raises(ValueError, match="risk_proba must be within \\[0,1\\]"):
        build_prediction_result(risk_proba=1.1, metadata=_valid_metadata())


def test_defaults_are_applied_for_incomplete_metadata() -> None:
    result = build_prediction_result(
        risk_proba=0.40,
        metadata={},
    )
    assert result.threshold_applied == pytest.approx(0.30)
    assert result.model_version == "unknown"
    assert result.model_family == "unknown"
    assert result.variant == "unknown"
    assert result.risk_class == 1
    assert result.notes is not None
    assert "fallback_default_threshold" in result.notes
    assert "fallback_unknown_model_version" in result.notes
    assert "fallback_unknown_model_family" in result.notes
    assert "fallback_unknown_variant" in result.notes


def test_legacy_threshold_policy_path_is_supported() -> None:
    metadata = {
        "model_version": "legacy-v1",
        "model_family": "baseline_logreg",
        "variant": "none",
        "threshold_policy": {
            "operational": {
                "threshold": 0.55,
            }
        },
    }
    result = build_prediction_result(risk_proba=0.54, metadata=metadata)
    assert result.threshold_applied == pytest.approx(0.55)
    assert result.risk_class == 0
    assert result.notes is not None
    assert "threshold_from_legacy_policy_operational" in result.notes
    response = PredictResponse(
        predictions=[result],
        count=1,
        generated_at="2026-02-20T12:00:00+00:00",
    )
    assert response.count == 1
