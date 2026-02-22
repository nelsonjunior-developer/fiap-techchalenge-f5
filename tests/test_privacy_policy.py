from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.online_metrics import summarize_online_batch
from src.privacy import is_safe_json_payload, redact_dict, safe_log_extra
from src.utils import get_logger, log_event, setup_logging

pytest.importorskip("fastapi")
try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - environment dependent
    TestClient = None  # type: ignore[assignment]

from fastapi import HTTPException

from app.main import app
from app.routes import predict
from src.cohort_stats import persist_skip_dataset_report as persist_skip_cohort_report
from src.validate import _persist_skip_dataset_report as persist_skip_validate_report


class DummyModel:
    def predict_proba(self, X):
        import numpy as np

        return np.asarray([[0.1, 0.9]] * len(X), dtype=float)


def _ctx(expected_raw_cols: list[str] | None = None) -> dict[str, object]:
    cols = expected_raw_cols if expected_raw_cols is not None else ["a", "b"]
    return {
        "expected_raw_cols": cols,
        "threshold": 0.30,
        "identity": {
            "model_version": "v-test",
            "model_family": "nonlinear_hgb",
            "variant": "default",
        },
        "metadata_loaded": bool(cols),
        "metadata": {},
        "notes": ["threshold_from_metadata"],
    }


def _patch_ready_model(monkeypatch: pytest.MonkeyPatch, expected_cols: list[str]) -> None:
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx(expected_cols))
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": DummyModel(), "model_loaded": True, "notes": ["model_loaded_ok"]},
    )


def _call_predict(payload):
    try:
        response = predict(payload=payload)
    except HTTPException as exc:
        return int(exc.status_code), {"detail": exc.detail}
    return 200, response.dict()


def test_log_event_redacts_ra_and_nome_anon(caplog: pytest.LogCaptureFixture) -> None:
    setup_logging(level="INFO", log_to_file=False, log_format="json")
    logger = get_logger("tests.privacy.log")

    with caplog.at_level(logging.INFO):
        log_event(logger, "privacy_event", RA="123", Nome_Anon="abc", ok=1)

    record = next(r for r in caplog.records if getattr(r, "event", None) == "privacy_event")
    context = getattr(record, "context", {}) or {}
    assert "RA" not in context
    assert "Nome_Anon" not in context
    assert context["ok"] == 1
    assert sorted(context.get("redacted_keys", [])) == ["Nome_Anon", "RA"]


def test_predict_with_ra_in_payload_does_not_echo_ra_value_in_response_or_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_ready_model(monkeypatch, ["a", "b"])
    pii_marker = "RA-SECRET-999"

    with caplog.at_level(logging.INFO):
        status, body = _call_predict({"a": 1, "b": 2, "RA": pii_marker, "target": 1})

    assert status == 400
    detail_text = str(body.get("detail"))
    assert "leakage-like extra columns" in detail_text
    assert pii_marker not in detail_text
    assert "RA" not in detail_text  # response should not echo sensitive field names
    assert pii_marker not in caplog.text
    assert "RA=" not in caplog.text


def test_predict_422_handler_does_not_echo_invalid_input_marker_in_response_or_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    if TestClient is None:
        pytest.skip("TestClient unavailable in this environment")

    marker = "PII-MARKER-XYZ"
    monkeypatch.setenv("ALLOW_PARTIAL_PAYLOAD", "1")
    monkeypatch.setenv("ONLINE_METRICS_PATH", str(tmp_path / "logs" / "online_metrics.jsonl"))

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            response = client.post("/predict", json={"records": marker})

    assert response.status_code == 422
    assert marker not in response.text
    payload = response.json()
    assert payload["detail"] == "Invalid request payload"
    assert "error_count" in payload
    assert marker not in caplog.text
    assert "request_validation_summary" in caplog.text


def test_online_metrics_event_is_privacy_safe_and_blocks_prohibited_shapes() -> None:
    event = summarize_online_batch(
        [0.2, 0.8],
        0.5,
        {"missing_cols_rate": 0.1, "missing_values_rate": 0.2},
        200,
        "v-test",
        n_records=2,
        reason_code="predict_success",
    )
    assert is_safe_json_payload(event) is True

    unsafe_payload = {
        "status_code": 200,
        "records": [{"RA": "123"}, {"RA": "456"}],
    }
    assert is_safe_json_payload(unsafe_payload) is False


def test_validate_and_cohort_skip_reports_are_privacy_safe(tmp_path: Path) -> None:
    validate_report = persist_skip_validate_report(output_dir=tmp_path / "validate", write_markdown=False)
    cohort_report = persist_skip_cohort_report(output_dir=tmp_path / "cohort", write_markdown=False)

    assert is_safe_json_payload(validate_report) is True
    assert is_safe_json_payload(cohort_report) is True

    validate_json = json.loads((tmp_path / "validate" / "data_quality_report.json").read_text(encoding="utf-8"))
    cohort_json = json.loads((tmp_path / "cohort" / "ra_intersections.json").read_text(encoding="utf-8"))
    assert is_safe_json_payload(validate_json) is True
    assert is_safe_json_payload(cohort_json) is True


def test_redact_dict_and_safe_log_extra_guard_large_lists_and_records() -> None:
    payload = {
        "records": [{"a": 1}, {"a": 2}],
        "hist": list(range(10)),
        "big_values": list(range(30)),
        "ok": True,
    }
    redacted = redact_dict(payload)
    assert redacted["records"] == "[REDACTED_LIST]"
    assert redacted["hist"] == list(range(10))
    assert "big_values" in redacted and str(redacted["big_values"]).startswith("[REDACTED_LIST_LEN:")
    safe = safe_log_extra(payload)
    assert isinstance(safe, dict)
    assert safe["ok"] is True
