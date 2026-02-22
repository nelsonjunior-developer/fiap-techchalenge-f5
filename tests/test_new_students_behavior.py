from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException

from app.routes import predict


class DummyModel:
    def predict_proba(self, X):
        import numpy as np

        return np.asarray([[0.1, 0.9]] * len(X), dtype=float)


def _ctx(expected_raw_cols: list[str] | None = None) -> dict[str, object]:
    cols = expected_raw_cols if expected_raw_cols is not None else ["a", "b", "c"]
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


def _call_predict(payload):
    try:
        response = predict(payload=payload)
    except HTTPException as exc:
        return int(exc.status_code), {"detail": exc.detail}
    return 200, response.dict()


def _patch_ready_model(monkeypatch: pytest.MonkeyPatch, expected_cols: list[str]) -> None:
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx(expected_cols))
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": DummyModel(), "model_loaded": True, "notes": ["model_loaded_ok"]},
    )


def test_new_student_payload_missing_columns_keeps_400_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("ALLOW_PARTIAL_PAYLOAD", raising=False)
    _patch_ready_model(monkeypatch, ["a", "b", "c"])

    with caplog.at_level(logging.INFO):
        status, body = _call_predict({"a": 1})

    assert status == 400
    assert body["detail"]["detail"] == "Missing required columns"
    assert body["detail"]["missing_columns"] == ["b", "c"]
    assert "predict_request_summary" in caplog.text
    assert "status_code=400" in caplog.text
    assert "allow_partial_enabled=False" in caplog.text
    assert "missing_cols_count=2" in caplog.text
    assert "RA=" not in caplog.text


def test_new_student_payload_partial_is_allowed_with_flag_and_logs_missing_stats(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ALLOW_PARTIAL_PAYLOAD", "true")
    online_metrics_path = tmp_path / "logs" / "online_metrics.jsonl"
    monkeypatch.setenv("ONLINE_METRICS_PATH", str(online_metrics_path))
    _patch_ready_model(monkeypatch, ["a", "b", "c"])

    with caplog.at_level(logging.INFO):
        status, body = _call_predict({"a": 987654})

    assert status == 200
    assert body["count"] == 1
    notes = body["predictions"][0]["notes"] or []
    assert any(str(note).startswith("missing_cols_rate=") for note in notes)
    assert any(str(note).startswith("missing_values_rate=") for note in notes)
    assert any(str(note) == "allow_partial_payload=1" for note in notes)
    assert any(str(note).startswith("top_missing_cols=") for note in notes)

    assert "predict_request_summary" in caplog.text
    assert "status_code=200" in caplog.text
    assert "allow_partial_enabled=True" in caplog.text
    assert "allow_partial_used=True" in caplog.text
    assert "missing_cols_count=2" in caplog.text
    assert "987654" not in caplog.text
    assert "RA=" not in caplog.text

    lines = online_metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["status_code"] == 200
    assert event["status_family"] == "2xx"
    assert event["n_records"] == 1
    assert event["score_histogram"] is not None
    assert event["positive_rate_at_threshold"] == pytest.approx(1.0)
    assert event["missing_cols_rate"] is not None


def test_new_student_payload_leakage_like_extra_is_blocked_even_with_partial_flag(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ALLOW_PARTIAL_PAYLOAD", "1")
    _patch_ready_model(monkeypatch, ["a", "b"])

    with caplog.at_level(logging.INFO):
        status, body = _call_predict({"a": 1, "target": 1})

    assert status == 400
    assert "leakage-like extra columns" in str(body["detail"])
    assert "predict_request_summary" in caplog.text
    assert "status_code=400" in caplog.text
    assert "allow_partial_enabled=True" in caplog.text
    assert "target" not in caplog.text
