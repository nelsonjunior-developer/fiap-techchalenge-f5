from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - environment dependent
    TestClient = None  # type: ignore[assignment]

import app.deps as deps
from app.main import app
from app.routes import health, version


def _configure_paths(tmp_path: Path) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    deps.MODEL_DIR = model_dir
    deps.MODEL_PATH = model_dir / "model.joblib"
    deps.METADATA_PATH = model_dir / "metadata.json"
    deps.get_serving_metadata.cache_clear()
    deps.get_prediction_context.cache_clear()
    deps.get_model.cache_clear()
    deps.get_model_loader_status.cache_clear()


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    _configure_paths(tmp_path)
    if TestClient is None:
        assert health() == {"status": "ok"}
        return
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint_fallback_without_metadata(tmp_path: Path) -> None:
    _configure_paths(tmp_path)
    if TestClient is None:
        payload = version()
    else:
        with TestClient(app) as client:
            response = client.get("/version")
        assert response.status_code == 200
        payload = response.json()
    assert payload["threshold_operational"] == pytest.approx(0.30)
    assert payload["metadata_loaded"] is False
    assert payload["model_version"] == "unknown"
    assert payload["model_family"] == "unknown"
    assert payload["variant"] == "unknown"


def test_version_endpoint_reads_metadata_when_available(tmp_path: Path) -> None:
    _configure_paths(tmp_path)
    metadata_payload = {
        "model_version": "2026-02-20T12-00-00Z",
        "model_family": "nonlinear_hgb",
        "variant": "default",
        "threshold_policy": {"operational_fixed_threshold": 0.42},
    }
    deps.METADATA_PATH.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    deps.get_serving_metadata.cache_clear()
    deps.get_prediction_context.cache_clear()
    deps.get_model.cache_clear()
    deps.get_model_loader_status.cache_clear()

    if TestClient is None:
        payload = version()
    else:
        with TestClient(app) as client:
            response = client.get("/version")
        assert response.status_code == 200
        payload = response.json()
    assert payload["metadata_loaded"] is True
    assert payload["model_version"] == "2026-02-20T12-00-00Z"
    assert payload["model_family"] == "nonlinear_hgb"
    assert payload["variant"] == "default"
    assert payload["threshold_operational"] == pytest.approx(0.42)


def test_global_request_validation_422_is_logged_without_payload_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    if TestClient is None:
        pytest.skip("TestClient unavailable in this environment")

    marker = "PII-MARKER-123"
    monkeypatch.setenv("ALLOW_PARTIAL_PAYLOAD", "1")
    online_metrics_path = tmp_path / "logs" / "online_metrics.jsonl"
    monkeypatch.setenv("ONLINE_METRICS_PATH", str(online_metrics_path))

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            response = client.post("/predict", json={"records": marker})

    assert response.status_code == 422
    assert "request_validation_summary" in caplog.text
    assert "status_code=422" in caplog.text
    assert "path=/predict" in caplog.text
    assert "predict_route=True" in caplog.text
    assert "allow_partial_enabled=True" in caplog.text
    assert marker not in caplog.text
    lines = online_metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["status_code"] == 422
    assert event["status_family"] == "4xx"
    assert event["reason_code"] == "request_validation_422"
    assert event["score_histogram"] is None
