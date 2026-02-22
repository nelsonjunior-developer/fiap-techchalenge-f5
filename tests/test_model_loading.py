from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
joblib = pytest.importorskip("joblib")
try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - environment dependent
    TestClient = None  # type: ignore[assignment]

import app.deps as deps
from app.main import app
from app.routes import predict, version
from fastapi import HTTPException


class DummyModel:
    def predict_proba(self, X):
        import numpy as np

        return np.asarray([[0.3, 0.7]] * len(X), dtype=float)


def _configure_paths(tmp_path: Path) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    deps.MODEL_DIR = model_dir
    deps.MODEL_PATH = model_dir / "model.joblib"
    deps.METADATA_PATH = model_dir / "metadata.json"
    deps.get_serving_metadata.cache_clear()
    deps.get_prediction_context.cache_clear()
    deps.invalidate_model_cache()


def _write_minimal_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model_version": "v-test",
                "model_family": "baseline_logreg",
                "variant": "none",
                "threshold_policy": {"operational_fixed_threshold": 0.30},
                "expected_raw_cols": ["a", "b"],
            }
        ),
        encoding="utf-8",
    )


def _post_predict(payload):
    if TestClient is None:
        try:
            resp = predict(payload=payload)
        except HTTPException as exc:
            return exc.status_code, {"detail": exc.detail}
        return 200, resp.dict()
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    return response.status_code, response.json()


def _get_version():
    if TestClient is None:
        return 200, version()
    with TestClient(app) as client:
        response = client.get("/version")
    return response.status_code, response.json()


def test_model_loader_missing_file_reports_false_and_predict_503(tmp_path: Path) -> None:
    _configure_paths(tmp_path)
    _write_minimal_metadata(deps.METADATA_PATH)
    deps.get_serving_metadata.cache_clear()
    deps.get_prediction_context.cache_clear()
    deps.invalidate_model_cache()

    state = deps.get_model()
    assert state["model_loaded"] is False
    assert state["model_joblib_exists"] is False
    assert "model_file_missing" in state["notes"]

    status_code, payload = _get_version()
    assert status_code == 200
    assert payload["model_loaded"] is False
    assert payload["model_joblib_exists"] is False
    assert "model_file_missing" in payload["model_notes"]

    predict_status, predict_payload = _post_predict({"a": 1, "b": 2})
    assert predict_status == 503
    assert predict_payload["detail"]["detail"] == "model not available"
    assert predict_payload["detail"]["model_loaded"] is False
    assert "model_file_missing" in predict_payload["detail"]["notes"]


def test_model_loader_success_uses_cache_and_predict_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path)
    _write_minimal_metadata(deps.METADATA_PATH)
    deps.MODEL_PATH.write_bytes(b"dummy-joblib-bytes")
    deps.get_serving_metadata.cache_clear()
    deps.get_prediction_context.cache_clear()
    deps.invalidate_model_cache()

    calls = {"n": 0}

    def _fake_load(path):
        calls["n"] += 1
        return DummyModel()

    monkeypatch.setattr(joblib, "load", _fake_load)

    state_first = deps.get_model()
    state_second = deps.get_model()
    assert state_first["model_loaded"] is True
    assert state_second["model_loaded"] is True
    assert calls["n"] == 1
    assert "model_loaded_ok" in state_first["notes"]

    status_code, payload = _get_version()
    assert status_code == 200
    assert payload["model_loaded"] is True
    assert payload["model_joblib_exists"] is True

    predict_status, predict_payload = _post_predict({"a": 1, "b": 2, "extra": 99})
    assert predict_status == 200
    assert predict_payload["count"] == 1
    assert predict_payload["predictions"][0]["risk_proba"] == pytest.approx(0.7)


def test_model_loader_failure_reports_exception_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path)
    deps.MODEL_PATH.write_bytes(b"dummy-joblib-bytes")
    deps.invalidate_model_cache()

    def _boom(path):
        raise RuntimeError("broken artifact")

    monkeypatch.setattr(joblib, "load", _boom)

    state = deps.get_model()
    assert state["model_loaded"] is False
    assert state["model_joblib_exists"] is True
    assert any(str(note).startswith("model_load_failed:RuntimeError") for note in state["notes"])
