from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
from fastapi import HTTPException
try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - environment dependent
    TestClient = None  # type: ignore[assignment]

from app.main import app
from app.routes import predict


class DummyModel:
    def predict_proba(self, X):
        import numpy as np

        return np.asarray([[0.2, 0.8]] * len(X), dtype=float)


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


def _post_json(payload):
    if TestClient is None:
        try:
            response_payload = predict(payload=payload)
        except HTTPException as exc:
            return exc.status_code, {"detail": exc.detail}
        return 200, response_payload.dict()
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    return response.status_code, response.json()


def test_predict_returns_503_when_service_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx([]))
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": None, "model_loaded": False, "notes": ["model_file_missing"]},
    )
    status, body = _post_json({"a": 1, "b": 2})
    assert status == 503
    payload = body["detail"]
    assert payload["metadata_loaded"] is False
    assert payload["model_loaded"] is False
    assert "notes" in payload


def test_predict_invalid_structural_payload_returns_400() -> None:
    status, body = _post_json({"records": "x"})
    if TestClient is None:
        assert status == 400
        assert "payload must be a dict" in str(body["detail"])
        return
    assert status == 422


def test_predict_missing_required_columns_returns_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx(["a", "b"]))
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": DummyModel(), "model_loaded": True, "notes": ["model_loaded_ok"]},
    )
    status, body = _post_json({"a": 1})
    assert status == 400
    detail = body["detail"]
    assert detail["detail"] == "Missing required columns"
    assert detail["missing_columns"] == ["b"]


def test_predict_allows_non_suspicious_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx(["a", "b"]))
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": DummyModel(), "model_loaded": True, "notes": ["model_loaded_ok"]},
    )
    status, payload = _post_json({"a": 1, "b": 2, "extra": 123})
    assert status == 200
    assert payload["count"] == 1
    assert payload["predictions"][0]["risk_proba"] == pytest.approx(0.8)
    assert payload["predictions"][0]["risk_class"] == 1


def test_predict_blocks_leakage_like_extra_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx(["a", "b"]))
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": DummyModel(), "model_loaded": True, "notes": ["model_loaded_ok"]},
    )
    status, body = _post_json({"a": 1, "b": 2, "target": 1})
    assert status == 400
    assert "leakage-like extra columns" in str(body["detail"])


def test_predict_supports_batch_and_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx(["a", "b"]))
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": DummyModel(), "model_loaded": True, "notes": ["model_loaded_ok"]},
    )
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

    status_batch, body_batch = _post_json(rows)
    status_envelope, body_envelope = _post_json({"records": rows})

    assert status_batch == 200
    assert body_batch["count"] == 2
    assert status_envelope == 200
    assert body_envelope["count"] == 2


def test_predict_batch_too_large_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx(["a", "b"]))
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": DummyModel(), "model_loaded": True, "notes": ["model_loaded_ok"]},
    )
    rows = [{"a": 1, "b": 2}] * 501
    status, body = _post_json(rows)
    assert status == 400
    detail = body["detail"]
    assert detail["detail"] == "batch too large"
    assert detail["max_batch_size"] == 500
