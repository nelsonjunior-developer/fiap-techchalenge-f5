from __future__ import annotations

from collections.abc import Mapping
import re

import pytest

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]

from app.main import app


def _parse_metric_samples(body: str, metric_name: str) -> list[tuple[dict[str, str], float]]:
    samples: list[tuple[dict[str, str], float]] = []
    line_re = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+([-+eE0-9.]+)$")
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = line_re.match(line)
        if match is None:
            continue
        name = match.group(1)
        if name != metric_name:
            continue
        labels: dict[str, str] = {}
        labels_blob = match.group(3)
        if labels_blob:
            for chunk in labels_blob.split(","):
                key, _, value = chunk.partition("=")
                key_norm = key.strip()
                value_norm = value.strip().strip('"')
                if key_norm:
                    labels[key_norm] = value_norm
        try:
            metric_value = float(match.group(4))
        except ValueError:
            continue
        samples.append((labels, metric_value))
    return samples


def _sample_value(
    body: str,
    metric_name: str,
    *,
    labels: Mapping[str, str] | None = None,
) -> float:
    labels = dict(labels or {})
    for sample_labels, value in _parse_metric_samples(body, metric_name):
        if all(sample_labels.get(key) == expected for key, expected in labels.items()):
            return float(value)
    return 0.0


def test_metrics_endpoint_exposes_prometheus_metrics() -> None:
    if TestClient is None:
        pytest.skip("TestClient unavailable in this environment")

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/version").status_code == 200

        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "").lower()

    body = response.text
    assert "http_requests_total" in body
    assert "http_request_duration_seconds_bucket" in body
    assert "inference_records_total" in body
    assert "inference_positive_total" in body
    assert "model_loaded" in body
    assert "metadata_loaded" in body
    assert 'path="/health"' in body
    assert re.search(r'model_loaded\s+[01](?:\.0+)?', body) is not None
    assert re.search(r'metadata_loaded\s+[01](?:\.0+)?', body) is not None
    assert "RA" not in body
    assert "Nome_Anon" not in body
    assert "Avaliador1" not in body


def test_metrics_counter_increments_relative_for_health() -> None:
    if TestClient is None:
        pytest.skip("TestClient unavailable in this environment")

    with TestClient(app) as client:
        before_text = client.get("/metrics").text
        before_health = _sample_value(
            before_text,
            "http_requests_total",
            labels={"method": "GET", "path": "/health", "status": "200"},
        )
        assert client.get("/health").status_code == 200
        after_text = client.get("/metrics").text
        after_health = _sample_value(
            after_text,
            "http_requests_total",
            labels={"method": "GET", "path": "/health", "status": "200"},
        )

    assert after_health >= before_health + 1.0


def test_predict_metrics_increment_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    if TestClient is None:
        pytest.skip("TestClient unavailable in this environment")

    class _DummyModel:
        def predict_proba(self, X):
            assert len(X) == 2
            return [[0.1, 0.9], [0.8, 0.2]]

    monkeypatch.setattr(
        "app.routes.deps.get_prediction_context",
        lambda: {
            "expected_raw_cols": ["a", "b"],
            "threshold": 0.30,
            "identity": {
                "model_version": "metrics-test",
                "model_family": "dummy",
                "variant": "test",
            },
            "metadata_loaded": True,
            "metadata_contract_valid": True,
            "metadata": {},
            "notes": [],
        },
    )
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {
            "model": _DummyModel(),
            "model_loaded": True,
            "model_exists": True,
            "model_joblib_exists": True,
            "model_path_basename": "model.joblib",
            "notes": [],
        },
    )
    monkeypatch.setattr(
        "app.routes.deps.get_model_loader_status",
        lambda: {
            "model_loaded": True,
            "metadata_loaded": True,
            "model_joblib_exists": True,
            "notes": [],
        },
    )

    with TestClient(app) as client:
        before_text = client.get("/metrics").text
        before_records = _sample_value(
            before_text,
            "inference_records_total",
            labels={"endpoint": "/predict"},
        )
        before_positive = _sample_value(
            before_text,
            "inference_positive_total",
            labels={"endpoint": "/predict", "threshold": "0.30"},
        )

        response = client.post(
            "/predict",
            json={"records": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
        )
        assert response.status_code == 200

        after_text = client.get("/metrics").text
        after_records = _sample_value(
            after_text,
            "inference_records_total",
            labels={"endpoint": "/predict"},
        )
        after_positive = _sample_value(
            after_text,
            "inference_positive_total",
            labels={"endpoint": "/predict", "threshold": "0.30"},
        )

    assert after_records >= before_records + 2.0
    assert after_positive >= before_positive + 1.0


def test_metrics_route_is_not_in_openapi() -> None:
    if TestClient is None:
        pytest.skip("TestClient unavailable in this environment")

    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()

    assert "/metrics" not in openapi.get("paths", {})
