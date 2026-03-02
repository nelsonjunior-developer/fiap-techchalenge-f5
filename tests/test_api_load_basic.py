from __future__ import annotations

from time import perf_counter

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - environment dependent
    TestClient = None  # type: ignore[assignment]

from app.main import app


class DummyModel:
    def predict_proba(self, X):
        import numpy as np

        # Keep the model deterministic and fast for load-oriented tests.
        return np.asarray([[0.25, 0.75]] * len(X), dtype=float)


def _ctx() -> dict[str, object]:
    return {
        "expected_raw_cols": ["a", "b"],
        "threshold": 0.30,
        "identity": {
            "model_version": "v-load-test",
            "model_family": "nonlinear_hgb",
            "variant": "default",
        },
        "metadata_loaded": True,
        "metadata": {},
        "notes": ["threshold_from_metadata"],
    }


def _percentile_ms(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = int((len(sorted_values) - 1) * float(percentile))
    return float(sorted_values[rank] * 1000.0)


def _run_predict_load(
    *,
    client: TestClient,
    n_requests: int,
    batch_size: int,
) -> dict[str, float]:
    latencies_s: list[float] = []
    start = perf_counter()
    for req_idx in range(n_requests):
        records = [{"a": float(req_idx), "b": float(req_idx + 1)} for _ in range(batch_size)]
        t0 = perf_counter()
        response = client.post("/predict", json={"records": records})
        latencies_s.append(perf_counter() - t0)
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == batch_size
    elapsed = perf_counter() - start
    total_records = n_requests * batch_size
    return {
        "elapsed_s": float(elapsed),
        "requests_per_s": float(n_requests / elapsed) if elapsed > 0 else 0.0,
        "records_per_s": float(total_records / elapsed) if elapsed > 0 else 0.0,
        "p95_ms": _percentile_ms(latencies_s, 0.95),
        "p99_ms": _percentile_ms(latencies_s, 0.99),
    }


def test_predict_load_small_requests_sequential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    if TestClient is None:
        pytest.skip("TestClient unavailable in this environment")

    monkeypatch.setenv("ONLINE_METRICS_PATH", str(tmp_path / "logs" / "online_metrics.jsonl"))
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx())
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": DummyModel(), "model_loaded": True, "notes": ["model_loaded_ok"]},
    )

    with TestClient(app) as client:
        summary = _run_predict_load(client=client, n_requests=80, batch_size=1)

    # Generous thresholds to avoid flaky CI while still catching severe regressions.
    assert summary["p95_ms"] < 600.0
    assert summary["requests_per_s"] > 10.0


def test_predict_load_batch_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    if TestClient is None:
        pytest.skip("TestClient unavailable in this environment")

    monkeypatch.setenv("ONLINE_METRICS_PATH", str(tmp_path / "logs" / "online_metrics.jsonl"))
    monkeypatch.setattr("app.routes.deps.get_prediction_context", lambda: _ctx())
    monkeypatch.setattr(
        "app.routes.deps.get_model",
        lambda: {"model": DummyModel(), "model_loaded": True, "notes": ["model_loaded_ok"]},
    )

    # Boundary load on max batch size (500) across multiple requests.
    with TestClient(app) as client:
        summary = _run_predict_load(client=client, n_requests=8, batch_size=500)

    assert summary["p95_ms"] < 2500.0
    assert summary["records_per_s"] > 100.0
