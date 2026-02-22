from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.online_metrics import append_online_event, build_score_histogram, summarize_online_batch


def test_build_score_histogram_counts_sum_to_total() -> None:
    hist = build_score_histogram(
        probas=[0.01, 0.08, 0.10, 0.55, 0.99],
        bins=[0.0, 0.1, 0.2, 0.6, 1.0],
    )
    assert hist["total"] == 5
    assert hist["n_bins"] == 4
    assert sum(hist["bin_counts"]) == 5


def test_summarize_online_batch_computes_positive_rate_and_status_family() -> None:
    event = summarize_online_batch(
        [0.1, 0.4, 0.9, 0.8],
        0.5,
        {"missing_cols_rate": 0.25, "missing_values_rate": 0.40, "missing_cols_count": 2},
        200,
        "v-test",
        model_family="nonlinear_hgb",
        variant="default",
        reason_code="predict_success",
    )
    assert event["status_family"] == "2xx"
    assert event["n_records"] == 4
    assert event["n_positive_at_threshold"] == 2
    assert event["positive_rate_at_threshold"] == pytest.approx(0.5)
    assert event["score_histogram"] is not None
    assert event["score_histogram"]["total"] == 4
    assert event["reason_code"] == "predict_success"
    assert event["missing_cols_rate"] == pytest.approx(0.25)


def test_summarize_online_batch_for_error_without_scores_omits_histogram() -> None:
    event = summarize_online_batch(
        None,
        0.3,
        None,
        503,
        "unknown",
        n_records=12,
        reason_code="model_unavailable",
    )
    assert event["status_family"] == "5xx"
    assert event["n_records"] == 12
    assert event["score_histogram"] is None
    assert event["positive_rate_at_threshold"] is None
    assert event["reason_code"] == "model_unavailable"


def test_append_online_event_writes_one_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "online_metrics.jsonl"
    event = summarize_online_batch(
        [0.2, 0.8],
        0.5,
        {"missing_cols_rate": 0.1, "missing_values_rate": 0.2},
        200,
        "v-test",
        n_records=2,
        reason_code="predict_success",
    )
    append_online_event(event, path=str(path))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "online_inference_aggregate"
    assert payload["n_records"] == 2
    assert "score_histogram" in payload

