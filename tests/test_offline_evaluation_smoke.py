from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.offline_evaluation as offline_evaluation


class DummyModel:
    def predict_proba(self, X):
        scores = np.asarray([0.2, 0.7, 0.9], dtype=float)[: len(X)]
        return np.column_stack([1.0 - scores, scores])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_keys(payload):
    keys = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys |= _collect_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_keys(item)
    return keys


def test_offline_evaluation_smoke_generates_reports_without_pii_keys(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"fake-joblib-bytes")
    _write_json(
        model_dir / "metadata.json",
        {
            "model_version": "2026-02-22T20-00-00Z",
            "model_family": "nonlinear_hgb",
            "variant": "default",
            "expected_raw_cols": ["a", "b", "c"],
            "threshold_policy": {"operational_fixed_threshold": 0.30},
        },
    )

    fake_dataset = tmp_path / "dataset.xlsx"
    fake_dataset.write_bytes(b"fake-dataset")

    y_true = pd.Series([0, 1, 1], dtype="Int64")
    ids = pd.Series(["S1", "S2", "S3"], dtype="string")
    x_raw = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})

    monkeypatch.setattr(
        offline_evaluation,
        "_require_eval_dependencies",
        lambda: {"joblib": type("JoblibStub", (), {"load": staticmethod(lambda _: DummyModel())})()},
    )
    monkeypatch.setattr(
        offline_evaluation,
        "load_pede_workbook_with_metadata",
        lambda file_path: ({2023: pd.DataFrame(), 2024: pd.DataFrame()}, {}, {}),
    )
    monkeypatch.setattr(
        offline_evaluation,
        "make_temporal_pairs",
        lambda *args, **kwargs: (pd.DataFrame(), y_true, ids),
    )
    monkeypatch.setattr(
        offline_evaluation,
        "build_raw_from_ids",
        lambda **kwargs: x_raw,
    )
    monkeypatch.setattr(
        offline_evaluation,
        "persist_dataset_version_event",
        lambda **kwargs: tmp_path / "artifacts" / "dataset_versions" / "noop.json",
    )

    out_json = tmp_path / "artifacts" / "offline_metrics_2023_2024.json"
    out_md = tmp_path / "artifacts" / "offline_metrics_2023_2024.md"
    report = offline_evaluation.run_offline_evaluation(
        dataset_path=fake_dataset,
        model_dir=model_dir,
        year_t=2023,
        year_t1=2024,
        out_json=out_json,
        out_md=out_md,
        write_markdown=True,
    )

    assert report["status"] == "PASS"
    assert report["evaluation_kind"] == "offline_ground_truth_delay_replay"
    assert report["pair"] == {"year_t": 2023, "year_t1": 2024}
    assert report["model"]["model_version"] == "2026-02-22T20-00-00Z"
    assert report["threshold_operational"] == 0.30
    assert "metrics_at_operational_threshold" in report
    assert "confusion_matrix" in report["metrics_at_operational_threshold"]
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    keys_found = _collect_keys(payload)
    assert "ra" not in keys_found
    assert "ids" not in keys_found

