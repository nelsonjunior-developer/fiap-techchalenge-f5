from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.drift import run_drift_report
from src.privacy import is_safe_json_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_reference_dir(tmp_path: Path, *, columns: list[str]) -> Path:
    reference_dir = tmp_path / "app" / "model" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({col: [0, 1, 2] for col in columns}).to_csv(
        reference_dir / "reference_model_frame.csv", index=False
    )
    _write_json(
        reference_dir / "reference_meta.json",
        {
            "model_version": "2026-02-22T00-00-00Z__deadbeef",
            "winner": {"model_family": "baseline_logreg", "variant": "v1"},
        },
    )
    return reference_dir


def test_drift_report_fails_when_reference_is_missing(tmp_path: Path) -> None:
    current_csv = tmp_path / "current.csv"
    pd.DataFrame({"f1": [1.0], "f2": [0.0]}).to_csv(current_csv, index=False)

    with pytest.raises(FileNotFoundError, match="Reference data not found"):
        run_drift_report(
            reference_dir=tmp_path / "missing_reference",
            current_csv=current_csv,
            out_html=tmp_path / "artifacts" / "drift_report.html",
            out_json=tmp_path / "artifacts" / "drift_report_summary.json",
        )


def test_drift_report_fails_on_missing_reference_columns(tmp_path: Path) -> None:
    reference_dir = _make_reference_dir(tmp_path, columns=["f1", "f2", "f3"])
    current_csv = tmp_path / "current.csv"
    pd.DataFrame({"f1": [1.0], "f2": [0.0]}).to_csv(current_csv, index=False)

    with pytest.raises(ValueError, match="missing required reference columns"):
        run_drift_report(
            reference_dir=reference_dir,
            current_csv=current_csv,
            out_html=tmp_path / "artifacts" / "drift_report.html",
            out_json=tmp_path / "artifacts" / "drift_report_summary.json",
        )


def test_drift_report_generates_html_and_json_with_evidently(tmp_path: Path) -> None:
    pytest.importorskip("evidently")

    reference_dir = tmp_path / "app" / "model" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    # MODEL frame only (no PII): numeric + categorical features
    reference_df = pd.DataFrame(
        {
            "feat_num": [0.0, 0.1, 0.2, 0.15, 0.05, 0.12],
            "feat_cat": ["A", "A", "B", "A", "B", "A"],
        }
    )
    current_df = pd.DataFrame(
        {
            "feat_num": [10.0, 9.5, 11.2, 8.7, 10.5, 9.9],
            "feat_cat": ["C", "C", "C", "A", "C", "B"],
            "extra_noise_col": [1, 2, 3, 4, 5, 6],  # should be ignored
        }
    )
    reference_df.to_csv(reference_dir / "reference_model_frame.csv", index=False)
    current_csv = tmp_path / "current_model_frame.csv"
    current_df.to_csv(current_csv, index=False)

    _write_json(
        reference_dir / "reference_meta.json",
        {
            "model_version": "2026-02-22T00-00-00Z__cafebabe",
            "winner": {"model_family": "baseline_logreg", "variant": "winner_v1"},
        },
    )

    out_html = tmp_path / "artifacts" / "drift_report.html"
    out_json = tmp_path / "artifacts" / "drift_report_summary.json"
    summary = run_drift_report(
        reference_dir=reference_dir,
        current_csv=current_csv,
        out_html=out_html,
        out_json=out_json,
        max_rows=100,
        seed=42,
    )

    assert out_html.exists()
    assert out_html.stat().st_size > 0
    assert out_json.exists()
    saved_summary = json.loads(out_json.read_text(encoding="utf-8"))

    assert summary["status"] in {"PASS", "WARNING", "FAIL"}
    assert saved_summary["status"] in {"PASS", "WARNING", "FAIL"}
    assert "drifted_features_count" in saved_summary["drift"]
    assert "share_drifted_features" in saved_summary["drift"]
    assert saved_summary["contract"]["extra_cols_dropped_count"] == 1
    assert saved_summary["contract"]["n_features"] == 2
    assert is_safe_json_payload(saved_summary) is True

