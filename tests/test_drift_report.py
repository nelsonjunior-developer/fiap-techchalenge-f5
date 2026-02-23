from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import drift as drift_module
from src.drift import (
    _assert_no_sensitive_columns,
    _extract_evidently_summary,
    _load_reference_assets,
    _parse_args,
    _report_as_dict,
    _sample_frame,
    _save_evidently_html,
    _status_from_drift_share,
    run_drift_report,
)
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


def test_load_reference_assets_uses_csv_header_and_emits_mismatch_note(tmp_path: Path) -> None:
    reference_dir = _make_reference_dir(tmp_path, columns=["f1", "f2"])
    meta_path = reference_dir / "reference_meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["expected_model_cols"] = ["f1", "fX"]
    _write_json(meta_path, payload)

    ref_df, meta, paths, notes = _load_reference_assets(reference_dir)
    assert list(ref_df.columns) == ["f1", "f2"]
    assert meta["model_version"] == "2026-02-22T00-00-00Z__deadbeef"
    assert paths["reference_csv"].name == "reference_model_frame.csv"
    assert "reference_meta_expected_model_cols_mismatch_csv_header_using_csv_header" in notes


def test_assert_no_sensitive_columns_blocks_ra() -> None:
    with pytest.raises(ValueError, match="sensitive columns"):
        _assert_no_sensitive_columns(["feat_ok", "RA"], context="current_model_frame")


def test_sample_frame_and_status_helpers_cover_branches() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

    same_df, meta_same = _sample_frame(df, max_rows=10, seed=42)
    assert len(same_df) == 5
    assert meta_same["sampled"] is False

    sampled_df, meta_sampled = _sample_frame(df, max_rows=2, seed=42)
    assert len(sampled_df) == 2
    assert meta_sampled["sampled"] is True

    with pytest.raises(ValueError, match="max_rows must be > 0"):
        _sample_frame(df, max_rows=0, seed=42)

    assert _status_from_drift_share(
        share_drifted_features=0.05, warn_share_threshold=0.10, fail_share_threshold=0.30
    )[0] == "PASS"
    assert _status_from_drift_share(
        share_drifted_features=0.10, warn_share_threshold=0.10, fail_share_threshold=0.30
    )[0] == "WARNING"
    assert _status_from_drift_share(
        share_drifted_features=0.30, warn_share_threshold=0.10, fail_share_threshold=0.30
    )[0] == "FAIL"
    assert _status_from_drift_share(
        share_drifted_features=None, warn_share_threshold=0.10, fail_share_threshold=0.30
    )[0] == "FAIL"
    with pytest.raises(ValueError, match="fail_share_threshold must be >="):
        _status_from_drift_share(
            share_drifted_features=0.1, warn_share_threshold=0.30, fail_share_threshold=0.10
        )


def test_extract_evidently_summary_handles_missing_and_drift_by_columns_variants() -> None:
    no_candidate = _extract_evidently_summary({"foo": {"bar": 1}})
    assert no_candidate["drifted_features_count"] is None
    assert "evidently_summary_fields_not_found" in no_candidate["notes"]

    report_dict = {
        "result": {
            "drift_by_columns": {
                "a": {"drift_detected": True},
                "b": {"drift_detected": False},
                "c": {"drifted": True},
            }
        }
    }
    parsed = _extract_evidently_summary(report_dict)
    assert parsed["drifted_features_count"] == 2
    assert parsed["share_drifted_features"] == pytest.approx(2 / 3)

    missing_values = _extract_evidently_summary({"x": {"dataset_drift": True}})
    assert missing_values["dataset_drift"] is True
    assert "share_drifted_features_unavailable" in missing_values["notes"]
    assert "drifted_features_count_unavailable" in missing_values["notes"]


def test_evidently_export_helpers_support_multiple_api_shapes(tmp_path: Path) -> None:
    out_html = tmp_path / "out" / "report.html"

    class SaveHtmlReport:
        def save_html(self, path: str) -> None:
            Path(path).write_text("<html>ok</html>", encoding="utf-8")

        def as_dict(self) -> dict:
            return {"ok": True}

    _save_evidently_html(SaveHtmlReport(), out_html)
    assert out_html.exists()
    assert "<html>" in out_html.read_text(encoding="utf-8")
    assert _report_as_dict(SaveHtmlReport()) == {"ok": True}

    class GetHtmlReport:
        def get_html(self) -> str:
            return "<html>fallback</html>"

        def dict(self) -> dict:
            return {"fallback": True}

    out_html_2 = tmp_path / "out2" / "report.html"
    _save_evidently_html(GetHtmlReport(), out_html_2)
    assert "fallback" in out_html_2.read_text(encoding="utf-8")
    assert _report_as_dict(GetHtmlReport()) == {"fallback": True}

    class BadReport:
        def get_html(self) -> str:
            return ""

    with pytest.raises(RuntimeError, match="Unable to export Evidently HTML report"):
        _save_evidently_html(BadReport(), tmp_path / "bad" / "report.html")
    with pytest.raises(RuntimeError, match="Unable to extract Evidently report as dict"):
        _report_as_dict(BadReport())


def test_parse_args_and_main_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drift.py",
            "--current-csv",
            "artifacts/current.csv",
            "--reference-dir",
            "app/model/reference",
            "--no-json",
            "--max-rows",
            "123",
            "--seed",
            "7",
            "--warn-share",
            "0.11",
            "--fail-share",
            "0.33",
        ],
    )
    args = _parse_args()
    assert args.current_csv == "artifacts/current.csv"
    assert args.no_json is True
    assert args.max_rows == 123
    assert args.seed == 7
    assert args.warn_share == 0.11
    assert args.fail_share == 0.33

    calls: dict[str, object] = {}

    def fake_setup_logging() -> None:
        calls["setup_logging"] = True

    def fake_run_drift_report(**kwargs):
        calls["kwargs"] = kwargs
        return {"status": "PASS"}

    monkeypatch.setattr(drift_module, "setup_logging", fake_setup_logging)
    monkeypatch.setattr(drift_module, "run_drift_report", fake_run_drift_report)
    monkeypatch.setattr(drift_module._logger, "info", lambda *a, **k: calls.setdefault("info", True))
    drift_module.main()

    assert calls.get("setup_logging") is True
    assert isinstance(calls.get("kwargs"), dict)
    assert calls["kwargs"]["write_json"] is False
    assert calls["kwargs"]["max_rows"] == 123

    # Error path: user-facing handled exception should exit(1)
    monkeypatch.setattr(sys, "argv", ["drift.py", "--current-csv", "missing.csv"])
    monkeypatch.setattr(drift_module, "setup_logging", lambda: None)
    monkeypatch.setattr(
        drift_module,
        "run_drift_report",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(drift_module._logger, "error", lambda *a, **k: None)
    with pytest.raises(SystemExit) as excinfo:
        drift_module.main()
    assert excinfo.value.code == 1
