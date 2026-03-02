from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.retrain_orchestrator as retrain


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_policy() -> dict:
    return {
        "version": "1.0.0",
        "time_trigger": {"max_age_days": 90},
        "drift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
        "shift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
    }


def test_safe_read_json_and_policy_merge_branches(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert retrain._safe_read_json(missing) is None

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert retrain._safe_read_json(broken) is None

    non_dict = tmp_path / "non_dict.json"
    non_dict.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert retrain._safe_read_json(non_dict) is None

    default_policy, default_notes = retrain._merge_policy(missing)
    assert default_policy["time_trigger"]["max_age_days"] == 90
    assert "policy_file_missing_or_invalid_using_defaults" in default_notes

    custom_policy_path = tmp_path / "policy.json"
    _write_json(
        custom_policy_path,
        {
            "version": "2.0.0",
            "notes": ["local test"],
            "time_trigger": "bad",
            "drift_trigger": ["bad"],
            "shift_trigger": {"warn_status": "WARNX"},
        },
    )
    merged_policy, merge_notes = retrain._merge_policy(custom_policy_path)
    assert merged_policy["version"] == "2.0.0"
    assert merged_policy["notes"] == ["local test"]
    assert merged_policy["shift_trigger"]["warn_status"] == "WARNX"
    assert "policy_time_trigger_invalid_using_defaults" in merge_notes
    assert "policy_drift_trigger_invalid_using_defaults" in merge_notes


def test_parse_and_conversion_helpers_cover_branches() -> None:
    dt_with_z = retrain._parse_iso_datetime("2026-03-02T01:02:03Z")
    assert dt_with_z is not None
    assert dt_with_z.tzinfo == timezone.utc

    dt_without_tz = retrain._parse_iso_datetime("2026-03-02T01:02:03")
    assert dt_without_tz is not None
    assert dt_without_tz.tzinfo == timezone.utc

    assert retrain._parse_iso_datetime("not-a-date") is None
    assert retrain._parse_iso_datetime("") is None
    assert retrain._to_int("11", default=0) == 11
    assert retrain._to_int("x", default=7) == 7
    assert retrain._to_bool_flag(True) is True
    assert retrain._to_bool_flag(False) is False
    assert retrain._to_bool_flag(1) is True
    assert retrain._to_bool_flag(0) is False


def test_safe_rel_dataset_basename_and_dataset_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert retrain._dataset_basename(None) is None
    assert retrain._dataset_basename("a/b/file.xlsx") == "file.xlsx"

    dataset = tmp_path / "dataset.xlsx"
    dataset.write_text("placeholder", encoding="utf-8")
    assert retrain._resolve_dataset_for_execution(dataset) == dataset

    with pytest.raises(FileNotFoundError, match="Dataset path is required"):
        retrain._resolve_dataset_for_execution(None)
    with pytest.raises(FileNotFoundError, match="Dataset path not found"):
        retrain._resolve_dataset_for_execution(tmp_path / "missing.xlsx")

    def _raise_relpath(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(retrain.os.path, "relpath", _raise_relpath)
    assert retrain._safe_rel(Path("/tmp/abc.txt")) == "/tmp/abc.txt"


def test_sanitize_command_replaces_dataset_and_abs_tokens(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset" / "input.xlsx"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("x", encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "src.train_baseline",
        "--dataset-path",
        str(dataset),
        "/absolute/path/secret/model.joblib",
        "relative_token",
    ]

    sanitized = retrain._sanitize_command(command, dataset_path=dataset)
    assert str(dataset) not in sanitized
    assert dataset.name in sanitized
    assert "model.joblib" in sanitized
    assert "/absolute/path/secret/model.joblib" not in sanitized
    assert retrain._sanitize_command(command, dataset_path=None) == command


def test_run_step_writes_log_and_sanitizes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset.xlsx"
    dataset.write_text("x", encoding="utf-8")
    logs_dir = tmp_path / "logs"

    def fake_run(command, capture_output, text):  # type: ignore[no-untyped-def]
        assert capture_output is True
        assert text is True
        assert isinstance(command, list)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(retrain.subprocess, "run", fake_run)
    step = retrain._run_step(
        name="train_baseline",
        command=[sys.executable, "--dataset", str(dataset), "/tmp/hidden/model.joblib"],
        logs_dir=logs_dir,
        dataset_path=dataset,
    )
    assert step["returncode"] == 0
    assert dataset.name in step["command"]
    assert "/tmp/hidden/model.joblib" not in step["command"]
    log_file = logs_dir / "train_baseline.log"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "[stdout]" in log_text
    assert "[returncode]" in log_text


def test_build_retrain_steps_handles_current_csv_variants(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.xlsx"
    dataset.write_text("x", encoding="utf-8")
    current_csv = tmp_path / "current_model_frame.csv"
    current_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    kwargs = {
        "python_bin": sys.executable,
        "dataset_path": dataset,
        "run_models_root": tmp_path / "artifacts" / "models" / "runs" / "run1",
        "selection_path": tmp_path / "artifacts" / "model_selection.json",
        "selection_md": tmp_path / "artifacts" / "model_selection.md",
        "stage_dir": tmp_path / "app" / "model" / "staging",
        "prod_dir": tmp_path / "app" / "model",
        "reference_dir": tmp_path / "app" / "model" / "reference",
        "allow_warning_promotion": True,
        "drift_out_html": tmp_path / "artifacts" / "drift_report.html",
        "drift_out_json": tmp_path / "artifacts" / "drift_report_summary.json",
    }

    steps_with_current, notes_with_current = retrain._build_retrain_steps(
        current_csv=current_csv, **kwargs
    )
    assert "drift_report_refresh" in [name for name, _ in steps_with_current]
    assert notes_with_current == []

    steps_without_current, notes_without_current = retrain._build_retrain_steps(
        current_csv=None, **kwargs
    )
    assert "drift_report_refresh" not in [name for name, _ in steps_without_current]
    assert "current_csv_not_provided_drift_refresh_skipped" in notes_without_current

    missing_current = tmp_path / "missing_current.csv"
    _, notes_missing_current = retrain._build_retrain_steps(
        current_csv=missing_current, **kwargs
    )
    assert (
        f"current_csv_not_found_drift_refresh_skipped={missing_current.name}"
        in notes_missing_current
    )


def test_execute_noop_and_recommended_skip_without_dataset(tmp_path: Path) -> None:
    policy_path = tmp_path / "docs" / "retrain_policy.json"
    _write_json(policy_path, _base_policy())
    metadata_path = tmp_path / "app" / "model" / "metadata.json"
    now = datetime.now(timezone.utc).isoformat()
    _write_json(metadata_path, {"model_version": "v1", "trained_at": now})

    noop_run_out = tmp_path / "artifacts" / "retrain_run_noop.json"
    noop_payload = retrain.run_retrain_orchestration(
        dataset_path=None,
        policy_path=policy_path,
        metadata_path=metadata_path,
        reference_meta_path=tmp_path / "app" / "model" / "reference" / "reference_meta.json",
        drift_summary_path=tmp_path / "artifacts" / "missing_drift.json",
        shift_report_path=tmp_path / "artifacts" / "missing_shift.json",
        run_out=noop_run_out,
        decision_out=tmp_path / "artifacts" / "decision_noop.json",
        execute=True,
    )
    assert noop_payload["status"] == "NOOP"
    assert "no_trigger_fired_execution_skipped" in noop_payload["notes"]
    assert noop_run_out.exists()

    _write_json(tmp_path / "artifacts" / "drift_report_summary.json", {"status": "WARNING"})
    skip_run_out = tmp_path / "artifacts" / "retrain_run_skip.json"
    skip_payload = retrain.run_retrain_orchestration(
        dataset_path=None,
        policy_path=policy_path,
        metadata_path=metadata_path,
        reference_meta_path=tmp_path / "app" / "model" / "reference" / "reference_meta.json",
        drift_summary_path=tmp_path / "artifacts" / "drift_report_summary.json",
        shift_report_path=tmp_path / "artifacts" / "missing_shift.json",
        run_out=skip_run_out,
        decision_out=tmp_path / "artifacts" / "decision_skip.json",
        execute=True,
        allow_recommended=False,
    )
    assert skip_payload["status"] == "SKIPPED_RECOMMENDED"
    assert any(
        "recommended_trigger_without_override_execution_skipped" in note
        for note in skip_payload["notes"]
    )
    assert skip_run_out.exists()


def test_execute_failure_writes_manifest_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path = tmp_path / "docs" / "retrain_policy.json"
    _write_json(policy_path, _base_policy())
    metadata_path = tmp_path / "app" / "model" / "metadata.json"
    _write_json(
        metadata_path,
        {"model_version": "v1", "trained_at": "2025-01-01T00:00:00+00:00"},
    )
    shift_path = tmp_path / "artifacts" / "temporal_shift_report.json"
    _write_json(shift_path, {"status": "FAIL"})
    dataset = tmp_path / "dataset.xlsx"
    dataset.write_text("x", encoding="utf-8")
    run_out = tmp_path / "artifacts" / "retrain_run_fail.json"

    monkeypatch.setattr(
        retrain,
        "_build_retrain_steps",
        lambda **kwargs: ([("train_baseline", ["python", "-V"])], ["step_note"]),  # type: ignore[arg-type]
    )

    def _fake_run_step(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "name": kwargs["name"],
            "command": kwargs["command"],
            "returncode": 1,
            "started_at": "2026-03-02T00:00:00+00:00",
            "finished_at": "2026-03-02T00:00:01+00:00",
            "duration_seconds": 1.0,
            "log_path": "artifacts/retrain_logs/fail.log",
        }

    monkeypatch.setattr(retrain, "_run_step", _fake_run_step)

    with pytest.raises(RuntimeError, match="failed at step 'train_baseline'"):
        retrain.run_retrain_orchestration(
            dataset_path=dataset,
            policy_path=policy_path,
            metadata_path=metadata_path,
            reference_meta_path=tmp_path / "app" / "model" / "reference" / "reference_meta.json",
            drift_summary_path=tmp_path / "artifacts" / "drift_report_summary.json",
            shift_report_path=shift_path,
            decision_out=tmp_path / "artifacts" / "decision_fail.json",
            run_out=run_out,
            execute=True,
        )

    saved = json.loads(run_out.read_text(encoding="utf-8"))
    assert saved["status"] == "FAIL"
    assert saved["failed_step"] == "train_baseline"
    assert saved["steps"][0]["returncode"] == 1


def test_execute_success_records_post_state_and_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path = tmp_path / "docs" / "retrain_policy.json"
    _write_json(policy_path, _base_policy())
    metadata_path = tmp_path / "app" / "model" / "metadata.json"
    _write_json(
        metadata_path,
        {"model_version": "v1", "trained_at": "2025-01-01T00:00:00+00:00"},
    )
    shift_path = tmp_path / "artifacts" / "temporal_shift_report.json"
    _write_json(shift_path, {"status": "FAIL"})
    dataset = tmp_path / "dataset.xlsx"
    dataset.write_text("x", encoding="utf-8")
    prod_dir = tmp_path / "app" / "model_new"

    monkeypatch.setattr(
        retrain,
        "_build_retrain_steps",
        lambda **kwargs: ([("train_baseline", ["python", "-V"])], ["step_note"]),  # type: ignore[arg-type]
    )

    def _fake_run_step(**kwargs):  # type: ignore[no-untyped-def]
        metadata_out = prod_dir / "metadata.json"
        _write_json(metadata_out, {"model_version": "v2"})
        return {
            "name": kwargs["name"],
            "command": kwargs["command"],
            "returncode": 0,
            "started_at": "2026-03-02T00:00:00+00:00",
            "finished_at": "2026-03-02T00:00:01+00:00",
            "duration_seconds": 1.0,
            "log_path": "artifacts/retrain_logs/success.log",
        }

    monkeypatch.setattr(retrain, "_run_step", _fake_run_step)
    payload = retrain.run_retrain_orchestration(
        dataset_path=dataset,
        policy_path=policy_path,
        metadata_path=metadata_path,
        reference_meta_path=tmp_path / "app" / "model" / "reference" / "reference_meta.json",
        drift_summary_path=tmp_path / "artifacts" / "drift_report_summary.json",
        shift_report_path=shift_path,
        decision_out=tmp_path / "artifacts" / "decision_success.json",
        run_out=tmp_path / "artifacts" / "retrain_run_success.json",
        execute=True,
        isolate_run=False,
        prod_dir=prod_dir,
    )
    assert payload["status"] == "PASS"
    assert payload["post_state"]["model_version"] == "v2"
    assert "step_note" in payload["notes"]
    assert "model_version_updated_after_retrain" in payload["notes"]
    assert payload["inputs"]["dataset_basename"] == "dataset.xlsx"
    assert len(payload["steps"]) == 1


def test_parse_args_and_main_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retrain_orchestrator.py",
            "--dataset-path",
            "dataset.xlsx",
            "--policy",
            "docs/retrain_policy.json",
            "--execute",
            "1",
            "--allow-recommended",
            "1",
            "--allow-warning-promotion",
            "0",
            "--isolate-run",
            "0",
            "--current-csv",
            "artifacts/current.csv",
            "--drift-out-html",
            "artifacts/custom_drift.html",
            "--drift-out-json",
            "artifacts/custom_drift.json",
        ],
    )
    args = retrain._parse_args()
    assert args.dataset_path == "dataset.xlsx"
    assert args.execute == 1
    assert args.allow_recommended == 1
    assert args.allow_warning_promotion == 0
    assert args.isolate_run == 0
    assert args.current_csv == "artifacts/current.csv"

    monkeypatch.setattr(retrain, "setup_logging", lambda: None)
    monkeypatch.setattr(
        retrain, "run_retrain_orchestration", lambda **kwargs: {"status": "DRY_RUN"}
    )
    monkeypatch.setattr(sys, "argv", ["retrain_orchestrator.py"])
    assert retrain.main() == 0

    monkeypatch.setattr(
        retrain, "run_retrain_orchestration", lambda **kwargs: {"status": "FAIL"}
    )
    monkeypatch.setattr(sys, "argv", ["retrain_orchestrator.py"])
    assert retrain.main() == 1

    calls: dict[str, bool] = {}

    def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(retrain, "run_retrain_orchestration", _raise)
    monkeypatch.setattr(retrain._logger, "error", lambda *a, **k: calls.setdefault("error", True))
    monkeypatch.setattr(sys, "argv", ["retrain_orchestrator.py"])
    assert retrain.main() == 1
    assert calls.get("error") is True
