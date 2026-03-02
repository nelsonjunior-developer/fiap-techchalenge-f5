from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import src.retrain_orchestrator as retrain


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_decision_required_by_time_trigger() -> None:
    now = datetime(2026, 3, 2, tzinfo=timezone.utc)
    old_dt = (now - timedelta(days=120)).isoformat()
    policy = {
        "time_trigger": {"max_age_days": 90},
        "drift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
        "shift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
    }
    state = {
        "metadata": {"trained_at": old_dt, "model_version": "v1"},
        "drift_summary": {},
        "shift_report": {},
        "reference_meta": {},
    }

    decision = retrain.evaluate_retrain_decision(policy=policy, state=state, now=now)
    assert decision["decision"] == "RETRAIN_REQUIRED"
    assert any("time_trigger" in reason for reason in decision["required_reasons"])
    assert decision["inputs"]["model_age_days"] == 120


def test_decision_required_on_drift_fail() -> None:
    now = datetime(2026, 3, 2, tzinfo=timezone.utc)
    policy = {
        "time_trigger": {"max_age_days": 90},
        "drift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
        "shift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
    }
    state = {
        "metadata": {"trained_at": now.isoformat()},
        "drift_summary": {"status": "FAIL"},
        "shift_report": {},
        "reference_meta": {},
    }
    decision = retrain.evaluate_retrain_decision(policy=policy, state=state, now=now)
    assert decision["decision"] == "RETRAIN_REQUIRED"
    assert "drift_status=FAIL" in decision["required_reasons"]


def test_decision_recommended_on_warning_status() -> None:
    now = datetime(2026, 3, 2, tzinfo=timezone.utc)
    policy = {
        "time_trigger": {"max_age_days": 90},
        "drift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
        "shift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
    }
    state = {
        "metadata": {"trained_at": now.isoformat()},
        "drift_summary": {"status": "WARNING"},
        "shift_report": {"status": "WARNING"},
        "reference_meta": {},
    }
    decision = retrain.evaluate_retrain_decision(policy=policy, state=state, now=now)
    assert decision["decision"] == "RETRAIN_RECOMMENDED"
    assert "drift_status=WARNING" in decision["recommended_reasons"]
    assert "shift_status=WARNING" in decision["recommended_reasons"]


def test_decision_noop_when_reports_missing() -> None:
    now = datetime(2026, 3, 2, tzinfo=timezone.utc)
    policy = {
        "time_trigger": {"max_age_days": 90},
        "drift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
        "shift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
    }
    state = {
        "metadata": {"model_version": "v1"},
        "drift_summary": {},
        "shift_report": {},
        "reference_meta": {},
    }
    decision = retrain.evaluate_retrain_decision(policy=policy, state=state, now=now)
    assert decision["decision"] == "NOOP"
    assert "trained_at_missing_time_trigger_not_evaluated" in decision["notes"]


def test_orchestrator_dry_run_writes_decision_without_dataset(tmp_path: Path) -> None:
    policy_path = tmp_path / "docs" / "retrain_policy.json"
    _write_json(
        policy_path,
        {
            "version": "1.0.0",
            "time_trigger": {"max_age_days": 90},
            "drift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
            "shift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
        },
    )
    metadata_path = tmp_path / "app" / "model" / "metadata.json"
    _write_json(
        metadata_path,
        {"model_version": "v1", "trained_at": "2025-01-01T00:00:00+00:00"},
    )
    shift_path = tmp_path / "artifacts" / "temporal_shift_report.json"
    _write_json(shift_path, {"status": "FAIL"})

    decision_out = tmp_path / "artifacts" / "retrain_decision.json"
    run_out = tmp_path / "artifacts" / "retrain_run.json"
    payload = retrain.run_retrain_orchestration(
        dataset_path=None,
        policy_path=policy_path,
        metadata_path=metadata_path,
        reference_meta_path=tmp_path / "app" / "model" / "reference" / "reference_meta.json",
        drift_summary_path=tmp_path / "artifacts" / "drift_report_summary.json",
        shift_report_path=shift_path,
        decision_out=decision_out,
        run_out=run_out,
        execute=False,
    )
    assert payload["status"] == "DRY_RUN"
    assert decision_out.exists()
    saved = json.loads(decision_out.read_text(encoding="utf-8"))
    assert saved["decision"] == "RETRAIN_REQUIRED"
    assert "shift_status=FAIL" in saved["required_reasons"]
    assert run_out.exists() is False


def test_orchestrator_execute_requires_dataset_path(tmp_path: Path) -> None:
    policy_path = tmp_path / "docs" / "retrain_policy.json"
    _write_json(
        policy_path,
        {
            "version": "1.0.0",
            "time_trigger": {"max_age_days": 90},
            "drift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
            "shift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
        },
    )
    metadata_path = tmp_path / "app" / "model" / "metadata.json"
    _write_json(
        metadata_path,
        {"model_version": "v1", "trained_at": "2025-01-01T00:00:00+00:00"},
    )
    shift_path = tmp_path / "artifacts" / "temporal_shift_report.json"
    _write_json(shift_path, {"status": "FAIL"})

    try:
        retrain.run_retrain_orchestration(
            dataset_path=None,
            policy_path=policy_path,
            metadata_path=metadata_path,
            reference_meta_path=tmp_path / "app" / "model" / "reference" / "reference_meta.json",
            drift_summary_path=tmp_path / "artifacts" / "drift_report_summary.json",
            shift_report_path=shift_path,
            decision_out=tmp_path / "artifacts" / "retrain_decision.json",
            run_out=tmp_path / "artifacts" / "retrain_run.json",
            execute=True,
        )
    except FileNotFoundError as exc:
        assert "Dataset path is required when --execute=1" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected FileNotFoundError when execute=1 without dataset path")
