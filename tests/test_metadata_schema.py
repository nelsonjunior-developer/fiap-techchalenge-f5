from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.metadata_schema import validate_metadata
from src.metadata_schema import main as metadata_schema_main


def _valid_metadata() -> dict[str, Any]:
    eval_train_05 = {
        "threshold": 0.5,
        "metrics": {
            "recall": 0.70,
            "precision": 0.65,
            "f1": 0.67,
            "roc_auc": 0.74,
            "pr_auc": 0.71,
            "positive_rate": 0.40,
        },
        "confusion_matrix": {"tn": 10, "fp": 4, "fn": 3, "tp": 8},
    }
    eval_train_030 = {
        "threshold": 0.30,
        "metrics": {
            "recall": 0.85,
            "precision": 0.60,
            "f1": 0.70,
            "roc_auc": 0.74,
            "pr_auc": 0.71,
            "positive_rate": 0.55,
        },
        "confusion_matrix": {"tn": 8, "fp": 6, "fn": 2, "tp": 9},
    }
    return {
        "model_family": "baseline_logreg",
        "variant": "none",
        "model_version": "2026-02-20T12-00-00Z",
        "trained_at": "2026-02-20T12:00:00+00:00",
        "promoted_at": "2026-02-20T13:00:00+00:00",
        "random_state": 42,
        "train_pair": {
            "year_t": 2022,
            "year_t1": 2023,
            "n": 25,
            "n_pos": 11,
            "prevalence": 0.44,
        },
        "holdout_pair": {
            "year_t": 2023,
            "year_t1": 2024,
            "n": 28,
            "n_pos": 14,
            "prevalence": 0.50,
        },
        "dataset": {
            "path_hint": "dataset/PEDE_PASSOS_DATASET_FIAP.xlsx",
            "basename": "PEDE_PASSOS_DATASET_FIAP.xlsx",
            "sha256": None,
        },
        "expected_raw_cols": ["Idade", "INDE", "Mat", "Por", "Ing"],
        "expected_model_cols": ["Idade", "INDE", "Mat", "Por", "Ing"],
        "excluded_cols": ["Nome_Anon", "Avaliador1"],
        "feature_engineering": {
            "enabled": False,
            "enable_age_bucket": False,
            "engineered_cols": [],
        },
        "feature_pruning": {
            "plan_hash": "abc123",
            "kept_model_cols_count": 5,
            "dropped_summary": {"dropped_all_missing_cols_count": 0},
        },
        "threshold_policy": {
            "operational_fixed_threshold": 0.30,
            "recall_target_for_calibration": 0.90,
            "calibrated_threshold": 0.28,
            "topk_fallback_fraction": 0.20,
            "notes": ["Top-k is batch only."],
        },
        "evaluation_train_at_0.5": eval_train_05,
        "evaluation_train_at_0.30": eval_train_030,
        "evaluation_holdout_at_0.5": eval_train_05,
        "evaluation_holdout_at_0.30": eval_train_030,
        "evaluation_holdout_at_calibrated_threshold": eval_train_030,
        "versions": {
            "python": "3.11.10",
            "pandas": "2.2.2",
            "numpy": "1.26.4",
            "scikit_learn": None,
            "joblib": None,
        },
        "artifact_hashes": {
            "model_joblib_sha256": "0" * 64,
            "metadata_sha256": None,
        },
    }


def test_validate_metadata_minimum_valid_payload_passes() -> None:
    payload = _valid_metadata()
    ok, errors = validate_metadata(payload)
    assert ok
    assert errors == []


def test_validate_metadata_missing_expected_raw_cols_fails() -> None:
    payload = _valid_metadata()
    payload.pop("expected_raw_cols", None)
    ok, errors = validate_metadata(payload)
    assert not ok
    assert any("expected_raw_cols" in error for error in errors)


def test_validate_metadata_empty_payload_collects_required_errors() -> None:
    ok, errors = validate_metadata({})
    assert not ok
    assert any("missing required key: model_family" in error for error in errors)
    assert any("invalid type for key 'train_pair'" in error for error in errors)
    assert any("invalid type for key 'threshold_policy'" in error for error in errors)


def test_validate_metadata_invalid_nested_types_reports_errors() -> None:
    payload = _valid_metadata()
    payload["model_family"] = "invalid_family"
    payload["promoted_at"] = 123
    payload["random_state"] = "42"
    payload["train_pair"] = {"year_t": "2022", "year_t1": 2023, "n": "10", "n_pos": 3, "prevalence": "0.3"}
    payload["dataset"] = {"path_hint": 1, "basename": 2, "sha256": 3}
    payload["feature_engineering"] = {"enabled": "yes", "enable_age_bucket": "no", "engineered_cols": [1]}
    payload["feature_pruning"] = {"plan_hash": 1, "kept_model_cols_count": "5", "dropped_summary": []}
    payload["threshold_policy"] = {
        "operational_fixed_threshold": "0.3",
        "recall_target_for_calibration": "0.9",
        "calibrated_threshold": "0.25",
        "topk_fallback_fraction": "0.2",
        "notes": [1],
    }
    payload["evaluation_train_at_0.5"] = {"threshold": "0.5", "metrics": {"recall": "0.8"}, "confusion_matrix": {"tn": "1"}}
    payload["versions"] = {"python": 3.11, "pandas": 2, "numpy": 1, "scikit_learn": 1, "joblib": 1}
    payload["artifact_hashes"] = {"model_joblib_sha256": 123, "metadata_sha256": 456}

    ok, errors = validate_metadata(payload)
    assert not ok
    assert any("model_family" in error for error in errors)
    assert any("promoted_at" in error for error in errors)
    assert any("train_pair.year_t" in error for error in errors)
    assert any("dataset.path_hint" in error for error in errors)
    assert any("feature_engineering.enabled" in error for error in errors)
    assert any("feature_pruning.plan_hash" in error for error in errors)
    assert any("threshold_policy.operational_fixed_threshold" in error for error in errors)
    assert any("evaluation_train_at_0.5.threshold" in error for error in errors)
    assert any("versions.python" in error for error in errors)
    assert any("artifact_hashes.model_joblib_sha256" in error for error in errors)


def test_metadata_schema_main_missing_path_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_path = Path("does-not-exist-metadata.json")
    monkeypatch.setattr("sys.argv", ["python", "--path", str(missing_path)])
    with pytest.raises(SystemExit) as exc:
        metadata_schema_main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "metadata path not found" in captured.err


def test_metadata_schema_main_invalid_json_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_json = tmp_path / "metadata.json"
    invalid_json.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["python", "--path", str(invalid_json)])
    with pytest.raises(SystemExit) as exc:
        metadata_schema_main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "invalid metadata json" in captured.err


def test_metadata_schema_main_valid_file_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    valid_json = tmp_path / "metadata.json"
    valid_json.write_text(json.dumps(_valid_metadata()), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["python", "--path", str(valid_json)])
    metadata_schema_main()
    captured = capsys.readouterr()
    assert "metadata schema valid" in captured.out


def test_metadata_schema_main_invalid_schema_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_schema = tmp_path / "metadata.json"
    invalid_schema.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["python", "--path", str(invalid_schema)])
    with pytest.raises(SystemExit) as exc:
        metadata_schema_main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "missing required key: model_family" in captured.err
