from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.metadata_schema import validate_metadata
from src.promote_model import main as promote_main
from src.promote_model import run_model_promotion


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _collect_lower_keys(payload: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys |= _collect_lower_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_lower_keys(item)
    return keys


def _build_valid_source_metadata(variant: str = "none") -> dict[str, object]:
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
    eval_holdout_05 = {
        "threshold": 0.5,
        "metrics": {
            "recall": 0.58,
            "precision": 0.62,
            "f1": 0.60,
            "roc_auc": 0.69,
            "pr_auc": 0.64,
            "positive_rate": 0.38,
        },
        "confusion_matrix": {"tn": 9, "fp": 5, "fn": 6, "tp": 8},
    }
    eval_holdout_030 = {
        "threshold": 0.3,
        "metrics": {
            "recall": 0.72,
            "precision": 0.57,
            "f1": 0.64,
            "roc_auc": 0.69,
            "pr_auc": 0.64,
            "positive_rate": 0.54,
        },
        "confusion_matrix": {"tn": 7, "fp": 7, "fn": 4, "tp": 10},
    }
    eval_holdout_calibrated = {
        "threshold": 0.28,
        "metrics": {
            "recall": 0.75,
            "precision": 0.55,
            "f1": 0.63,
            "roc_auc": 0.69,
            "pr_auc": 0.64,
            "positive_rate": 0.58,
        },
        "confusion_matrix": {"tn": 6, "fp": 8, "fn": 3, "tp": 11},
    }
    return {
        "model_family": "baseline_logreg",
        "model_kind": "LogisticRegression",
        "variant": variant,
        "model_version": "2026-02-20T12-00-00Z",
        "trained_at": "2026-02-20T12:00:00+00:00",
        "promoted_at": None,
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
            "operational": {"mode": "fixed", "threshold": 0.30, "rule": "alert_if_proba>=0.30"},
            "capacity_fallback": {
                "mode": "topk",
                "topk_fraction": 0.20,
                "rule": "alert_top_20_percent_by_score",
            },
            "notes": ["Top-k is batch only."],
        },
        "evaluation_train": eval_train_05,
        "evaluation_holdout": eval_holdout_05,
        "evaluation_train_at_0.5": eval_train_05,
        "evaluation_train_at_0.30": eval_train_030,
        "evaluation_holdout_at_0.5": eval_holdout_05,
        "evaluation_holdout_at_0.30": eval_holdout_030,
        "evaluation_holdout_at_calibrated_threshold": eval_holdout_calibrated,
        "threshold_calibration": {
            "threshold_selected": 0.28,
            "recall_target": 0.90,
            "selection_rule": "max_precision_subject_to_recall>=0.90",
        },
        "versions": {
            "python": "3.11.10",
            "pandas": "2.2.2",
            "numpy": "1.26.4",
            "scikit_learn": None,
            "joblib": None,
            "sklearn": None,
        },
        "artifact_hashes": {
            "model_joblib_sha256": "0" * 64,
            "metadata_sha256": None,
        },
    }


def _build_basic_fixture(root: Path) -> tuple[Path, Path, Path]:
    selection_path = root / "artifacts" / "model_selection.json"
    models_root = root / "artifacts" / "models"
    variant_dir = models_root / "baseline_logreg" / "none"
    variant_dir.mkdir(parents=True, exist_ok=True)

    (variant_dir / "model.joblib").write_bytes(b"MODEL_V1")
    _write_json(variant_dir / "metadata.json", _build_valid_source_metadata("none"))
    _write_json(
        selection_path,
        {
            "status": "PASS",
            "winner": {
                "model_family": "baseline_logreg",
                "variant": "none",
            },
        },
    )
    out_dir = root / "app" / "model"
    return selection_path, models_root, out_dir


def _set_selection_status(selection_path: Path, status: str) -> None:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    payload["status"] = status
    _write_json(selection_path, payload)


def test_promote_model_happy_path(tmp_path: Path) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)
    promoted = run_model_promotion(
        selection_path=selection_path,
        models_root=models_root,
        out_dir=out_dir,
        force=False,
        backup=True,
    )

    dest_model = out_dir / "model.joblib"
    dest_meta = out_dir / "metadata.json"
    promoted_json = out_dir / "promoted_model.json"
    assert dest_model.exists()
    assert dest_meta.exists()
    assert promoted_json.exists()

    promoted_payload = json.loads(promoted_json.read_text(encoding="utf-8"))
    assert promoted_payload["winner"] == {"model_family": "baseline_logreg", "variant": "none"}
    assert promoted_payload["sha256"]["model"] == _sha256_bytes(b"MODEL_V1")
    assert isinstance(promoted_payload["sha256"]["metadata"], str)
    assert len(promoted_payload["sha256"]["metadata"]) == 64
    assert promoted["dest_paths"]["model"] == str(dest_model)
    assert promoted_payload["decision"]["decision"] == "ALLOW"
    assert promoted_payload["summary"]["threshold_used"] == pytest.approx(0.30)
    assert promoted_payload["summary"]["recall_holdout"] == pytest.approx(0.72)
    assert promoted_payload["summary"]["pr_auc_holdout"] == pytest.approx(0.64)

    promoted_metadata = json.loads(dest_meta.read_text(encoding="utf-8"))
    ok, errors = validate_metadata(promoted_metadata)
    assert ok, errors
    forbidden_keys = {"ids", "ra_list", "students", "records"}
    assert forbidden_keys.isdisjoint(_collect_lower_keys(promoted_metadata))


def test_destination_exists_without_force_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.joblib").write_bytes(b"OLD_MODEL")

    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--selection-path",
            str(selection_path),
            "--models-root",
            str(models_root),
            "--out-dir",
            str(out_dir),
            "--force",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        promote_main()
    assert exc.value.code == 1


def test_backup_enabled_creates_backup_snapshot(tmp_path: Path) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.joblib").write_bytes(b"OLD_MODEL")
    _write_json(out_dir / "metadata.json", {"old": True})

    run_model_promotion(
        selection_path=selection_path,
        models_root=models_root,
        out_dir=out_dir,
        force=True,
        backup=True,
    )

    backups_root = out_dir / "backups"
    backup_dirs = [path for path in backups_root.iterdir() if path.is_dir()]
    assert len(backup_dirs) == 1
    backup_dir = backup_dirs[0]
    assert (backup_dir / "model.joblib").read_bytes() == b"OLD_MODEL"
    assert json.loads((backup_dir / "metadata.json").read_text(encoding="utf-8")) == {"old": True}
    assert (out_dir / "model.joblib").read_bytes() == b"MODEL_V1"


def test_warning_selection_requires_allow_warning_flag(tmp_path: Path) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)
    _set_selection_status(selection_path, "WARNING")

    with pytest.raises(ValueError, match="allow-warning 1"):
        run_model_promotion(
            selection_path=selection_path,
            models_root=models_root,
            out_dir=out_dir,
            force=False,
            backup=True,
            allow_warning=False,
        )

    promoted = run_model_promotion(
        selection_path=selection_path,
        models_root=models_root,
        out_dir=out_dir,
        force=False,
        backup=True,
        allow_warning=True,
    )
    assert promoted["decision"]["decision"] == "ALLOW_WITH_OVERRIDE"
    assert promoted["decision"]["allow_warning_used"] is True
    promoted_payload = json.loads((out_dir / "promoted_model.json").read_text(encoding="utf-8"))
    assert promoted_payload["decision"]["decision"] == "ALLOW_WITH_OVERRIDE"


def test_stage_only_then_promote_from_staging_flow(tmp_path: Path) -> None:
    selection_path, models_root, prod_dir = _build_basic_fixture(tmp_path)
    stage_dir = tmp_path / "app" / "model" / "staging"

    staged = run_model_promotion(
        selection_path=selection_path,
        models_root=models_root,
        out_dir=stage_dir,
        force=False,
        backup=True,
        stage_only=True,
    )
    assert staged["mode"] == "stage_only"
    assert (stage_dir / "model.joblib").exists()
    assert (stage_dir / "metadata.json").exists()
    assert (stage_dir / "staging_manifest.json").exists()
    assert not (stage_dir / "promoted_model.json").exists()

    promoted = run_model_promotion(
        selection_path=selection_path,
        models_root=models_root,
        out_dir=prod_dir,
        force=False,
        backup=True,
        promote=True,
        from_staging=stage_dir,
    )
    assert promoted["mode"] == "promote_from_staging"
    assert (prod_dir / "model.joblib").exists()
    assert (prod_dir / "metadata.json").exists()
    assert (prod_dir / "promoted_model.json").exists()
    promoted_payload = json.loads((prod_dir / "promoted_model.json").read_text(encoding="utf-8"))
    assert promoted_payload["source_paths"]["model"].endswith("app/model/staging/model.joblib")
    assert promoted_payload["dest_paths"]["manifest"].endswith("app/model/promoted_model.json")


def test_invalid_selection_or_missing_source_fail_with_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)

    # Case 1: selection without winner.
    _write_json(selection_path, {"status": "PASS"})
    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--selection-path",
            str(selection_path),
            "--models-root",
            str(models_root),
            "--out-dir",
            str(out_dir),
        ],
    )
    with pytest.raises(SystemExit) as exc1:
        promote_main()
    assert exc1.value.code == 1

    # Case 2: source model.joblib missing.
    _write_json(
        selection_path,
        {
            "status": "PASS",
            "winner": {
                "model_family": "baseline_logreg",
                "variant": "none",
            },
        },
    )
    (models_root / "baseline_logreg" / "none" / "model.joblib").unlink()
    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--selection-path",
            str(selection_path),
            "--models-root",
            str(models_root),
            "--out-dir",
            str(out_dir),
        ],
    )
    with pytest.raises(SystemExit) as exc2:
        promote_main()
    assert exc2.value.code == 1


def test_promote_flag_requires_from_staging(tmp_path: Path) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires --from-staging"):
        run_model_promotion(
            selection_path=selection_path,
            models_root=models_root,
            out_dir=out_dir,
            promote=True,
            from_staging=None,
        )


def test_promote_model_legacy_metadata_without_dataset_sha_adds_null_and_note(
    tmp_path: Path,
) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)
    legacy_metadata_path = models_root / "baseline_logreg" / "none" / "metadata.json"
    legacy_payload = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
    legacy_payload["dataset"] = {
        "path_hint": "/private/tmp/secret/dataset.xlsx",
        "basename": "dataset.xlsx",
    }
    _write_json(legacy_metadata_path, legacy_payload)

    run_model_promotion(
        selection_path=selection_path,
        models_root=models_root,
        out_dir=out_dir,
        force=False,
        backup=True,
    )

    promoted_metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "dataset" in promoted_metadata
    assert promoted_metadata["dataset"]["sha256"] is None
    assert promoted_metadata["dataset"]["path_hint"] == "dataset.xlsx"
    assert any(
        "dataset.sha256 unavailable in legacy metadata" in str(note)
        for note in promoted_metadata.get("notes", [])
    )
