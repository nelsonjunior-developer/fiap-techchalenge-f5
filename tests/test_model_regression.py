from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.regression_check import check_model_regression
from src.regression_thresholds import MIN_PRAUC_HOLDOUT, MIN_RECALL_HOLDOUT_AT_030


def _holdout_block(*, recall: float, pr_auc: float, positive_rate: float = 0.5) -> dict:
    return {
        "threshold": 0.3,
        "metrics": {
            "recall": float(recall),
            "pr_auc": float(pr_auc),
            "positive_rate": float(positive_rate),
            "precision": 0.4,
            "f1": 0.5,
            "roc_auc": 0.7,
        },
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_selection(
    *,
    root: Path,
    model_family: str,
    variant: str,
    metadata_path: Path | None,
    status: str = "PASS",
) -> Path:
    winner = {
        "model_family": model_family,
        "variant": variant,
    }
    if metadata_path is not None:
        winner["path_metadata"] = str(metadata_path)
    payload = {
        "status": status,
        "winner": winner,
        "selection_criteria": {
            "threshold_used": 0.3,
            "min_recall_holdout": float(MIN_RECALL_HOLDOUT_AT_030),
            "min_pr_auc_holdout": float(MIN_PRAUC_HOLDOUT),
        },
    }
    return _write_json(root / "artifacts" / "model_selection.json", payload)


def test_check_model_regression_pass_with_holdout_at_030(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    metadata_path = models_root / "nonlinear_hgb" / "default" / "metadata.json"
    _write_json(
        metadata_path,
        {
            "model_family": "nonlinear_hgb",
            "variant": "default",
            "model_version": "v-pass",
            "evaluation_holdout_at_0.30": _holdout_block(
                recall=float(MIN_RECALL_HOLDOUT_AT_030 + 0.05),
                pr_auc=float(MIN_PRAUC_HOLDOUT + 0.02),
            ),
        },
    )
    selection_path = _write_selection(
        root=tmp_path,
        model_family="nonlinear_hgb",
        variant="default",
        metadata_path=metadata_path,
    )

    result = check_model_regression(
        selection_path=str(selection_path),
        models_root=str(models_root),
    )

    assert result["status"] == "PASS"
    assert result["threshold_used"] == pytest.approx(0.30)
    assert result["recall"] >= MIN_RECALL_HOLDOUT_AT_030
    assert result["pr_auc"] >= MIN_PRAUC_HOLDOUT
    assert result["winner"]["model_version"] == "v-pass"


def test_check_model_regression_fails_when_recall_below_minimum(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    metadata_path = models_root / "baseline_logreg" / "none" / "metadata.json"
    _write_json(
        metadata_path,
        {
            "model_family": "baseline_logreg",
            "variant": "none",
            "evaluation_holdout_at_0.30": _holdout_block(
                recall=float(MIN_RECALL_HOLDOUT_AT_030 - 0.10),
                pr_auc=float(MIN_PRAUC_HOLDOUT + 0.01),
            ),
        },
    )
    selection_path = _write_selection(
        root=tmp_path,
        model_family="baseline_logreg",
        variant="none",
        metadata_path=metadata_path,
    )

    result = check_model_regression(
        selection_path=str(selection_path),
        models_root=str(models_root),
    )

    assert result["status"] == "FAIL"
    assert result["reason"] == "metrics_below_minimum"
    assert result["threshold_used"] == pytest.approx(0.30)
    assert any("failed_gate_recall" in str(note) for note in result["notes"])


def test_check_model_regression_uses_fallback_05_with_warning(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    metadata_path = models_root / "nonlinear_hgb" / "tuned" / "metadata.json"
    _write_json(
        metadata_path,
        {
            "model_family": "nonlinear_hgb",
            "variant": "tuned",
            "evaluation_holdout_at_0.5": {
                "threshold": 0.5,
                "metrics": {
                    "recall": float(MIN_RECALL_HOLDOUT_AT_030 + 0.01),
                    "pr_auc": float(MIN_PRAUC_HOLDOUT + 0.01),
                    "positive_rate": 0.42,
                },
            },
        },
    )
    selection_path = _write_selection(
        root=tmp_path,
        model_family="nonlinear_hgb",
        variant="tuned",
        metadata_path=metadata_path,
        status="WARNING",
    )

    result = check_model_regression(
        selection_path=str(selection_path),
        models_root=str(models_root),
        allow_fallback_05=True,
    )

    assert result["status"] == "WARNING"
    assert result["reason"] == "gates_passed_with_fallback_threshold"
    assert result["threshold_used"] == pytest.approx(0.5)
    assert any("fallback_threshold_used" in str(note) for note in result["notes"])


def test_check_model_regression_skips_when_selection_missing(tmp_path: Path) -> None:
    result = check_model_regression(
        selection_path=str(tmp_path / "artifacts" / "model_selection.json"),
        models_root=str(tmp_path / "artifacts" / "models"),
    )
    assert result["status"] == "SKIPPED"
    assert result["reason"] == "selection_not_found"


def test_check_model_regression_fails_when_winner_metadata_missing(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    selection_path = _write_selection(
        root=tmp_path,
        model_family="nonlinear_hgb",
        variant="default",
        metadata_path=tmp_path / "artifacts" / "models" / "nonlinear_hgb" / "default" / "metadata.json",
    )
    # ensure file does not exist
    if (models_root / "nonlinear_hgb" / "default" / "metadata.json").exists():
        raise AssertionError("fixture setup error")

    result = check_model_regression(
        selection_path=str(selection_path),
        models_root=str(models_root),
    )
    assert result["status"] == "FAIL"
    assert result["reason"] == "winner_metadata_not_found"


def test_check_model_regression_fails_when_winner_metadata_invalid_json(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    metadata_path = models_root / "nonlinear_hgb" / "default" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("{invalid-json", encoding="utf-8")
    selection_path = _write_selection(
        root=tmp_path,
        model_family="nonlinear_hgb",
        variant="default",
        metadata_path=metadata_path,
    )

    result = check_model_regression(
        selection_path=str(selection_path),
        models_root=str(models_root),
    )
    assert result["status"] == "FAIL"
    assert result["reason"] == "winner_metadata_invalid"

