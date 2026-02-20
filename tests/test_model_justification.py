from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.model_justification import (
    build_model_justification,
    load_model_selection,
    main as justification_main,
    persist_justification,
    render_justification_md,
)


def _build_selection_payload(
    *,
    status: str = "PASS",
    winner_threshold: float = 0.30,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    return {
        "generated_at": "2026-02-20T17:30:39.507056+00:00",
        "status": status,
        "selection_criteria": {
            "threshold_used": 0.30,
            "min_recall_holdout": 0.45,
            "min_pr_auc_holdout": 0.60,
            "ranking_order": [
                "recall_desc",
                "pr_auc_desc",
                "positive_rate_asc",
                "name_lex",
            ],
        },
        "winner": {
            "model_family": "nonlinear_hgb",
            "variant": "default",
            "path_model": "artifacts/models/nonlinear_hgb/default/model.joblib",
            "path_metadata": "artifacts/models/nonlinear_hgb/default/metadata.json",
            "metrics_holdout": {
                "recall": 0.7857142857,
                "precision": 0.4481481481,
                "f1": 0.5707547169,
                "roc_auc": 0.6926099065,
                "pr_auc": 0.6905036381,
                "positive_rate": 0.7058823529,
                "confusion_matrix": {"tn": 159, "fp": 298, "fn": 66, "tp": 242},
            },
        },
        "ranked_candidates": [
            {
                "model_family": "nonlinear_hgb",
                "variant": "default",
                "path_model": "artifacts/models/nonlinear_hgb/default/model.joblib",
                "path_metadata": "artifacts/models/nonlinear_hgb/default/metadata.json",
                "eligible": True,
                "passed_gates": True,
                "threshold_used": winner_threshold,
                "metrics_holdout": {
                    "recall": 0.7857142857,
                    "precision": 0.4481481481,
                    "f1": 0.5707547169,
                    "roc_auc": 0.6926099065,
                    "pr_auc": 0.6905036381,
                    "positive_rate": 0.7058823529,
                    "confusion_matrix": {"tn": 159, "fp": 298, "fn": 66, "tp": 242},
                },
                "notes": [],
                "rank": 1,
            },
            {
                "model_family": "nonlinear_hgb",
                "variant": "tuned",
                "path_model": "artifacts/models/nonlinear_hgb/tuned/model.joblib",
                "path_metadata": "artifacts/models/nonlinear_hgb/tuned/metadata.json",
                "eligible": True,
                "passed_gates": True,
                "threshold_used": 0.50,
                "metrics_holdout": {
                    "recall": 0.7370129870,
                    "precision": 0.5218390805,
                    "f1": 0.6110363392,
                    "roc_auc": 0.7226690159,
                    "pr_auc": 0.7114147094,
                    "positive_rate": 0.5686274510,
                    "confusion_matrix": {"tn": 249, "fp": 208, "fn": 81, "tp": 227},
                },
                "notes": [],
                "rank": 2,
            },
        ],
        "notes": [],
        "warnings": warnings or [],
        "errors": [],
    }


def test_happy_path_generates_markdown_with_winner_threshold_and_metrics(tmp_path: Path) -> None:
    selection_payload = _build_selection_payload(status="PASS", winner_threshold=0.30)
    selection_path = tmp_path / "model_selection.json"
    selection_path.write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    selection = load_model_selection(selection_path)
    justif = build_model_justification(selection)
    markdown = render_justification_md(justif)

    assert justif["status"] == "PASS"
    assert justif["winner"]["family"] == "nonlinear_hgb"
    assert justif["chosen_threshold"]["label"] == "0.30"
    assert "Modelo campeao: `nonlinear_hgb/default`" in markdown
    assert "| Recall | 0.7857 |" in markdown
    assert "Threshold aplicado na selecao: `0.30`" in markdown

    output_md = tmp_path / "docs" / "model_final_justification.md"
    output_json = tmp_path / "artifacts" / "model_final_justification.json"
    persisted = persist_justification(
        justif,
        output_md=output_md,
        output_json=output_json,
        write_json=True,
    )
    assert Path(persisted["output_md"] or "").exists()
    assert Path(persisted["output_json"] or "").exists()


def test_warning_path_explicitly_mentions_warning_and_fallback() -> None:
    selection_payload = _build_selection_payload(
        status="WARNING",
        winner_threshold=0.50,
        warnings=[
            "Fallback threshold 0.5 used because holdout@0.30 is unavailable on some variants."
        ],
    )
    justif = build_model_justification(selection_payload)
    markdown = render_justification_md(justif)

    assert justif["status"] == "WARNING"
    assert justif["chosen_threshold"]["is_fallback"] is True
    assert "WARNING" in markdown
    assert "fallback" in markdown.lower()
    assert "Status da selecao: **WARNING**" in markdown
    assert "Motivo:" in markdown
    assert "Decisao aplicada:" in markdown
    assert "ALERTA" in markdown


def test_winner_missing_in_ranked_candidates_forces_fail() -> None:
    selection_payload = _build_selection_payload(status="PASS", winner_threshold=0.30)
    selection_payload["winner"] = {
        "model_family": "baseline_logreg",
        "variant": "ghost_variant",
        "path_model": "artifacts/models/baseline_logreg/ghost_variant/model.joblib",
        "path_metadata": "artifacts/models/baseline_logreg/ghost_variant/metadata.json",
        "metrics_holdout": {
            "recall": 0.5,
            "pr_auc": 0.6,
            "positive_rate": 0.4,
        },
    }
    justif = build_model_justification(selection_payload)
    markdown = render_justification_md(justif)

    assert justif["status"] == "FAIL"
    assert "winner not found in ranked_candidates" in justif["errors"]
    assert "winner not found in ranked_candidates" in justif["notes"]
    assert "status: **FAIL**" in markdown


def test_cli_missing_selection_fails_with_clear_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing_model_selection.json"
    output_md = tmp_path / "docs" / "model_final_justification.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--selection-path",
            str(missing),
            "--output-md",
            str(output_md),
            "--no-json",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        justification_main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "model_selection" in captured.err.lower()
    assert "before generating justification" in captured.err.lower()
