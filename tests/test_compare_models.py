from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.compare_models import (
    build_comparison_report,
    discover_metadata_files,
    main as compare_main,
    run_compare_models,
)


def _write_metadata(
    root: Path,
    family: str,
    variant: str,
    *,
    train_recall: float,
    train_pr_auc: float,
    holdout_recall: float | None,
    holdout_pr_auc: float | None,
    holdout_pos_rate: float | None = None,
) -> Path:
    target = root / family / variant
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_kind": "dummy",
        "variant": variant,
        "train_pair": {"year_t": 2022, "year_t1": 2023},
        "evaluation_train": {
            "pair": "2022->2023",
            "threshold": 0.5,
            "n": 100,
            "n_pos": 40,
            "prevalence": 0.4,
            "metrics": {
                "recall": train_recall,
                "precision": 0.5,
                "f1": 0.5,
                "roc_auc": 0.6,
                "pr_auc": train_pr_auc,
                "positive_rate": 0.4,
            },
            "pred_proba_summary": {"min": 0.1, "mean": 0.5, "p50": 0.5, "p95": 0.9, "max": 0.99},
            "notes": [],
        },
        "evaluation_holdout": (
            None
            if holdout_recall is None
            else {
                "pair": "2023->2024",
                "threshold": 0.5,
                "n": 100,
                "n_pos": 40,
                "prevalence": 0.4,
                "metrics": {
                    "recall": holdout_recall,
                    "precision": 0.5,
                    "f1": 0.5,
                    "roc_auc": 0.6,
                    "pr_auc": holdout_pr_auc,
                    "positive_rate": holdout_pos_rate
                    if holdout_pos_rate is not None
                    else 0.4,
                },
                "pred_proba_summary": {
                    "min": 0.1,
                    "mean": 0.5,
                    "p50": 0.5,
                    "p95": 0.9,
                    "max": 0.99,
                },
                "notes": [],
            }
        ),
        "metrics_train_at_0.5": {
            "recall": train_recall,
            "precision": 0.5,
            "f1": 0.5,
            "roc_auc": 0.6,
            "pr_auc": train_pr_auc,
            "positive_rate_at_threshold": 0.4,
        },
        "class_imbalance_strategy": {
            "evidence": {
                "by_variant_threshold_0.5": {
                    variant: {
                        "holdout": (
                            None
                            if holdout_recall is None
                            else {
                                "recall": holdout_recall,
                                "precision": 0.5,
                                "f1": 0.5,
                                "roc_auc": 0.6,
                                "pr_auc": holdout_pr_auc,
                                "positive_rate_at_threshold": holdout_pos_rate
                                if holdout_pos_rate is not None
                                else 0.4,
                            }
                        )
                    }
                }
            }
        },
    }
    output = target / "metadata.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _build_fake_tree(root: Path) -> None:
    _write_metadata(
        root,
        "baseline_logreg",
        "none",
        train_recall=0.90,
        train_pr_auc=0.95,
        holdout_recall=0.51,
        holdout_pr_auc=0.63,
        holdout_pos_rate=0.38,
    )
    _write_metadata(
        root,
        "baseline_logreg",
        "balanced",
        train_recall=0.86,
        train_pr_auc=0.96,
        holdout_recall=0.45,
        holdout_pr_auc=0.65,
        holdout_pos_rate=0.33,
    )
    _write_metadata(
        root,
        "nonlinear_hgb",
        "default",
        train_recall=0.88,
        train_pr_auc=0.94,
        holdout_recall=0.54,
        holdout_pr_auc=0.66,
        holdout_pos_rate=0.36,
    )
    _write_metadata(
        root,
        "nonlinear_hgb",
        "tuned",
        train_recall=0.87,
        train_pr_auc=0.95,
        holdout_recall=0.50,
        holdout_pr_auc=0.64,
        holdout_pos_rate=0.34,
    )


def test_discovery_and_parse_reads_legacy_holdout_path(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    _build_fake_tree(models_root)

    discovered = discover_metadata_files(models_root)
    assert len(discovered) == 4

    report = build_comparison_report(models_root=models_root)
    assert len(report["rows"]) == 4
    assert report["status"] in {"PASS", "WARNING"}


def test_parse_uses_new_evaluation_paths_when_available(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    _build_fake_tree(models_root)
    report = build_comparison_report(models_root=models_root)
    row = next(
        item
        for item in report["rows"]
        if item["model_family"] == "baseline_logreg" and item["variant"] == "none"
    )
    assert row["metrics"]["train"]["recall_at_0.5"] == pytest.approx(0.90)
    assert row["metrics"]["holdout"]["recall_at_0.5"] == pytest.approx(0.51)


def test_ranking_policy_recall_then_pr_auc(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    _build_fake_tree(models_root)
    report = build_comparison_report(models_root=models_root)
    winner = report["ranking"]["winner"]
    assert winner == {"model_family": "nonlinear_hgb", "variant": "default"}
    table = report["ranking"]["table"]
    assert table[0]["recall_holdout_at_0.5"] >= table[1]["recall_holdout_at_0.5"]


def test_missing_holdout_warning_and_strict_fail(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    _build_fake_tree(models_root)
    # Remove holdout for one variant.
    _write_metadata(
        models_root,
        "nonlinear_hgb",
        "tuned",
        train_recall=0.87,
        train_pr_auc=0.95,
        holdout_recall=None,
        holdout_pr_auc=None,
    )

    report_warn = build_comparison_report(
        models_root=models_root,
        fail_on_missing_holdout=False,
    )
    assert report_warn["status"] == "WARNING"
    assert report_warn["warnings"]

    report_fail = build_comparison_report(
        models_root=models_root,
        fail_on_missing_holdout=True,
    )
    assert report_fail["status"] == "FAIL"
    assert report_fail["errors"]


def test_privacy_report_has_no_forbidden_keys(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    _build_fake_tree(models_root)
    out_json = tmp_path / "comparison.json"
    report = run_compare_models(
        models_root=models_root,
        out_json=out_json,
        write_markdown=False,
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    keys = set()

    def _collect(obj: object) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.add(str(key).lower())
                _collect(value)
        elif isinstance(obj, list):
            for item in obj:
                _collect(item)

    _collect(payload)
    forbidden = {"ra", "ra_list", "ids", "student_ids", "students", "records"}
    assert forbidden.isdisjoint(keys)
    assert report["status"] in {"PASS", "WARNING"}


def test_cli_smoke_and_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_root = tmp_path / "artifacts" / "models"
    _build_fake_tree(models_root)
    out_json = tmp_path / "model_comparison.json"
    out_md = tmp_path / "model_comparison.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--models-root",
            str(models_root),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
    )
    compare_main()
    assert out_json.exists()
    assert out_md.exists()


def test_empty_discovery_is_fail(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    report = build_comparison_report(models_root=models_root)
    assert report["status"] == "FAIL"
    assert report["errors"]
