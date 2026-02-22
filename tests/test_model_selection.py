from __future__ import annotations

import json
from pathlib import Path

from src.model_selection import (
    discover_model_metadatas,
    extract_holdout_metrics,
    run_model_selection,
    select_best_model,
)


def _write_metadata(
    root: Path,
    family: str,
    variant: str,
    *,
    holdout_030: dict[str, float] | None = None,
    holdout_050: dict[str, float] | None = None,
) -> Path:
    target = root / family / variant
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.joblib").write_bytes(b"fake")

    payload: dict[str, object] = {
        "model_kind": "dummy",
        "variant": variant,
        "train_pair": {"year_t": 2022, "year_t1": 2023},
    }

    if holdout_030 is not None:
        payload["evaluation_holdout_at_0.30"] = {
            "pair": "2023->2024",
            "threshold": 0.30,
            "n": 100,
            "n_pos": 40,
            "prevalence": 0.40,
            "metrics": {
                "recall": holdout_030["recall"],
                "precision": holdout_030.get("precision", 0.5),
                "f1": holdout_030.get("f1", 0.5),
                "roc_auc": holdout_030.get("roc_auc", 0.6),
                "pr_auc": holdout_030["pr_auc"],
                "positive_rate": holdout_030["positive_rate"],
            },
            "confusion_matrix": {"tn": 40, "fp": 20, "fn": 10, "tp": 30},
            "notes": [],
        }

    if holdout_050 is not None:
        payload["evaluation_holdout_at_0.5"] = {
            "pair": "2023->2024",
            "threshold": 0.50,
            "n": 100,
            "n_pos": 40,
            "prevalence": 0.40,
            "metrics": {
                "recall": holdout_050["recall"],
                "precision": holdout_050.get("precision", 0.5),
                "f1": holdout_050.get("f1", 0.5),
                "roc_auc": holdout_050.get("roc_auc", 0.6),
                "pr_auc": holdout_050["pr_auc"],
                "positive_rate": holdout_050["positive_rate"],
            },
            "confusion_matrix_at_0.5": {"tn": 42, "fp": 18, "fn": 12, "tp": 28},
            "notes": [],
        }

    output = target / "metadata.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def test_empty_discovery_returns_fail(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    out_json = tmp_path / "model_selection.json"
    report = run_model_selection(
        models_root=models_root,
        output_json=out_json,
        write_markdown=False,
    )
    assert report["status"] == "FAIL"
    assert report["errors"]
    assert out_json.exists()


def test_missing_holdout_warning_and_strict_fail(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    _write_metadata(
        models_root,
        "baseline_logreg",
        "none",
        holdout_030={"recall": 0.60, "pr_auc": 0.62, "positive_rate": 0.45},
    )
    _write_metadata(
        models_root,
        "nonlinear_hgb",
        "tuned",
        holdout_030=None,
        holdout_050=None,
    )

    discovered = discover_model_metadatas(models_root)
    warn_report = select_best_model(discovered, fail_on_missing_holdout=False)
    assert warn_report["status"] == "WARNING"
    assert any("missing holdout metrics" in warning.lower() for warning in warn_report["warnings"])

    fail_report = select_best_model(discovered, fail_on_missing_holdout=True)
    assert fail_report["status"] == "FAIL"
    assert fail_report["errors"]


def test_ranking_is_deterministic(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    _write_metadata(
        models_root,
        "baseline_logreg",
        "none",
        holdout_030={"recall": 0.70, "pr_auc": 0.65, "positive_rate": 0.40},
    )
    _write_metadata(
        models_root,
        "nonlinear_hgb",
        "default",
        holdout_030={"recall": 0.70, "pr_auc": 0.66, "positive_rate": 0.80},
    )
    _write_metadata(
        models_root,
        "nonlinear_hgb",
        "tuned",
        holdout_030={"recall": 0.70, "pr_auc": 0.66, "positive_rate": 0.30},
    )

    discovered = discover_model_metadatas(models_root)
    report = select_best_model(
        discovered,
        min_recall=0.45,
        min_pr_auc=0.60,
        fail_on_missing_holdout=False,
    )
    assert report["status"] == "PASS"
    assert report["winner"]["model_family"] == "nonlinear_hgb"
    assert report["winner"]["variant"] == "tuned"
    assert report["winner"]["path_model"] == str(
        models_root / "nonlinear_hgb" / "tuned" / "model.joblib"
    )
    assert report["winner"]["path_metadata"] == str(
        models_root / "nonlinear_hgb" / "tuned" / "metadata.json"
    )
    ranked = report["ranked_candidates"]
    assert ranked[0]["model_family"] == "nonlinear_hgb"
    assert ranked[0]["variant"] == "tuned"
    assert ranked[1]["variant"] == "default"
    assert ranked[2]["model_family"] == "baseline_logreg"


def test_threshold_fallback_from_030_to_05_adds_warning(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    _write_metadata(
        models_root,
        "baseline_logreg",
        "none",
        holdout_030=None,
        holdout_050={"recall": 0.55, "pr_auc": 0.61, "positive_rate": 0.35},
    )

    discovered = discover_model_metadatas(models_root)
    candidate = next(item for item in discovered if item["variant"] == "none")
    extracted = extract_holdout_metrics(candidate["metadata"])
    assert extracted["available"] is True
    assert extracted["threshold_used"] == 0.5
    assert any("fallback_to_threshold_0.5_used" in note for note in extracted["notes"])

    report = select_best_model(discovered)
    assert report["status"] == "WARNING"
    assert any("fallback threshold 0.5 used" in warning.lower() for warning in report["warnings"])


def test_discovery_ignores_release_directories(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    _write_metadata(
        models_root,
        "baseline_logreg",
        "none",
        holdout_030={"recall": 0.60, "pr_auc": 0.62, "positive_rate": 0.45},
    )

    release_dir = models_root / "releases" / "2026-02-22T14-05-33Z__deadbeef"
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "model.joblib").write_bytes(b"release")
    (release_dir / "metadata.json").write_text(
        json.dumps({"variant": "release-copy"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    discovered = discover_model_metadatas(models_root)
    keys = {(item["model_family"], item["variant"]) for item in discovered}
    assert ("baseline_logreg", "none") in keys
    assert all(family != "releases" for family, _ in keys)
