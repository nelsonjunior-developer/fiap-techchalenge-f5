from __future__ import annotations

from src.promotion_policy import evaluate_candidate_against_policy, promotion_decision


def _metadata_with_holdout(
    *,
    holdout_030: dict[str, float] | None = None,
    holdout_050: dict[str, float] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "variant": "default",
        "threshold_policy": {"operational_fixed_threshold": 0.30},
    }
    if holdout_030 is not None:
        payload["evaluation_holdout_at_0.30"] = {
            "threshold": 0.30,
            "metrics": {
                "recall": holdout_030["recall"],
                "pr_auc": holdout_030["pr_auc"],
                "precision": holdout_030.get("precision", 0.5),
                "f1": holdout_030.get("f1", 0.5),
                "roc_auc": holdout_030.get("roc_auc", 0.6),
                "positive_rate": holdout_030["positive_rate"],
            },
            "confusion_matrix": {"tn": 10, "fp": 5, "fn": 4, "tp": 11},
            "notes": [],
        }
    if holdout_050 is not None:
        payload["evaluation_holdout_at_0.5"] = {
            "threshold": 0.5,
            "metrics": {
                "recall": holdout_050["recall"],
                "pr_auc": holdout_050["pr_auc"],
                "precision": holdout_050.get("precision", 0.5),
                "f1": holdout_050.get("f1", 0.5),
                "roc_auc": holdout_050.get("roc_auc", 0.6),
                "positive_rate": holdout_050["positive_rate"],
            },
            "confusion_matrix_at_0.5": {"tn": 12, "fp": 3, "fn": 5, "tp": 10},
            "notes": [],
        }
    return payload


def test_evaluate_candidate_against_policy_pass() -> None:
    metadata = _metadata_with_holdout(
        holdout_030={"recall": 0.62, "pr_auc": 0.71, "positive_rate": 0.55}
    )
    result = evaluate_candidate_against_policy(metadata)
    assert result["status"] == "PASS"
    assert result["passed_gates"] is True
    assert result["threshold_used"] == 0.30


def test_evaluate_candidate_against_policy_fail_when_recall_below_gate() -> None:
    metadata = _metadata_with_holdout(
        holdout_030={"recall": 0.40, "pr_auc": 0.71, "positive_rate": 0.55}
    )
    result = evaluate_candidate_against_policy(metadata)
    assert result["status"] == "FAIL"
    assert result["passed_gates"] is False
    assert any("failed_gate_recall" in str(note) for note in result["notes"])


def test_evaluate_candidate_against_policy_warning_on_threshold_fallback_05() -> None:
    metadata = _metadata_with_holdout(
        holdout_030=None,
        holdout_050={"recall": 0.58, "pr_auc": 0.66, "positive_rate": 0.40},
    )
    result = evaluate_candidate_against_policy(metadata)
    assert result["passed_gates"] is True
    assert result["threshold_used"] == 0.5
    assert result["status"] == "WARNING"
    assert any("fallback_to_threshold_0.5_used" in str(note) for note in result["notes"])


def test_promotion_decision_warning_selection_requires_override() -> None:
    selection_payload = {
        "status": "WARNING",
        "winner": {
            "model_family": "nonlinear_hgb",
            "variant": "default",
            "path_model": "artifacts/models/nonlinear_hgb/default/model.joblib",
            "path_metadata": "artifacts/models/nonlinear_hgb/default/metadata.json",
            "metrics_holdout": {"recall": 0.62, "pr_auc": 0.71, "positive_rate": 0.55},
        },
    }
    metadata = _metadata_with_holdout(
        holdout_030={"recall": 0.62, "pr_auc": 0.71, "positive_rate": 0.55}
    )
    result = promotion_decision(selection_payload, winner_metadata=metadata)
    assert result["status"] == "WARNING"
    assert result["decision"] == "ALLOW_WITH_OVERRIDE"


def test_promotion_decision_fail_selection_blocks() -> None:
    selection_payload = {
        "status": "FAIL",
        "winner": {
            "model_family": "baseline_logreg",
            "variant": "none",
            "metrics_holdout": {"recall": 0.62, "pr_auc": 0.71, "positive_rate": 0.55},
        },
    }
    metadata = _metadata_with_holdout(
        holdout_030={"recall": 0.62, "pr_auc": 0.71, "positive_rate": 0.55}
    )
    result = promotion_decision(selection_payload, winner_metadata=metadata)
    assert result["status"] == "FAIL"
    assert result["decision"] == "BLOCK"
