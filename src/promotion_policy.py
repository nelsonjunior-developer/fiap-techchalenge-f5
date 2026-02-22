"""Objective promotion policy (staging -> prod) based on holdout metrics."""

from __future__ import annotations

from typing import Any

from src.model_selection import extract_holdout_metrics

DEFAULT_MIN_RECALL_HOLDOUT = 0.45
DEFAULT_MIN_PRAUC_HOLDOUT = 0.60
DEFAULT_POSITIVE_RATE_WARNING = 0.85


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _winner_from_selection(selection: dict[str, Any]) -> dict[str, Any] | None:
    winner = selection.get("winner")
    return winner if isinstance(winner, dict) else None


def evaluate_candidate_against_policy(
    metadata: dict[str, Any],
    *,
    min_recall_holdout: float = DEFAULT_MIN_RECALL_HOLDOUT,
    min_pr_auc_holdout: float = DEFAULT_MIN_PRAUC_HOLDOUT,
    positive_rate_warning: float = DEFAULT_POSITIVE_RATE_WARNING,
) -> dict[str, Any]:
    """Evaluate winner candidate metadata against promotion gates and guardrails."""
    notes: list[str] = []
    extracted = extract_holdout_metrics(metadata)
    notes.extend(list(extracted.get("notes", [])))
    threshold_used = _to_float_or_none(extracted.get("threshold_used"))

    metrics_raw = extracted.get("metrics")
    metrics = dict(metrics_raw) if isinstance(metrics_raw, dict) else None
    if not bool(extracted.get("available")) or not isinstance(metrics, dict):
        notes.append("promotion_policy_holdout_metrics_unavailable")
        return {
            "threshold_used": threshold_used,
            "passed_gates": False,
            "gates": {
                "min_recall_holdout": float(min_recall_holdout),
                "min_pr_auc_holdout": float(min_pr_auc_holdout),
                "positive_rate_warning": float(positive_rate_warning),
                "recall_passed": None,
                "pr_auc_passed": None,
                "positive_rate_guardrail_ok": None,
            },
            "metrics": None,
            "status": "FAIL",
            "notes": list(dict.fromkeys(str(note) for note in notes)),
        }

    recall = _to_float_or_none(metrics.get("recall"))
    pr_auc = _to_float_or_none(metrics.get("pr_auc"))
    positive_rate = _to_float_or_none(metrics.get("positive_rate"))

    recall_passed = recall is not None and recall >= float(min_recall_holdout)
    pr_auc_passed = pr_auc is not None and pr_auc >= float(min_pr_auc_holdout)
    passed_gates = bool(recall_passed and pr_auc_passed)

    positive_rate_guardrail_ok: bool | None = None
    if positive_rate is not None:
        positive_rate_guardrail_ok = positive_rate <= float(positive_rate_warning)
        if positive_rate_guardrail_ok is False:
            notes.append(
                "positive_rate_guardrail_warning: positive_rate>{:.2f} (actual={:.4f})".format(
                    float(positive_rate_warning), float(positive_rate)
                )
            )

    if recall is not None and not recall_passed:
        notes.append(
            "failed_gate_recall<{:.2f} (actual={:.4f})".format(
                float(min_recall_holdout), float(recall)
            )
        )
    if pr_auc is not None and not pr_auc_passed:
        notes.append(
            "failed_gate_pr_auc<{:.2f} (actual={:.4f})".format(
                float(min_pr_auc_holdout), float(pr_auc)
            )
        )

    status = "PASS"
    if not passed_gates:
        status = "FAIL"
    else:
        used_fallback_threshold = threshold_used is not None and abs(float(threshold_used) - 0.30) > 1e-9
        if used_fallback_threshold:
            status = "WARNING"
        if positive_rate_guardrail_ok is False:
            status = "WARNING"

    return {
        "threshold_used": threshold_used,
        "passed_gates": bool(passed_gates),
        "gates": {
            "min_recall_holdout": float(min_recall_holdout),
            "min_pr_auc_holdout": float(min_pr_auc_holdout),
            "positive_rate_warning": float(positive_rate_warning),
            "recall_passed": bool(recall_passed) if recall is not None else None,
            "pr_auc_passed": bool(pr_auc_passed) if pr_auc is not None else None,
            "positive_rate_guardrail_ok": positive_rate_guardrail_ok,
        },
        "metrics": metrics,
        "status": status,
        "notes": list(dict.fromkeys(str(note) for note in notes)),
    }


def promotion_decision(
    model_selection: dict[str, Any],
    winner_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic promotion decision from selection status + policy evaluation."""
    notes: list[str] = []
    warnings = model_selection.get("warnings")
    errors = model_selection.get("errors")
    if isinstance(warnings, list):
        notes.extend([f"selection_warning:{str(item)}" for item in warnings])
    if isinstance(errors, list):
        notes.extend([f"selection_error:{str(item)}" for item in errors])

    winner = _winner_from_selection(model_selection)
    if winner is None:
        return {
            "status": "FAIL",
            "decision": "BLOCK",
            "reason": "Selection artifact missing winner block.",
            "winner": None,
            "selection_status": str(model_selection.get("status") or "UNKNOWN"),
            "policy_evaluation": {
                "status": "FAIL",
                "passed_gates": False,
                "threshold_used": None,
                "metrics": None,
                "gates": {
                    "min_recall_holdout": float(DEFAULT_MIN_RECALL_HOLDOUT),
                    "min_pr_auc_holdout": float(DEFAULT_MIN_PRAUC_HOLDOUT),
                    "positive_rate_warning": float(DEFAULT_POSITIVE_RATE_WARNING),
                    "recall_passed": None,
                    "pr_auc_passed": None,
                    "positive_rate_guardrail_ok": None,
                },
                "notes": ["promotion_policy_holdout_metrics_unavailable"],
            },
            "notes": list(dict.fromkeys(notes)),
        }

    policy_eval: dict[str, Any]
    if isinstance(winner_metadata, dict):
        policy_eval = evaluate_candidate_against_policy(winner_metadata)
    else:
        # Fallback when caller only has selection artifact: partial evidence from winner.metrics_holdout.
        winner_metrics = winner.get("metrics_holdout")
        metrics = dict(winner_metrics) if isinstance(winner_metrics, dict) else None
        if isinstance(metrics, dict):
            recall = _to_float_or_none(metrics.get("recall"))
            pr_auc = _to_float_or_none(metrics.get("pr_auc"))
            positive_rate = _to_float_or_none(metrics.get("positive_rate"))
            passed_gates = (
                recall is not None
                and pr_auc is not None
                and recall >= DEFAULT_MIN_RECALL_HOLDOUT
                and pr_auc >= DEFAULT_MIN_PRAUC_HOLDOUT
            )
            policy_eval = {
                "threshold_used": None,
                "passed_gates": bool(passed_gates),
                "gates": {
                    "min_recall_holdout": float(DEFAULT_MIN_RECALL_HOLDOUT),
                    "min_pr_auc_holdout": float(DEFAULT_MIN_PRAUC_HOLDOUT),
                    "positive_rate_warning": float(DEFAULT_POSITIVE_RATE_WARNING),
                    "recall_passed": None if recall is None else recall >= DEFAULT_MIN_RECALL_HOLDOUT,
                    "pr_auc_passed": None if pr_auc is None else pr_auc >= DEFAULT_MIN_PRAUC_HOLDOUT,
                    "positive_rate_guardrail_ok": (
                        None
                        if positive_rate is None
                        else positive_rate <= DEFAULT_POSITIVE_RATE_WARNING
                    ),
                },
                "metrics": metrics,
                "status": "WARNING",
                "notes": ["policy_evaluation_from_model_selection_winner_metrics_only"],
            }
        else:
            policy_eval = {
                "threshold_used": None,
                "passed_gates": False,
                "gates": {
                    "min_recall_holdout": float(DEFAULT_MIN_RECALL_HOLDOUT),
                    "min_pr_auc_holdout": float(DEFAULT_MIN_PRAUC_HOLDOUT),
                    "positive_rate_warning": float(DEFAULT_POSITIVE_RATE_WARNING),
                    "recall_passed": None,
                    "pr_auc_passed": None,
                    "positive_rate_guardrail_ok": None,
                },
                "metrics": None,
                "status": "FAIL",
                "notes": ["promotion_policy_holdout_metrics_unavailable"],
            }

    selection_status = str(model_selection.get("status") or "UNKNOWN").upper()
    reason = ""
    decision = "BLOCK"
    status = "FAIL"

    metrics_available = isinstance(policy_eval.get("metrics"), dict)
    if selection_status == "FAIL":
        decision = "BLOCK"
        status = "FAIL"
        reason = "Selection artifact status=FAIL."
    elif selection_status == "WARNING":
        if not metrics_available:
            decision = "BLOCK"
            status = "FAIL"
            reason = "Selection status=WARNING but policy metrics are unavailable."
        else:
            decision = "ALLOW_WITH_OVERRIDE"
            status = "WARNING"
            reason = "Selection artifact status=WARNING; explicit override is required to promote."
    elif selection_status == "PASS":
        if not metrics_available:
            decision = "BLOCK"
            status = "FAIL"
            reason = "Selection status=PASS but policy metrics are unavailable."
        elif bool(policy_eval.get("passed_gates")):
            decision = "ALLOW"
            status = "PASS"
            reason = "Selection status=PASS and promotion policy gates passed."
        else:
            decision = "BLOCK"
            status = "FAIL"
            reason = "Selection status=PASS but promotion policy gates failed (inconsistent artifacts)."
    else:
        decision = "BLOCK"
        status = "FAIL"
        reason = f"Unsupported selection status: {selection_status!r}"

    notes.extend(list(policy_eval.get("notes", [])) if isinstance(policy_eval.get("notes"), list) else [])
    return {
        "status": status,
        "decision": decision,
        "reason": reason,
        "winner": {
            "model_family": str(winner.get("model_family") or ""),
            "variant": str(winner.get("variant") or ""),
            "path_model": winner.get("path_model"),
            "path_metadata": winner.get("path_metadata"),
        },
        "selection_status": selection_status,
        "policy_evaluation": policy_eval,
        "notes": list(dict.fromkeys(str(note) for note in notes)),
    }


__all__ = [
    "DEFAULT_MIN_PRAUC_HOLDOUT",
    "DEFAULT_MIN_RECALL_HOLDOUT",
    "DEFAULT_POSITIVE_RATE_WARNING",
    "evaluate_candidate_against_policy",
    "promotion_decision",
]
