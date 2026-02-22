"""Champion model non-regression check from selection + metadata artifacts.

CI-friendly mode:
- Reads only JSON artifacts (`artifacts/model_selection.json` and winner `metadata.json`)
- Does not require dataset, sklearn, or model.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.model_selection import extract_holdout_metrics
from src.regression_thresholds import (
    ALLOW_FALLBACK_05,
    FALLBACK_THRESHOLD,
    MIN_PRAUC_HOLDOUT,
    MIN_RECALL_HOLDOUT_AT_030,
    THRESHOLD_PREFERRED,
)
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object at: {path}")
    return payload


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_notes(notes: list[str]) -> list[str]:
    return list(dict.fromkeys(str(note) for note in notes if str(note).strip()))


def _resolve_preference(
    preferred_threshold: float,
    fallback_threshold: float,
    allow_fallback_05: bool,
) -> tuple[float, ...]:
    preferred = float(preferred_threshold)
    fallback = float(fallback_threshold)
    if not bool(allow_fallback_05):
        return (preferred,)
    values: list[float] = [preferred]
    if abs(fallback - preferred) > 1e-9:
        values.append(fallback)
    return tuple(values)


def _resolve_winner_metadata_path(
    *,
    selection: dict[str, Any],
    models_root: str | Path,
) -> tuple[Path | None, dict[str, str], list[str], str | None]:
    notes: list[str] = []
    winner_raw = selection.get("winner")
    if not isinstance(winner_raw, dict):
        return None, {"model_family": "unknown", "variant": "unknown"}, notes, "selection_winner_missing"

    model_family = str(winner_raw.get("model_family") or "unknown").strip() or "unknown"
    variant = str(winner_raw.get("variant") or "unknown").strip() or "unknown"
    identity = {"model_family": model_family, "variant": variant}

    path_metadata_raw = str(winner_raw.get("path_metadata") or "").strip()
    provided_path = Path(path_metadata_raw) if path_metadata_raw else None
    fallback_path = Path(models_root) / model_family / variant / "metadata.json"

    if provided_path is not None and provided_path.exists():
        return provided_path, identity, notes + ["winner_metadata_path_from_selection"], None
    if provided_path is not None and not provided_path.exists():
        notes.append("winner_metadata_path_from_selection_missing_fallback_to_models_root")

    if fallback_path.exists():
        return fallback_path, identity, notes + ["winner_metadata_path_from_models_root"], None

    return None, identity, notes, "winner_metadata_not_found"


def _winner_model_version(
    winner_identity: dict[str, str],
    winner_block: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> str:
    if isinstance(metadata, dict):
        raw = str(metadata.get("model_version") or "").strip()
        if raw:
            return raw
    if isinstance(winner_block, dict):
        raw = str(winner_block.get("model_version") or "").strip()
        if raw:
            return raw
    family = winner_identity.get("model_family", "unknown")
    variant = winner_identity.get("variant", "unknown")
    return f"{family}/{variant}"


def check_model_regression(
    *,
    selection_path: str = "artifacts/model_selection.json",
    models_root: str = "artifacts/models",
    preferred_threshold: float = THRESHOLD_PREFERRED,
    allow_fallback_05: bool = ALLOW_FALLBACK_05,
    min_recall_holdout: float = MIN_RECALL_HOLDOUT_AT_030,
    min_pr_auc_holdout: float = MIN_PRAUC_HOLDOUT,
    fallback_threshold: float = FALLBACK_THRESHOLD,
) -> dict[str, Any]:
    """Check winner holdout metrics against minimum gates without recomputing metrics."""
    selection_file = Path(selection_path)
    if not selection_file.exists():
        return {
            "status": "SKIPPED",
            "reason": "selection_not_found",
            "threshold_used": None,
            "threshold_preferred": float(preferred_threshold),
            "allow_fallback_05": bool(allow_fallback_05),
            "recall": None,
            "pr_auc": None,
            "min_recall": float(min_recall_holdout),
            "min_pr_auc": float(min_pr_auc_holdout),
            "winner": {
                "model_family": "unknown",
                "variant": "unknown",
                "model_version": "unknown",
            },
            "notes": ["selection_artifact_missing_ci_not_blocked"],
        }

    notes: list[str] = []
    try:
        selection = _safe_read_json(selection_file)
    except Exception as exc:
        return {
            "status": "FAIL",
            "reason": "selection_invalid",
            "threshold_used": None,
            "threshold_preferred": float(preferred_threshold),
            "allow_fallback_05": bool(allow_fallback_05),
            "recall": None,
            "pr_auc": None,
            "min_recall": float(min_recall_holdout),
            "min_pr_auc": float(min_pr_auc_holdout),
            "winner": {
                "model_family": "unknown",
                "variant": "unknown",
                "model_version": "unknown",
            },
            "notes": [f"selection_invalid:{exc.__class__.__name__}"],
        }

    winner_block = selection.get("winner") if isinstance(selection.get("winner"), dict) else None
    metadata_path, winner_identity, resolve_notes, resolve_error = _resolve_winner_metadata_path(
        selection=selection,
        models_root=models_root,
    )
    notes.extend(resolve_notes)
    if resolve_error is not None or metadata_path is None:
        return {
            "status": "FAIL",
            "reason": str(resolve_error or "winner_metadata_not_found"),
            "threshold_used": None,
            "threshold_preferred": float(preferred_threshold),
            "allow_fallback_05": bool(allow_fallback_05),
            "recall": None,
            "pr_auc": None,
            "min_recall": float(min_recall_holdout),
            "min_pr_auc": float(min_pr_auc_holdout),
            "winner": {
                "model_family": winner_identity["model_family"],
                "variant": winner_identity["variant"],
                "model_version": _winner_model_version(winner_identity, winner_block, None),
            },
            "notes": _dedupe_notes(notes),
        }

    try:
        metadata = _safe_read_json(metadata_path)
    except Exception as exc:
        notes.append(f"winner_metadata_invalid:{exc.__class__.__name__}")
        return {
            "status": "FAIL",
            "reason": "winner_metadata_invalid",
            "threshold_used": None,
            "threshold_preferred": float(preferred_threshold),
            "allow_fallback_05": bool(allow_fallback_05),
            "recall": None,
            "pr_auc": None,
            "min_recall": float(min_recall_holdout),
            "min_pr_auc": float(min_pr_auc_holdout),
            "winner": {
                "model_family": winner_identity["model_family"],
                "variant": winner_identity["variant"],
                "model_version": _winner_model_version(winner_identity, winner_block, None),
            },
            "notes": _dedupe_notes(notes),
        }

    threshold_preference = _resolve_preference(
        preferred_threshold=preferred_threshold,
        fallback_threshold=fallback_threshold,
        allow_fallback_05=bool(allow_fallback_05),
    )
    extracted = extract_holdout_metrics(metadata, threshold_preference=threshold_preference)
    notes.extend([str(note) for note in extracted.get("notes", [])] if isinstance(extracted, dict) else [])

    metrics_raw = extracted.get("metrics") if isinstance(extracted, dict) else None
    metrics = dict(metrics_raw) if isinstance(metrics_raw, dict) else None
    threshold_used = _to_float_or_none(extracted.get("threshold_used") if isinstance(extracted, dict) else None)
    recall = _to_float_or_none(None if metrics is None else metrics.get("recall"))
    pr_auc = _to_float_or_none(None if metrics is None else metrics.get("pr_auc"))

    winner = {
        "model_family": str(metadata.get("model_family") or winner_identity["model_family"]),
        "variant": str(metadata.get("variant") or winner_identity["variant"]),
        "model_version": _winner_model_version(winner_identity, winner_block, metadata),
        "path_metadata": str(metadata_path),
    }

    if not bool(extracted.get("available")) or metrics is None or recall is None or pr_auc is None:
        return {
            "status": "FAIL",
            "reason": "holdout_metrics_missing",
            "threshold_used": threshold_used,
            "threshold_preferred": float(preferred_threshold),
            "allow_fallback_05": bool(allow_fallback_05),
            "recall": recall,
            "pr_auc": pr_auc,
            "min_recall": float(min_recall_holdout),
            "min_pr_auc": float(min_pr_auc_holdout),
            "winner": winner,
            "notes": _dedupe_notes(notes),
        }

    recall_ok = float(recall) >= float(min_recall_holdout)
    pr_auc_ok = float(pr_auc) >= float(min_pr_auc_holdout)
    if not recall_ok:
        notes.append(f"failed_gate_recall<{float(min_recall_holdout):.2f}")
    if not pr_auc_ok:
        notes.append(f"failed_gate_pr_auc<{float(min_pr_auc_holdout):.2f}")

    status = "PASS"
    reason = "gates_passed"
    used_fallback = (
        threshold_used is not None
        and abs(float(threshold_used) - float(preferred_threshold)) > 1e-9
    )
    if used_fallback:
        notes.append(
            "fallback_threshold_used:{:.2f}_preferred:{:.2f}".format(
                float(threshold_used), float(preferred_threshold)
            )
        )

    if not (recall_ok and pr_auc_ok):
        status = "FAIL"
        reason = "metrics_below_minimum"
    elif used_fallback:
        # Exit code remains 0 in CLI, but status is explicit so the caller can escalate if desired.
        status = "WARNING"
        reason = "gates_passed_with_fallback_threshold"

    selection_status = str(selection.get("status") or "UNKNOWN").upper()
    if selection_status == "WARNING":
        notes.append("selection_artifact_status_warning")
    elif selection_status == "FAIL":
        notes.append("selection_artifact_status_fail")

    return {
        "status": str(status),
        "reason": str(reason),
        "threshold_used": threshold_used,
        "threshold_preferred": float(preferred_threshold),
        "fallback_threshold": float(fallback_threshold),
        "allow_fallback_05": bool(allow_fallback_05),
        "recall": float(recall),
        "pr_auc": float(pr_auc),
        "min_recall": float(min_recall_holdout),
        "min_pr_auc": float(min_pr_auc_holdout),
        "winner": winner,
        "selection_status": selection_status,
        "selection_path": str(selection_file),
        "models_root": str(models_root),
        "notes": _dedupe_notes(notes),
    }


def _exit_code_for_status(status: str) -> int:
    normalized = str(status or "").upper()
    if normalized == "FAIL":
        return 1
    if normalized in {"PASS", "WARNING", "SKIPPED"}:
        return 0
    return 1


def _print_human_summary(result: dict[str, Any]) -> None:
    print(
        "Regression check | status={status} | reason={reason} | threshold_used={threshold} | "
        "recall={recall} | pr_auc={pr_auc} | winner={family}/{variant}".format(
            status=result.get("status"),
            reason=result.get("reason"),
            threshold=result.get("threshold_used"),
            recall=result.get("recall"),
            pr_auc=result.get("pr_auc"),
            family=((result.get("winner") or {}).get("model_family") if isinstance(result.get("winner"), dict) else "unknown"),
            variant=((result.get("winner") or {}).get("variant") if isinstance(result.get("winner"), dict) else "unknown"),
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check champion non-regression gates from artifacts/model_selection.json and winner metadata."
        )
    )
    parser.add_argument("--selection-path", type=str, default="artifacts/model_selection.json")
    parser.add_argument("--models-root", type=str, default="artifacts/models")
    parser.add_argument("--preferred-threshold", type=float, default=THRESHOLD_PREFERRED)
    parser.add_argument("--allow-fallback-05", type=int, default=1 if ALLOW_FALLBACK_05 else 0, choices=[0, 1])
    parser.add_argument("--min-recall-holdout", type=float, default=MIN_RECALL_HOLDOUT_AT_030)
    parser.add_argument("--min-prauc-holdout", type=float, default=MIN_PRAUC_HOLDOUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    result = check_model_regression(
        selection_path=str(args.selection_path),
        models_root=str(args.models_root),
        preferred_threshold=float(args.preferred_threshold),
        allow_fallback_05=bool(int(args.allow_fallback_05)),
        min_recall_holdout=float(args.min_recall_holdout),
        min_pr_auc_holdout=float(args.min_prauc_holdout),
    )
    _print_human_summary(result)
    status = str(result.get("status") or "FAIL")
    if status == "FAIL":
        _logger.error("model regression check failed | reason=%s", result.get("reason"))
    elif status == "WARNING":
        _logger.warning("model regression check warning | reason=%s", result.get("reason"))
    elif status == "SKIPPED":
        _logger.info("model regression check skipped | reason=%s", result.get("reason"))
    else:
        _logger.info("model regression check passed")
    raise SystemExit(_exit_code_for_status(status))


if __name__ == "__main__":
    main()
