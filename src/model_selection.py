"""Deterministic model champion selection based on holdout metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.privacy import find_forbidden_json_keys
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_RANKING_ORDER = ["recall_desc", "pr_auc_desc", "positive_rate_asc", "name_lex"]
_ALLOWED_MODEL_FAMILIES = {"baseline_logreg", "nonlinear_hgb"}


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid metadata format (expected object): {path}")
    return payload


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _threshold_label(threshold: float) -> str:
    value = float(threshold)
    if abs(value - 0.30) < 1e-9:
        return "0.30"
    if abs(value - 0.50) < 1e-9:
        return "0.5"
    return f"{value:.2f}"


def _extract_confusion_matrix(
    block: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, int] | None:
    cm = block.get("confusion_matrix")
    if not isinstance(cm, dict):
        key = f"confusion_matrix_at_{_threshold_label(threshold)}"
        cm = block.get(key)
    if not isinstance(cm, dict):
        # Legacy fallback for old holdout@0.5 blocks.
        cm = block.get("confusion_matrix_at_0.5")
    if not isinstance(cm, dict):
        return None
    required = {"tn", "fp", "fn", "tp"}
    if not required.issubset(set(cm.keys())):
        return None
    try:
        return {k: int(cm[k]) for k in ("tn", "fp", "fn", "tp")}
    except (TypeError, ValueError):
        return None


def _extract_from_eval_block(
    block: dict[str, Any] | None,
    *,
    threshold: float,
) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    metrics = block.get("metrics")
    if not isinstance(metrics, dict):
        return None

    positive_rate = metrics.get("positive_rate")
    if positive_rate is None:
        positive_rate = metrics.get("positive_rate_at_threshold")
    if positive_rate is None:
        positive_rate = block.get("positive_rate")

    return {
        "recall": _to_float_or_none(metrics.get("recall")),
        "precision": _to_float_or_none(metrics.get("precision")),
        "f1": _to_float_or_none(metrics.get("f1")),
        "roc_auc": _to_float_or_none(metrics.get("roc_auc")),
        "pr_auc": _to_float_or_none(metrics.get("pr_auc")),
        "positive_rate": _to_float_or_none(positive_rate),
        "confusion_matrix": _extract_confusion_matrix(block, threshold=threshold),
    }


def _extract_from_metrics_block(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    positive_rate = raw.get("positive_rate")
    if positive_rate is None:
        positive_rate = raw.get("positive_rate_at_threshold")
    return {
        "recall": _to_float_or_none(raw.get("recall")),
        "precision": _to_float_or_none(raw.get("precision")),
        "f1": _to_float_or_none(raw.get("f1")),
        "roc_auc": _to_float_or_none(raw.get("roc_auc")),
        "pr_auc": _to_float_or_none(raw.get("pr_auc")),
        "positive_rate": _to_float_or_none(positive_rate),
        "confusion_matrix": None,
    }


def _is_complete_holdout_metrics(metrics: dict[str, Any] | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    return (
        metrics.get("recall") is not None
        and metrics.get("pr_auc") is not None
        and metrics.get("positive_rate") is not None
    )


def discover_model_metadatas(models_root: str | Path = "artifacts/models") -> list[dict[str, Any]]:
    """Discover and parse metadata under artifacts/models/<family>/<variant>/metadata.json."""
    root = Path(models_root)
    items: list[dict[str, Any]] = []
    if not root.exists():
        return items

    for metadata_path in sorted(root.rglob("metadata.json")):
        rel = metadata_path.relative_to(root)
        if len(rel.parts) < 3:
            continue
        model_family = str(rel.parts[0])
        if model_family not in _ALLOWED_MODEL_FAMILIES:
            continue
        variant = str(rel.parts[1])
        model_path = metadata_path.parent / "model.joblib"
        try:
            metadata = _safe_read_json(metadata_path)
        except Exception as exc:
            items.append(
                {
                    "model_family": model_family,
                    "variant": variant,
                    "path_model": str(model_path),
                    "path_metadata": str(metadata_path),
                    "metadata": None,
                    "notes": [f"invalid_metadata_json: {exc}"],
                }
            )
            continue

        items.append(
            {
                "model_family": model_family,
                "variant": variant,
                "path_model": str(model_path),
                "path_metadata": str(metadata_path),
                "metadata": metadata,
                "notes": [],
            }
        )
    return items


def extract_holdout_metrics(
    meta: dict[str, Any],
    threshold_preference: tuple[float, ...] = (0.30, 0.5),
) -> dict[str, Any]:
    """Extract holdout aggregate metrics preferring threshold 0.30 then 0.5."""
    notes: list[str] = []
    preference = tuple(float(value) for value in threshold_preference)
    if not preference:
        preference = (0.30, 0.5)
    preferred_threshold = float(preference[0])

    for threshold in preference:
        candidates_keys: list[str] = [f"evaluation_holdout_at_{_threshold_label(threshold)}"]
        if abs(float(threshold) - 0.5) < 1e-9:
            # Legacy block name.
            candidates_keys.append("evaluation_holdout")
        for key in candidates_keys:
            block = meta.get(key)
            parsed = _extract_from_eval_block(block if isinstance(block, dict) else None, threshold=threshold)
            if not _is_complete_holdout_metrics(parsed):
                continue
            if abs(float(threshold) - preferred_threshold) > 1e-9:
                notes.append(
                    f"fallback_to_threshold_{_threshold_label(threshold)}_used; preferred_threshold_{_threshold_label(preferred_threshold)}_missing_or_incomplete"
                )
            return {
                "available": True,
                "threshold_used": float(threshold),
                "metrics": parsed,
                "notes": notes,
            }

    # Legacy deep fallback in class_imbalance_strategy evidence.
    variant = str(meta.get("variant", "")).strip()
    evidence = meta.get("class_imbalance_strategy")
    if isinstance(evidence, dict):
        by_variant = evidence.get("evidence", {})
        if isinstance(by_variant, dict):
            threshold_05 = by_variant.get("by_variant_threshold_0.5")
            if isinstance(threshold_05, dict):
                variant_block = threshold_05.get(variant)
                if isinstance(variant_block, dict):
                    holdout_raw = variant_block.get("holdout")
                    parsed_legacy = _extract_from_metrics_block(
                        holdout_raw if isinstance(holdout_raw, dict) else None
                    )
                    if _is_complete_holdout_metrics(parsed_legacy):
                        notes.append(
                            "fallback_to_threshold_0.5_used; sourced_from_class_imbalance_strategy_evidence"
                        )
                        return {
                            "available": True,
                            "threshold_used": 0.5,
                            "metrics": parsed_legacy,
                            "notes": notes,
                        }

    notes.append("holdout_metrics_unavailable")
    return {
        "available": False,
        "threshold_used": None,
        "metrics": None,
        "notes": notes,
    }


def select_best_model(
    candidates: list[dict[str, Any]],
    min_recall: float = 0.45,
    min_pr_auc: float = 0.60,
    fail_on_missing_holdout: bool = False,
) -> dict[str, Any]:
    """Select champion model with deterministic holdout ranking and gates."""
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    notes: list[str] = []
    fallback_threshold_variants: list[str] = []
    missing_holdout_variants: list[str] = []

    if not candidates:
        errors.append("No metadata.json discovered under models root.")

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            str(item.get("model_family", "")),
            str(item.get("variant", "")),
            str(item.get("path_metadata", "")),
        ),
    )

    for candidate in sorted_candidates:
        model_family = str(candidate.get("model_family", ""))
        variant = str(candidate.get("variant", ""))
        metadata = candidate.get("metadata")
        row_notes = list(candidate.get("notes", []))
        metrics_holdout: dict[str, Any] | None = None
        threshold_used: float | None = None
        eligible = False
        passed_gates = False

        if isinstance(metadata, dict):
            extracted = extract_holdout_metrics(metadata)
            row_notes.extend(extracted.get("notes", []))
            metrics_holdout = extracted.get("metrics") if isinstance(extracted.get("metrics"), dict) else None
            threshold_used_raw = extracted.get("threshold_used")
            threshold_used = _to_float_or_none(threshold_used_raw)
            eligible = bool(extracted.get("available")) and _is_complete_holdout_metrics(metrics_holdout)
        else:
            row_notes.append("metadata_missing_or_invalid")

        if not eligible:
            missing_holdout_variants.append(f"{model_family}/{variant}")
        else:
            if threshold_used is not None and abs(threshold_used - 0.30) > 1e-9:
                fallback_threshold_variants.append(f"{model_family}/{variant}")

            recall = float(metrics_holdout["recall"])  # type: ignore[index]
            pr_auc = float(metrics_holdout["pr_auc"])  # type: ignore[index]
            passed_gates = recall >= float(min_recall) and pr_auc >= float(min_pr_auc)
            if not passed_gates:
                gate_fail_notes: list[str] = []
                if recall < float(min_recall):
                    gate_fail_notes.append(
                        f"failed_gate_recall<{float(min_recall):.2f} (actual={recall:.4f})"
                    )
                if pr_auc < float(min_pr_auc):
                    gate_fail_notes.append(
                        f"failed_gate_pr_auc<{float(min_pr_auc):.2f} (actual={pr_auc:.4f})"
                    )
                row_notes.extend(gate_fail_notes)

        rows.append(
            {
                "model_family": model_family,
                "variant": variant,
                "path_model": str(candidate.get("path_model", "")),
                "path_metadata": str(candidate.get("path_metadata", "")),
                "eligible": bool(eligible),
                "passed_gates": bool(passed_gates),
                "threshold_used": threshold_used,
                "metrics_holdout": metrics_holdout,
                "notes": row_notes,
            }
        )

    if missing_holdout_variants:
        warnings.append(
            "Candidates excluded from ranking due to missing holdout metrics: "
            f"{sorted(missing_holdout_variants)}"
        )
        if fail_on_missing_holdout:
            errors.append(
                "fail_on_missing_holdout enabled and some candidates are missing holdout metrics."
            )

    if fallback_threshold_variants:
        warnings.append(
            "Fallback threshold 0.5 used because holdout@0.30 is unavailable on: "
            f"{sorted(fallback_threshold_variants)}"
        )

    eligible_rows = [row for row in rows if row["eligible"] and isinstance(row["metrics_holdout"], dict)]
    ranked_eligible = sorted(
        eligible_rows,
        key=lambda row: (
            -float(row["metrics_holdout"]["recall"]),  # type: ignore[index]
            -float(row["metrics_holdout"]["pr_auc"]),  # type: ignore[index]
            float(row["metrics_holdout"]["positive_rate"]),  # type: ignore[index]
            str(row["model_family"]),
            str(row["variant"]),
        ),
    )
    rank_by_key: dict[tuple[str, str], int] = {}
    for rank, row in enumerate(ranked_eligible, start=1):
        rank_by_key[(str(row["model_family"]), str(row["variant"]))] = int(rank)
    for row in rows:
        row["rank"] = rank_by_key.get((str(row["model_family"]), str(row["variant"])))

    passed_rows = [row for row in ranked_eligible if row["passed_gates"]]
    winner_row: dict[str, Any] | None = None
    if passed_rows:
        winner_row = passed_rows[0]
    elif ranked_eligible:
        winner_row = ranked_eligible[0]
        warnings.append(
            "No candidate passed holdout gates; selected highest recall candidate with WARNING."
        )
        notes.append(
            "Fallback winner selected by highest recall because no candidate met both minimum gates."
        )
    else:
        errors.append("No eligible candidate with complete holdout metrics available for ranking.")

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            0 if row.get("rank") is not None else 1,
            int(row["rank"]) if row.get("rank") is not None else 999999,
            str(row["model_family"]),
            str(row["variant"]),
        ),
    )

    winner_payload: dict[str, Any] | None = None
    if winner_row is not None:
        winner_payload = {
            "model_family": winner_row["model_family"],
            "variant": winner_row["variant"],
            "path_model": winner_row["path_model"],
            "path_metadata": winner_row["path_metadata"],
            "metrics_holdout": winner_row["metrics_holdout"],
        }

    status = "PASS"
    if errors:
        status = "FAIL"
    elif warnings:
        status = "WARNING"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selection_criteria": {
            "threshold_used": 0.30,
            "min_recall_holdout": float(min_recall),
            "min_pr_auc_holdout": float(min_pr_auc),
            "ranking_order": list(_RANKING_ORDER),
        },
        "winner": winner_payload,
        "ranked_candidates": ordered_rows,
        "notes": notes,
        "warnings": warnings,
        "errors": errors,
    }

    forbidden_present = find_forbidden_json_keys(report)
    if forbidden_present:
        report["status"] = "FAIL"
        report["errors"].append(
            f"Privacy check failed: forbidden keys found in report: {forbidden_present}"
        )
    return report


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Model Selection")
    lines.append("")
    lines.append(f"Status: **{report.get('status', 'UNKNOWN')}**")
    lines.append("")

    criteria = report.get("selection_criteria", {})
    lines.append("## Criteria")
    lines.append("")
    lines.append(
        "- Threshold: holdout@{:.2f}".format(float(criteria.get("threshold_used", 0.30)))
    )
    lines.append(
        "- Gates: recall >= {:.2f}, pr_auc >= {:.2f}".format(
            float(criteria.get("min_recall_holdout", 0.45)),
            float(criteria.get("min_pr_auc_holdout", 0.60)),
        )
    )
    lines.append(
        "- Ranking: {}".format(", ".join(criteria.get("ranking_order", [])))
    )
    lines.append("")

    winner = report.get("winner")
    if isinstance(winner, dict):
        lines.append("## Winner")
        lines.append("")
        lines.append(
            f"- `{winner.get('model_family')}/{winner.get('variant')}`"
        )
        metrics = winner.get("metrics_holdout", {})
        if isinstance(metrics, dict):
            lines.append(
                "- Holdout metrics: recall={:.4f}, pr_auc={:.4f}, positive_rate={:.4f}".format(
                    float(metrics.get("recall", 0.0)),
                    float(metrics.get("pr_auc", 0.0)),
                    float(metrics.get("positive_rate", 0.0)),
                )
            )
    else:
        lines.append("No winner selected.")

    ranked = report.get("ranked_candidates", [])
    if isinstance(ranked, list) and ranked:
        lines.append("")
        lines.append("## Ranked Candidates")
        lines.append("")
        lines.append(
            "| Rank | Model Family | Variant | Eligible | Passed gates | Recall | PR-AUC | Positive rate | Threshold |"
        )
        lines.append("|---:|---|---|---|---|---:|---:|---:|---:|")
        for row in ranked:
            metrics = row.get("metrics_holdout", {}) if isinstance(row.get("metrics_holdout"), dict) else {}
            rank_value = row.get("rank")
            rank_text = "-" if rank_value is None else str(rank_value)
            recall = metrics.get("recall")
            pr_auc = metrics.get("pr_auc")
            pos_rate = metrics.get("positive_rate")
            threshold_used = row.get("threshold_used")
            lines.append(
                "| {rank} | {family} | {variant} | {eligible} | {passed} | {recall} | {pr_auc} | {pos_rate} | {threshold} |".format(
                    rank=rank_text,
                    family=row.get("model_family"),
                    variant=row.get("variant"),
                    eligible=str(bool(row.get("eligible"))),
                    passed=str(bool(row.get("passed_gates"))),
                    recall=(
                        f"{float(recall):.4f}" if recall is not None else "-"
                    ),
                    pr_auc=(
                        f"{float(pr_auc):.4f}" if pr_auc is not None else "-"
                    ),
                    pos_rate=(
                        f"{float(pos_rate):.4f}" if pos_rate is not None else "-"
                    ),
                    threshold=(
                        f"{float(threshold_used):.2f}" if threshold_used is not None else "-"
                    ),
                )
            )

    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")

    errors = report.get("errors", [])
    if isinstance(errors, list) and errors:
        lines.append("")
        lines.append("## Errors")
        for error in errors:
            lines.append(f"- {error}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_model_selection(
    *,
    models_root: str | Path = "artifacts/models",
    output_json: str | Path = "artifacts/model_selection.json",
    output_md: str | Path = "artifacts/model_selection.md",
    write_markdown: bool = True,
    min_recall_holdout: float = 0.45,
    min_pr_auc_holdout: float = 0.60,
    fail_on_missing_holdout: bool = False,
) -> dict[str, Any]:
    discovered = discover_model_metadatas(models_root=models_root)
    report = select_best_model(
        candidates=discovered,
        min_recall=float(min_recall_holdout),
        min_pr_auc=float(min_pr_auc_holdout),
        fail_on_missing_holdout=bool(fail_on_missing_holdout),
    )

    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if write_markdown:
        write_markdown_report(report, Path(output_md))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select champion model using deterministic holdout criteria."
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="artifacts/models",
        help="Root directory with model metadata.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/model_selection.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="artifacts/model_selection.md",
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable Markdown output.",
    )
    parser.add_argument(
        "--min-recall-holdout",
        type=float,
        default=0.45,
        help="Minimum recall gate on holdout.",
    )
    parser.add_argument(
        "--min-prauc-holdout",
        type=float,
        default=0.60,
        help="Minimum PR-AUC gate on holdout.",
    )
    parser.add_argument(
        "--fail-on-missing-holdout",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, fail when any discovered candidate has missing holdout metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    report = run_model_selection(
        models_root=args.models_root,
        output_json=args.output_json,
        output_md=args.output_md,
        write_markdown=not bool(args.no_markdown),
        min_recall_holdout=float(args.min_recall_holdout),
        min_pr_auc_holdout=float(args.min_prauc_holdout),
        fail_on_missing_holdout=bool(int(args.fail_on_missing_holdout)),
    )
    _logger.info(
        "Model selection generated | status=%s winner=%s",
        report.get("status"),
        report.get("winner"),
    )
    if report.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
