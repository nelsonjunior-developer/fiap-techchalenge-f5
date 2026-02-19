"""Model comparison CLI based on training metadata artifacts (no retraining)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.training_policy import OFFICIAL_HOLDOUT_PAIR
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_FORBIDDEN_KEYS = {"ra", "ra_list", "ids", "student_ids", "students", "records"}


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid metadata format (expected object): {path}")
    return payload


def discover_metadata_files(models_root: Path) -> list[dict[str, str]]:
    """Discover metadata files under artifacts/models/<family>/<variant>/metadata.json."""
    items: list[dict[str, str]] = []
    if not models_root.exists():
        return items
    for path in sorted(models_root.rglob("metadata.json")):
        rel = path.relative_to(models_root)
        if len(rel.parts) < 3:
            continue
        family = rel.parts[0]
        variant = rel.parts[1]
        items.append(
            {
                "model_family": str(family),
                "variant": str(variant),
                "path": str(path),
            }
        )
    return items


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_metrics_block(raw_block: dict[str, Any] | None) -> dict[str, float | None] | None:
    if not isinstance(raw_block, dict):
        return None
    positive_rate = raw_block.get("positive_rate_at_threshold")
    if positive_rate is None:
        positive_rate = raw_block.get("positive_rate_at_0.5")
    if positive_rate is None:
        positive_rate = raw_block.get("positive_rate")
    return {
        "recall_at_0.5": _to_float_or_none(raw_block.get("recall")),
        "precision_at_0.5": _to_float_or_none(raw_block.get("precision")),
        "f1_at_0.5": _to_float_or_none(raw_block.get("f1")),
        "roc_auc": _to_float_or_none(raw_block.get("roc_auc")),
        "pr_auc": _to_float_or_none(raw_block.get("pr_auc")),
        "positive_rate_at_0.5": _to_float_or_none(positive_rate),
    }


def _extract_train_metrics(metadata: dict[str, Any]) -> tuple[dict[str, float | None] | None, list[str]]:
    warnings: list[str] = []
    evaluation_train = metadata.get("evaluation_train")
    if isinstance(evaluation_train, dict):
        normalized = _normalize_metrics_block(
            evaluation_train.get("metrics")
            if isinstance(evaluation_train.get("metrics"), dict)
            else None
        )
        if normalized is not None:
            return normalized, warnings

    direct = _normalize_metrics_block(
        metadata.get("metrics_train_at_0.5")
        if isinstance(metadata.get("metrics_train_at_0.5"), dict)
        else None
    )
    if direct is not None:
        return direct, warnings

    warnings.append("metrics_train_at_0.5/evaluation_train.metrics missing in metadata.")
    return None, warnings


def _extract_holdout_metrics(
    metadata: dict[str, Any],
    *,
    variant: str,
) -> tuple[dict[str, float | None] | None, list[str]]:
    warnings: list[str] = []

    direct = _normalize_metrics_block(
        metadata.get("metrics_holdout_at_0.5")
        if isinstance(metadata.get("metrics_holdout_at_0.5"), dict)
        else None
    )
    if direct is not None:
        return direct, warnings

    evaluation_holdout = metadata.get("evaluation_holdout")
    if isinstance(evaluation_holdout, dict):
        eval_metrics = _normalize_metrics_block(
            evaluation_holdout.get("metrics")
            if isinstance(evaluation_holdout.get("metrics"), dict)
            else None
        )
        if eval_metrics is not None:
            return eval_metrics, warnings

    evidence = (
        metadata.get("class_imbalance_strategy", {})
        if isinstance(metadata.get("class_imbalance_strategy"), dict)
        else {}
    )
    by_variant = (
        evidence.get("evidence", {}).get("by_variant_threshold_0.5", {})
        if isinstance(evidence.get("evidence", {}), dict)
        else {}
    )
    if isinstance(by_variant, dict):
        variant_block = by_variant.get(variant)
        if isinstance(variant_block, dict):
            holdout_block = variant_block.get("holdout")
            normalized = _normalize_metrics_block(holdout_block)
            if normalized is not None:
                return normalized, warnings

    warnings.append(
        f"Missing holdout metrics for variant={variant}; expected evaluation_holdout.metrics or metrics_holdout_at_0.5 or class_imbalance_strategy.evidence.by_variant_threshold_0.5.{variant}.holdout"
    )
    return None, warnings


def _extract_hyperparams(
    metadata: dict[str, Any],
    *,
    model_family: str,
    notes: list[str],
) -> dict[str, Any]:
    if model_family == "baseline_logreg":
        hyper = {
            "class_weight": metadata.get("class_weight"),
            "C": metadata.get("C"),
            "penalty": metadata.get("penalty"),
            "solver": metadata.get("solver"),
            "max_iter": metadata.get("max_iter"),
        }
        missing = [key for key, value in hyper.items() if value is None and key != "class_weight"]
        if missing:
            notes.append(
                "Baseline hyperparameters missing in metadata: "
                f"{missing}. Stored as null."
            )
        return hyper

    if model_family == "nonlinear_hgb":
        resolved = metadata.get("resolved_params")
        if isinstance(resolved, dict):
            return dict(resolved)
        notes.append("HGB resolved_params missing in metadata.")
        return {}

    return {}


def build_comparison_report(
    *,
    models_root: Path,
    fail_on_missing_holdout: bool = False,
) -> dict[str, Any]:
    discovered = discover_metadata_files(models_root)
    warnings: list[str] = []
    errors: list[str] = []

    if not discovered:
        errors.append(f"No metadata.json files found under: {models_root}")

    inputs_discovered: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []

    for item in discovered:
        model_family = item["model_family"]
        variant = item["variant"]
        metadata_path = Path(item["path"])
        inputs_discovered.setdefault(model_family, [])
        if variant not in inputs_discovered[model_family]:
            inputs_discovered[model_family].append(variant)

        row_notes: list[str] = []
        try:
            metadata = _safe_read_json(metadata_path)
        except Exception as exc:
            errors.append(f"Failed to parse metadata: {metadata_path} | error={exc}")
            continue

        train_pair_raw = metadata.get("train_pair", {})
        if isinstance(train_pair_raw, dict):
            year_t = train_pair_raw.get("year_t")
            year_t1 = train_pair_raw.get("year_t1")
            train_pair = (
                f"{year_t}->{year_t1}"
                if year_t is not None and year_t1 is not None
                else None
            )
        else:
            train_pair = None
            row_notes.append("train_pair missing or invalid in metadata.")

        holdout_pair = f"{OFFICIAL_HOLDOUT_PAIR[0]}->{OFFICIAL_HOLDOUT_PAIR[1]}"
        train_metrics, train_warnings = _extract_train_metrics(metadata)
        row_notes.extend(train_warnings)
        warnings.extend([f"{model_family}/{variant}: {message}" for message in train_warnings])
        if train_metrics is None:
            train_metrics = {
                "recall_at_0.5": None,
                "precision_at_0.5": None,
                "f1_at_0.5": None,
                "roc_auc": None,
                "pr_auc": None,
                "positive_rate_at_0.5": None,
            }

        holdout_metrics, holdout_warnings = _extract_holdout_metrics(
            metadata,
            variant=variant,
        )
        row_notes.extend(holdout_warnings)
        warnings.extend(
            [
                f"{model_family}/{variant}: {message}"
                for message in holdout_warnings
            ]
        )

        row = {
            "model_family": model_family,
            "variant": variant,
            "train_pair": train_pair,
            "holdout_pair": holdout_pair,
            "metrics": {
                "train": train_metrics,
                "holdout": holdout_metrics,
            },
            "hyperparams": _extract_hyperparams(
                metadata,
                model_family=model_family,
                notes=row_notes,
            ),
            "notes": row_notes,
        }
        rows.append(row)

    # Ranking candidates require full holdout ranking keys.
    candidates: list[dict[str, Any]] = []
    missing_holdout_variants: list[str] = []
    for row in rows:
        holdout = row["metrics"].get("holdout")
        if not isinstance(holdout, dict):
            missing_holdout_variants.append(
                f"{row['model_family']}/{row['variant']}"
            )
            continue
        if (
            holdout.get("recall_at_0.5") is None
            or holdout.get("pr_auc") is None
            or holdout.get("positive_rate_at_0.5") is None
        ):
            missing_holdout_variants.append(
                f"{row['model_family']}/{row['variant']}"
            )
            continue
        candidates.append(row)

    if missing_holdout_variants:
        warnings.append(
            "Missing holdout metrics for ranking on variants: "
            f"{sorted(missing_holdout_variants)}"
        )
        if fail_on_missing_holdout:
            errors.append(
                "fail_on_missing_holdout enabled and some variants have missing holdout metrics."
            )

    ranking_table: list[dict[str, Any]] = []
    winner: dict[str, str] | None = None
    if candidates:
        sorted_candidates = sorted(
            candidates,
            key=lambda row: (
                -float(row["metrics"]["holdout"]["recall_at_0.5"]),
                -float(row["metrics"]["holdout"]["pr_auc"]),
                float(row["metrics"]["holdout"]["positive_rate_at_0.5"]),
                str(row["model_family"]),
                str(row["variant"]),
            ),
        )
        for idx, row in enumerate(sorted_candidates, start=1):
            holdout = row["metrics"]["holdout"]
            ranking_table.append(
                {
                    "rank": int(idx),
                    "model_family": row["model_family"],
                    "variant": row["variant"],
                    "recall_holdout_at_0.5": holdout["recall_at_0.5"],
                    "pr_auc_holdout": holdout["pr_auc"],
                    "positive_rate_holdout_at_0.5": holdout["positive_rate_at_0.5"],
                }
            )
        winner = {
            "model_family": sorted_candidates[0]["model_family"],
            "variant": sorted_candidates[0]["variant"],
        }
    else:
        warnings.append("No model variant had complete holdout metrics for ranking.")
        if fail_on_missing_holdout and not errors:
            errors.append("No ranking candidate available due to missing holdout metrics.")

    status = "PASS"
    if errors:
        status = "FAIL"
    elif warnings:
        status = "WARNING"

    # Privacy safety check for keys in output object.
    def _collect_keys(obj: Any) -> set[str]:
        keys: set[str] = set()
        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.add(str(key).lower())
                keys |= _collect_keys(value)
        elif isinstance(obj, list):
            for item in obj:
                keys |= _collect_keys(item)
        return keys

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_root": str(models_root),
        "policy": {
            "primary": "recall_holdout_at_0.5",
            "secondary": "pr_auc_holdout",
            "tertiary": "positive_rate_holdout_at_0.5",
        },
        "inputs_discovered": {
            family: sorted(variants)
            for family, variants in sorted(inputs_discovered.items(), key=lambda item: item[0])
        },
        "rows": rows,
        "ranking": {
            "sorted_by": [
                "recall_holdout_at_0.5 desc",
                "pr_auc_holdout desc",
                "positive_rate_holdout_at_0.5 asc",
            ],
            "winner": winner,
            "table": ranking_table,
        },
        "status": status,
        "errors": errors,
        "warnings": warnings,
    }
    keys_found = _collect_keys(report)
    forbidden_present = _FORBIDDEN_KEYS & keys_found
    # rows/table are legitimate keys for report shape.
    if forbidden_present:
        report["status"] = "FAIL"
        report["errors"].append(
            f"Privacy check failed: forbidden keys found in report: {sorted(forbidden_present)}"
        )
    return report


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Model Comparison")
    lines.append("")
    lines.append(
        "Ranking policy: Recall holdout@0.5 (desc), PR-AUC holdout (desc), positive rate holdout@0.5 (asc)."
    )
    lines.append("")
    lines.append(f"Status: **{report.get('status', 'UNKNOWN')}**")
    lines.append("")
    table = report.get("ranking", {}).get("table", [])
    if table:
        lines.append(
            "| Rank | Model Family | Variant | Recall holdout@0.5 | PR-AUC holdout | Positive rate holdout@0.5 |"
        )
        lines.append("|---:|---|---|---:|---:|---:|")
        for row in table:
            lines.append(
                "| {rank} | {model_family} | {variant} | {recall:.4f} | {pr_auc:.4f} | {pos_rate:.4f} |".format(
                    rank=row["rank"],
                    model_family=row["model_family"],
                    variant=row["variant"],
                    recall=float(row["recall_holdout_at_0.5"]),
                    pr_auc=float(row["pr_auc_holdout"]),
                    pos_rate=float(row["positive_rate_holdout_at_0.5"]),
                )
            )
        winner = report.get("ranking", {}).get("winner")
        if isinstance(winner, dict):
            lines.append("")
            lines.append(
                f"Winner: **{winner.get('model_family')}/{winner.get('variant')}**"
            )
    else:
        lines.append("No ranking candidates available.")

    warnings = report.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")

    errors = report.get("errors", [])
    if errors:
        lines.append("")
        lines.append("## Errors")
        for error in errors:
            lines.append(f"- {error}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_compare_models(
    *,
    models_root: str | Path = "artifacts/models",
    out_json: str | Path = "artifacts/model_comparison.json",
    out_md: str | Path = "artifacts/model_comparison.md",
    write_markdown: bool = True,
    fail_on_missing_holdout: bool = False,
) -> dict[str, Any]:
    report = build_comparison_report(
        models_root=Path(models_root),
        fail_on_missing_holdout=bool(fail_on_missing_holdout),
    )
    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if write_markdown:
        write_markdown_report(report, Path(out_md))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trained model variants from metadata.json artifacts."
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="artifacts/models",
        help="Root directory containing model family/variant metadata files.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="artifacts/model_comparison.json",
        help="Output JSON comparison report path.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="artifacts/model_comparison.md",
        help="Output Markdown comparison report path.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable markdown report generation.",
    )
    parser.add_argument(
        "--fail-on-missing-holdout",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, fail when any discovered variant has missing holdout metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    report = run_compare_models(
        models_root=args.models_root,
        out_json=args.out_json,
        out_md=args.out_md,
        write_markdown=not bool(args.no_markdown),
        fail_on_missing_holdout=bool(int(args.fail_on_missing_holdout)),
    )
    _logger.info(
        "Model comparison generated | status=%s rows=%d winner=%s",
        report.get("status"),
        len(report.get("rows", [])),
        report.get("ranking", {}).get("winner"),
    )
    if report.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
