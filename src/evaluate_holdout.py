"""Evaluate saved model artifacts on official holdout pair (2023->2024)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import (
    get_default_dataset_path,
    load_pede_workbook_with_metadata,
    make_temporal_pairs,
)
from src.dataset_versioning import (
    get_dataset_fingerprint,
    persist_dataset_version_event,
)
from src.metrics import compute_metrics_threshold, compute_prevalence
from src.preprocessing import get_expected_raw_feature_columns
from src.training_policy import OFFICIAL_HOLDOUT_PAIR
from src.training_utils import build_raw_from_ids
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_FORBIDDEN_KEYS = {"ra", "ra_list", "ids", "students", "student_ids", "records"}


def _require_eval_dependencies() -> dict[str, Any]:
    try:
        import joblib
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "joblib is required to evaluate holdout artifacts. Install requirements-dev.txt"
        ) from exc
    return {"joblib": joblib}


def _resolve_dataset_path(dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        return get_default_dataset_path()
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return path


def _discover_model_artifacts(models_root: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if not models_root.exists():
        return items
    for model_path in sorted(models_root.rglob("model.joblib")):
        rel = model_path.relative_to(models_root)
        if len(rel.parts) < 3:
            continue
        items.append(
            {
                "model_family": rel.parts[0],
                "variant": rel.parts[1],
                "model_path": str(model_path),
            }
        )
    return items


def _probability_summary(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(scores)),
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
        "p05": float(np.quantile(scores, 0.05)),
        "p50": float(np.quantile(scores, 0.50)),
        "p95": float(np.quantile(scores, 0.95)),
    }


def _extract_expected_raw_cols(model: Any) -> list[str]:
    raw_cols: list[str] | None = None
    try:
        raw_step = model.named_steps["raw_to_model"]
        raw_cols = list(
            getattr(raw_step, "expected_raw_cols_", getattr(raw_step, "expected_raw_cols", []))
        )
    except Exception:
        raw_cols = None

    if raw_cols:
        return [str(col).strip() for col in raw_cols if str(col).strip()]
    return get_expected_raw_feature_columns()


def _collect_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys |= _collect_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_keys(item)
    return keys


def run_holdout_evaluation(
    *,
    models_root: str | Path = "artifacts/models",
    dataset_path: str | Path | None = None,
    output_json: str | Path = "artifacts/holdout_evaluation.json",
    output_md: str | Path = "artifacts/holdout_evaluation.md",
    write_markdown: bool = True,
) -> dict[str, Any]:
    deps = _require_eval_dependencies()
    models_root_path = Path(models_root)
    discovered = _discover_model_artifacts(models_root_path)

    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    if not discovered:
        errors.append(f"No model.joblib files found under: {models_root_path}")

    resolved_dataset_path = _resolve_dataset_path(dataset_path) if discovered else None
    dataset_fingerprint: dict[str, Any] | None = None
    yearly_frames: dict[int, pd.DataFrame] = {}
    if resolved_dataset_path is not None:
        dataset_fingerprint = get_dataset_fingerprint(resolved_dataset_path)
        persist_dataset_version_event(
            context="evaluate_holdout",
            dataset_fingerprint=dataset_fingerprint,
        )
        yearly_frames, _, _ = load_pede_workbook_with_metadata(file_path=resolved_dataset_path)
        if OFFICIAL_HOLDOUT_PAIR[0] not in yearly_frames or OFFICIAL_HOLDOUT_PAIR[1] not in yearly_frames:
            errors.append(
                f"Holdout years missing in dataset: expected {OFFICIAL_HOLDOUT_PAIR[0]} and {OFFICIAL_HOLDOUT_PAIR[1]}"
            )

    y_holdout: pd.Series | None = None
    ids_holdout: pd.Series | None = None
    if not errors and yearly_frames:
        _, y_holdout, ids_holdout = make_temporal_pairs(
            yearly_frames[OFFICIAL_HOLDOUT_PAIR[0]],
            yearly_frames[OFFICIAL_HOLDOUT_PAIR[1]],
            OFFICIAL_HOLDOUT_PAIR[0],
            OFFICIAL_HOLDOUT_PAIR[1],
        )

    for item in discovered:
        model_family = item["model_family"]
        variant = item["variant"]
        model_path = Path(item["model_path"])
        try:
            model = deps["joblib"].load(model_path)
            expected_raw_cols = _extract_expected_raw_cols(model)
            if ids_holdout is None or y_holdout is None:
                raise RuntimeError("Holdout cohort unavailable for evaluation.")
            X_raw_holdout = build_raw_from_ids(
                yearly_frames[OFFICIAL_HOLDOUT_PAIR[0]],
                ids_holdout,
                expected_raw_cols,
            )
            scores = model.predict_proba(X_raw_holdout)[:, 1]
            prevalence = compute_prevalence(y_holdout)
            metrics = compute_metrics_threshold(y_holdout, scores, threshold=0.5)
            rows.append(
                {
                    "model_family": model_family,
                    "variant": variant,
                    "model_path": str(model_path),
                    "pair": f"{OFFICIAL_HOLDOUT_PAIR[0]}->{OFFICIAL_HOLDOUT_PAIR[1]}",
                    "n": int(prevalence["n"]),
                    "n_pos": int(prevalence["n_pos"]),
                    "prevalence": float(prevalence["prevalence"]),
                    "threshold": 0.5,
                    "metrics": metrics,
                    "pred_proba_summary": _probability_summary(scores),
                    "notes": [],
                }
            )
        except Exception as exc:
            errors.append(
                f"Failed holdout evaluation for {model_family}/{variant}: {exc}"
            )

    status = "PASS"
    if errors:
        status = "FAIL"
    elif warnings:
        status = "WARNING"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_root": str(models_root_path),
        "dataset_path": None if resolved_dataset_path is None else str(resolved_dataset_path),
        "dataset": dataset_fingerprint,
        "pair": f"{OFFICIAL_HOLDOUT_PAIR[0]}->{OFFICIAL_HOLDOUT_PAIR[1]}",
        "rows": rows,
        "status": status,
        "errors": errors,
        "warnings": warnings,
    }

    keys_found = _collect_keys(report)
    forbidden_present = _FORBIDDEN_KEYS & keys_found
    if forbidden_present:
        report["status"] = "FAIL"
        report["errors"].append(
            f"Privacy check failed: forbidden keys found: {sorted(forbidden_present)}"
        )

    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if write_markdown:
        lines = [
            "# Holdout Evaluation",
            "",
            f"Pair: `{report['pair']}`",
            "",
            f"Status: **{report['status']}**",
            "",
        ]
        if rows:
            lines.append("| Model Family | Variant | Recall@0.5 | PR-AUC | Precision@0.5 |")
            lines.append("|---|---|---:|---:|---:|")
            for row in rows:
                metrics = row["metrics"]
                lines.append(
                    f"| {row['model_family']} | {row['variant']} | "
                    f"{float(metrics['recall']):.4f} | {float(metrics['pr_auc']):.4f} | {float(metrics['precision']):.4f} |"
                )
        else:
            lines.append("No evaluated models.")

        if report["warnings"]:
            lines.extend(["", "## Warnings"])
            lines.extend([f"- {warning}" for warning in report["warnings"]])
        if report["errors"]:
            lines.extend(["", "## Errors"])
            lines.extend([f"- {error}" for error in report["errors"]])
        output_md_path = Path(output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved model artifacts on official holdout pair (2023->2024)."
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="artifacts/models",
        help="Root folder with trained model artifacts.",
    )
    parser.add_argument(
        "--dataset-path",
        "--file-path",
        dest="dataset_path",
        type=str,
        default=None,
        help="Path to XLSX dataset. Defaults to DATASET_PATH env / project default.",
    )
    parser.add_argument(
        "--output",
        "--out-json",
        dest="output_json",
        type=str,
        default="artifacts/holdout_evaluation.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="artifacts/holdout_evaluation.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable Markdown report generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    report = run_holdout_evaluation(
        models_root=args.models_root,
        dataset_path=args.dataset_path,
        output_json=args.output_json,
        output_md=args.out_md,
        write_markdown=not bool(args.no_markdown),
    )
    _logger.info(
        "Holdout evaluation finished | status=%s rows=%d",
        report.get("status"),
        len(report.get("rows", [])),
    )
    if report.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
