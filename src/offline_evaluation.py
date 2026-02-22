"""Offline post-facto evaluation for production measurement with ground-truth delay."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import get_default_dataset_path, load_pede_workbook_with_metadata, make_temporal_pairs
from src.dataset_versioning import get_dataset_fingerprint, persist_dataset_version_event
from src.metrics import compute_classification_metrics_at_threshold, summarize_proba
from src.preprocessing import get_expected_raw_feature_columns
from src.serving_context import extract_model_identity, extract_operational_threshold, load_serving_metadata
from src.training_utils import build_raw_from_ids
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_FORBIDDEN_KEYS = {"ra", "ra_list", "ids", "students", "student_ids", "records"}
_PROMOTED_MISSING_MESSAGE = "Promoted serving model/metadata not found. Run src.promote_model first."


def _require_eval_dependencies() -> dict[str, Any]:
    try:
        import joblib
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "joblib is required for offline evaluation. Install requirements-dev.txt"
        ) from exc
    return {"joblib": joblib}


def _resolve_dataset_path(dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        return get_default_dataset_path()
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return path


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


def _extract_expected_raw_cols_from_model(model: Any) -> list[str]:
    try:
        raw_step = model.named_steps["raw_to_model"]
        raw_cols = list(
            getattr(raw_step, "expected_raw_cols_", getattr(raw_step, "expected_raw_cols", []))
        )
    except Exception:
        raw_cols = []
    cleaned = [str(col).strip() for col in raw_cols if str(col).strip()]
    if cleaned:
        return cleaned
    return get_expected_raw_feature_columns()


def _resolve_expected_raw_cols(metadata: dict[str, Any], model: Any) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    raw_cols = metadata.get("expected_raw_cols")
    if isinstance(raw_cols, list):
        cleaned = [str(col).strip() for col in raw_cols if str(col).strip()]
        if cleaned:
            return cleaned, notes + ["expected_raw_cols_from_metadata"]
    notes.append("expected_raw_cols_fallback_from_model")
    return _extract_expected_raw_cols_from_model(model), notes


def _resolve_model_artifacts(model_dir: str | Path) -> dict[str, Path]:
    base = Path(model_dir)
    return {
        "model_dir": base,
        "model_path": base / "model.joblib",
        "metadata_path": base / "metadata.json",
    }


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def run_offline_evaluation(
    *,
    dataset_path: str | Path | None = None,
    model_dir: str | Path = "app/model",
    year_t: int = 2023,
    year_t1: int = 2024,
    out_json: str | Path = "artifacts/offline_metrics_2023_2024.json",
    out_md: str | Path = "artifacts/offline_metrics_2023_2024.md",
    write_markdown: bool = True,
) -> dict[str, Any]:
    deps = _require_eval_dependencies()
    artifacts = _resolve_model_artifacts(model_dir)
    model_path = artifacts["model_path"]
    metadata_path = artifacts["metadata_path"]

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"{_PROMOTED_MISSING_MESSAGE} model={model_path.exists()} metadata={metadata_path.exists()}"
        )

    model = deps["joblib"].load(model_path)
    metadata = load_serving_metadata(metadata_path)
    threshold, threshold_notes = extract_operational_threshold(metadata)
    identity, identity_notes = extract_model_identity(metadata)
    expected_raw_cols, expected_notes = _resolve_expected_raw_cols(metadata, model)

    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    dataset_fingerprint = get_dataset_fingerprint(resolved_dataset_path)
    persist_dataset_version_event(
        context="offline_evaluation",
        dataset_fingerprint=dataset_fingerprint,
    )

    yearly_frames, _, _ = load_pede_workbook_with_metadata(file_path=resolved_dataset_path)
    if int(year_t) not in yearly_frames or int(year_t1) not in yearly_frames:
        raise ValueError(
            f"Years not available in dataset for offline evaluation: {year_t}->{year_t1}"
        )

    _, y_true, ids = make_temporal_pairs(
        yearly_frames[int(year_t)],
        yearly_frames[int(year_t1)],
        int(year_t),
        int(year_t1),
    )
    X_raw = build_raw_from_ids(
        df_year_t=yearly_frames[int(year_t)],
        ids=ids,
        expected_raw_cols=expected_raw_cols,
    )

    if not hasattr(model, "predict_proba"):
        raise ValueError("Serving model does not expose predict_proba.")
    proba_matrix = np.asarray(model.predict_proba(X_raw), dtype=float)
    if proba_matrix.ndim != 2 or proba_matrix.shape[1] < 2:
        raise ValueError("model predict_proba output must have shape (n, 2)")
    scores = np.asarray(proba_matrix[:, 1], dtype=float)

    metrics_payload = compute_classification_metrics_at_threshold(
        y_true=y_true,
        y_proba=scores,
        threshold=float(threshold),
    )
    score_summary = summarize_proba(scores)

    notes: list[str] = [
        "offline_replay_ground_truth_delay",
        "replay_local_without_prediction_ids",
        "no_pii_logs_online_offline_uses_dataset_replay",
    ]
    notes.extend(threshold_notes)
    notes.extend(identity_notes)
    notes.extend(expected_notes)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "evaluation_kind": "offline_ground_truth_delay_replay",
        "pair": {"year_t": int(year_t), "year_t1": int(year_t1)},
        "model": {
            "dir": str(artifacts["model_dir"]),
            "model_path_basename": model_path.name,
            "metadata_path_basename": metadata_path.name,
            "model_version": str(identity["model_version"]),
            "model_family": str(identity["model_family"]),
            "variant": str(identity["variant"]),
        },
        "dataset": dataset_fingerprint,
        "contract": {
            "expected_raw_cols_count": int(len(expected_raw_cols)),
        },
        "threshold_operational": float(threshold),
        "pred_proba_summary": score_summary,
        "metrics_at_operational_threshold": metrics_payload,
        "notes": sorted(set(str(note) for note in notes if str(note).strip())),
    }

    forbidden_present = _FORBIDDEN_KEYS & _collect_keys(report)
    if forbidden_present:
        report["status"] = "FAIL"
        report.setdefault("errors", [])
        report["errors"].append(
            f"Privacy check failed: forbidden keys found: {sorted(forbidden_present)}"
        )

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if write_markdown:
        m = report["metrics_at_operational_threshold"]
        lines = [
            "# Offline Metrics (Ground Truth Delay Replay)",
            "",
            f"Status: **{report['status']}**",
            "",
            f"Pair: `{int(year_t)}->{int(year_t1)}`",
            f"Model: `{report['model']['model_family']}/{report['model']['variant']}`",
            f"Model version: `{report['model']['model_version']}`",
            f"Threshold operacional: `{float(report['threshold_operational']):.4f}`",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Recall | {_fmt_metric(m.get('recall'))} |",
            f"| PR-AUC | {_fmt_metric(m.get('pr_auc'))} |",
            f"| Precision | {_fmt_metric(m.get('precision'))} |",
            f"| F1 | {_fmt_metric(m.get('f1'))} |",
            f"| ROC-AUC | {_fmt_metric(m.get('roc_auc'))} |",
            f"| Positive rate @ threshold | {_fmt_metric(m.get('positive_rate'))} |",
            f"| n | {int(m.get('n', 0))} |",
            f"| n_pos | {int(m.get('n_pos', 0))} |",
            f"| prevalence | {_fmt_metric(m.get('prevalence'))} |",
            "",
            "## Confusion Matrix",
            "",
        ]
        cm = m.get("confusion_matrix", {}) or {}
        lines.extend(
            [
                f"- TN: `{int(cm.get('tn', 0))}`",
                f"- FP: `{int(cm.get('fp', 0))}`",
                f"- FN: `{int(cm.get('fn', 0))}`",
                f"- TP: `{int(cm.get('tp', 0))}`",
                "",
                "## Notes",
            ]
        )
        lines.extend([f"- {note}" for note in report.get("notes", [])])
        out_md_path = Path(out_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline post-facto evaluation (ground truth delay replay) for serving model."
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="PEDE XLSX dataset path.")
    parser.add_argument("--model-dir", type=str, default="app/model", help="Serving model directory.")
    parser.add_argument("--year-t", type=int, default=2023, help="Feature year (t).")
    parser.add_argument("--year-t1", type=int, default=2024, help="Label year (t+1).")
    parser.add_argument(
        "--out-json",
        type=str,
        default="artifacts/offline_metrics_2023_2024.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="artifacts/offline_metrics_2023_2024.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable Markdown report generation.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    run_offline_evaluation(
        dataset_path=args.dataset_path,
        model_dir=args.model_dir,
        year_t=int(args.year_t),
        year_t1=int(args.year_t1),
        out_json=args.out_json,
        out_md=args.out_md,
        write_markdown=not bool(args.no_markdown),
    )


if __name__ == "__main__":
    main()

