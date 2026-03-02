"""Local retraining orchestrator (time + drift + shift triggers)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)

DEFAULT_POLICY = {
    "version": "1.0.0",
    "time_trigger": {"max_age_days": 90},
    "drift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
    "shift_trigger": {"fail_status": "FAIL", "warn_status": "WARNING"},
}


@dataclass
class TriggerOutcome:
    required_reasons: list[str]
    recommended_reasons: list[str]
    notes: list[str]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _now_utc().isoformat()


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_bool_flag(value: int | bool) -> bool:
    if isinstance(value, bool):
        return value
    return bool(int(value))


def _norm_status(value: Any) -> str:
    return str(value or "").strip().upper()


def _safe_rel(path: Path) -> str:
    try:
        return os.path.relpath(path)
    except Exception:
        return str(path)


def _dataset_basename(dataset_path: str | Path | None) -> str | None:
    if dataset_path is None:
        return None
    return Path(dataset_path).name


def _merge_policy(policy_path: Path) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    payload = _safe_read_json(policy_path)
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    if payload is None:
        notes.append("policy_file_missing_or_invalid_using_defaults")
        return policy, notes

    for key in ("version", "notes"):
        if key in payload:
            policy[key] = payload[key]

    for key in ("time_trigger", "drift_trigger", "shift_trigger"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            merged = dict(policy.get(key, {}))
            merged.update(value)
            policy[key] = merged
        elif key in payload:
            notes.append(f"policy_{key}_invalid_using_defaults")

    return policy, notes


def _read_state(
    *,
    metadata_path: Path,
    reference_meta_path: Path,
    drift_summary_path: Path,
    shift_report_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []

    metadata = _safe_read_json(metadata_path)
    if metadata is None:
        notes.append("serving_metadata_missing_or_invalid")
        metadata = {}
    reference_meta = _safe_read_json(reference_meta_path)
    if reference_meta is None:
        notes.append("reference_meta_missing_or_invalid")
        reference_meta = {}
    drift_summary = _safe_read_json(drift_summary_path)
    if drift_summary is None:
        notes.append("drift_summary_missing_or_invalid")
        drift_summary = {}
    shift_report = _safe_read_json(shift_report_path)
    if shift_report is None:
        notes.append("shift_report_missing_or_invalid")
        shift_report = {}

    state = {
        "metadata": metadata,
        "reference_meta": reference_meta,
        "drift_summary": drift_summary,
        "shift_report": shift_report,
        "paths": {
            "metadata_path": _safe_rel(metadata_path),
            "reference_meta_path": _safe_rel(reference_meta_path),
            "drift_summary_path": _safe_rel(drift_summary_path),
            "shift_report_path": _safe_rel(shift_report_path),
        },
    }
    return state, notes


def evaluate_retrain_decision(
    *,
    policy: Mapping[str, Any],
    state: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    now_utc = (now or _now_utc()).astimezone(timezone.utc)
    required_reasons: list[str] = []
    recommended_reasons: list[str] = []
    notes: list[str] = []

    metadata = state.get("metadata")
    metadata_dict = metadata if isinstance(metadata, Mapping) else {}
    drift_summary = state.get("drift_summary")
    drift_dict = drift_summary if isinstance(drift_summary, Mapping) else {}
    shift_report = state.get("shift_report")
    shift_dict = shift_report if isinstance(shift_report, Mapping) else {}

    # Time trigger.
    time_cfg = policy.get("time_trigger")
    time_cfg_dict = time_cfg if isinstance(time_cfg, Mapping) else {}
    max_age_days = _to_int(time_cfg_dict.get("max_age_days"), 90)
    trained_at = _parse_iso_datetime(
        metadata_dict.get("trained_at") or metadata_dict.get("created_at")
    )
    model_age_days: int | None = None
    if trained_at is None:
        notes.append("trained_at_missing_time_trigger_not_evaluated")
    else:
        model_age_days = int((now_utc - trained_at).days)
        if model_age_days >= max_age_days:
            required_reasons.append(
                f"time_trigger(age_days={model_age_days}>=max_age_days={max_age_days})"
            )

    def _apply_status_trigger(
        *,
        source_name: str,
        payload: Mapping[str, Any],
        trigger_key: str,
    ) -> None:
        cfg = policy.get(trigger_key)
        cfg_dict = cfg if isinstance(cfg, Mapping) else {}
        fail_status = _norm_status(cfg_dict.get("fail_status") or "FAIL")
        warn_status = _norm_status(cfg_dict.get("warn_status") or "WARNING")
        status = _norm_status(payload.get("status"))
        if not status:
            notes.append(f"{source_name}_status_missing")
            return
        if status == fail_status:
            required_reasons.append(f"{source_name}_status={status}")
            return
        if status == warn_status:
            recommended_reasons.append(f"{source_name}_status={status}")
            return
        notes.append(f"{source_name}_status_ignored={status}")

    if drift_dict:
        _apply_status_trigger(
            source_name="drift",
            payload=drift_dict,
            trigger_key="drift_trigger",
        )
    else:
        notes.append("drift_summary_unavailable")

    if shift_dict:
        _apply_status_trigger(
            source_name="shift",
            payload=shift_dict,
            trigger_key="shift_trigger",
        )
    else:
        notes.append("shift_report_unavailable")

    if required_reasons:
        decision = "RETRAIN_REQUIRED"
    elif recommended_reasons:
        decision = "RETRAIN_RECOMMENDED"
    else:
        decision = "NOOP"

    return {
        "generated_at": now_utc.isoformat(),
        "decision": decision,
        "required_reasons": required_reasons,
        "recommended_reasons": recommended_reasons,
        "reasons": required_reasons + recommended_reasons,
        "inputs": {
            "model_version": str(metadata_dict.get("model_version") or "unknown"),
            "model_family": str(metadata_dict.get("model_family") or "unknown"),
            "variant": str(metadata_dict.get("variant") or "unknown"),
            "trained_at": (
                trained_at.isoformat() if trained_at is not None else None
            ),
            "model_age_days": model_age_days,
            "max_age_days": max_age_days,
            "drift_status": _norm_status(drift_dict.get("status")) or None,
            "shift_status": _norm_status(shift_dict.get("status")) or None,
            "reference_generated_at": str(
                (
                    state.get("reference_meta", {})
                    if isinstance(state.get("reference_meta"), Mapping)
                    else {}
                ).get("generated_at")
                or ""
            )
            or None,
        },
        "notes": notes,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_dataset_for_execution(dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        raise FileNotFoundError(
            "Dataset path is required when --execute=1. Use --dataset-path or DATASET_PATH."
        )
    resolved = Path(dataset_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path not found: {resolved}")
    return resolved


def _sanitize_command(
    command: list[str],
    *,
    dataset_path: Path | None,
) -> list[str]:
    if dataset_path is None:
        return list(command)
    sanitized: list[str] = []
    dataset_variants = {
        str(dataset_path),
        str(dataset_path.resolve()),
    }
    replacement = dataset_path.name
    for token in command:
        if token in dataset_variants:
            sanitized.append(replacement)
            continue
        if os.path.isabs(token):
            sanitized.append(Path(token).name or token)
            continue
        sanitized.append(token)
    return sanitized


def _run_step(
    *,
    name: str,
    command: list[str],
    logs_dir: Path,
    dataset_path: Path | None,
) -> dict[str, Any]:
    started = _now_utc()
    proc = subprocess.run(command, capture_output=True, text=True)
    finished = _now_utc()
    log_path = logs_dir / f"{name}.log"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (
            f"$ {' '.join(command)}\n\n"
            f"[stdout]\n{proc.stdout}\n"
            f"[stderr]\n{proc.stderr}\n"
            f"[returncode]\n{proc.returncode}\n"
        ),
        encoding="utf-8",
    )
    step = {
        "name": name,
        "command": _sanitize_command(command, dataset_path=dataset_path),
        "returncode": int(proc.returncode),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": float((finished - started).total_seconds()),
        "log_path": _safe_rel(log_path),
    }
    return step


def _build_retrain_steps(
    *,
    python_bin: str,
    dataset_path: Path,
    run_models_root: Path,
    selection_path: Path,
    selection_md: Path,
    stage_dir: Path,
    prod_dir: Path,
    reference_dir: Path,
    allow_warning_promotion: bool,
    current_csv: Path | None,
    drift_out_html: Path,
    drift_out_json: Path,
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    notes: list[str] = []
    baseline_out = run_models_root / "baseline_logreg"
    hgb_out = run_models_root / "nonlinear_hgb"

    steps: list[tuple[str, list[str]]] = [
        (
            "train_baseline",
            [
                python_bin,
                "-m",
                "src.train_baseline",
                "--dataset-path",
                str(dataset_path),
                "--year-t",
                "2022",
                "--year-t1",
                "2023",
                "--out-dir",
                str(baseline_out),
                "--variants",
                "none",
                "--eval-holdout",
                "1",
                "--strict",
                "1",
            ],
        ),
        (
            "train_hgb",
            [
                python_bin,
                "-m",
                "src.train_hgb",
                "--dataset-path",
                str(dataset_path),
                "--year-t",
                "2022",
                "--year-t1",
                "2023",
                "--out-dir",
                str(hgb_out),
                "--variants",
                "default,tuned",
                "--eval-holdout",
                "1",
                "--strict",
                "1",
            ],
        ),
        (
            "model_selection",
            [
                python_bin,
                "-m",
                "src.model_selection",
                "--models-root",
                str(run_models_root),
                "--output-json",
                str(selection_path),
                "--output-md",
                str(selection_md),
            ],
        ),
        (
            "promote_stage",
            [
                python_bin,
                "-m",
                "src.promote_model",
                "--selection-path",
                str(selection_path),
                "--models-root",
                str(run_models_root),
                "--out-dir",
                str(stage_dir),
                "--stage-only",
                "1",
                "--backup",
                "1",
                "--force",
                "1",
                "--allow-warning",
                "1" if allow_warning_promotion else "0",
            ],
        ),
        (
            "promote_prod",
            [
                python_bin,
                "-m",
                "src.promote_model",
                "--selection-path",
                str(selection_path),
                "--models-root",
                str(run_models_root),
                "--from-staging",
                str(stage_dir),
                "--out-dir",
                str(prod_dir),
                "--promote",
                "1",
                "--backup",
                "1",
                "--force",
                "1",
                "--allow-warning",
                "1" if allow_warning_promotion else "0",
            ],
        ),
        (
            "build_reference_data",
            [
                python_bin,
                "-m",
                "src.build_reference_data",
                "--dataset-path",
                str(dataset_path),
                "--model-dir",
                str(prod_dir),
                "--out-dir",
                str(reference_dir),
                "--max-rows",
                "1000",
                "--backup",
                "1",
                "--force",
                "1",
            ],
        ),
    ]

    if current_csv is not None and current_csv.exists():
        steps.append(
            (
                "drift_report_refresh",
                [
                    python_bin,
                    "-m",
                    "src.drift",
                    "--reference-dir",
                    str(reference_dir),
                    "--current-csv",
                    str(current_csv),
                    "--out-html",
                    str(drift_out_html),
                    "--out-json",
                    str(drift_out_json),
                    "--max-rows",
                    "2000",
                    "--seed",
                    "42",
                ],
            )
        )
    elif current_csv is None:
        notes.append("current_csv_not_provided_drift_refresh_skipped")
    else:
        notes.append(f"current_csv_not_found_drift_refresh_skipped={current_csv.name}")

    return steps, notes


def run_retrain_orchestration(
    *,
    dataset_path: str | Path | None = None,
    policy_path: str | Path = "docs/retrain_policy.json",
    models_root: str | Path = "artifacts/models",
    selection_path: str | Path = "artifacts/model_selection.json",
    stage_dir: str | Path = "app/model/staging",
    prod_dir: str | Path = "app/model",
    reference_dir: str | Path = "app/model/reference",
    drift_summary_path: str | Path = "artifacts/drift_report_summary.json",
    shift_report_path: str | Path = "artifacts/temporal_shift_report.json",
    metadata_path: str | Path = "app/model/metadata.json",
    reference_meta_path: str | Path = "app/model/reference/reference_meta.json",
    decision_out: str | Path = "artifacts/retrain_decision.json",
    run_out: str | Path = "artifacts/retrain_run.json",
    logs_dir: str | Path = "artifacts/retrain_logs",
    execute: bool = False,
    allow_recommended: bool = False,
    allow_warning_promotion: bool = False,
    isolate_run: bool = True,
    current_csv: str | Path | None = None,
    drift_out_html: str | Path = "artifacts/drift_report.html",
    drift_out_json: str | Path = "artifacts/drift_report_summary.json",
) -> dict[str, Any]:
    policy_file = Path(policy_path)
    policy, policy_notes = _merge_policy(policy_file)

    state, state_notes = _read_state(
        metadata_path=Path(metadata_path),
        reference_meta_path=Path(reference_meta_path),
        drift_summary_path=Path(drift_summary_path),
        shift_report_path=Path(shift_report_path),
    )
    decision = evaluate_retrain_decision(policy=policy, state=state)
    decision["policy_path"] = _safe_rel(policy_file)
    decision["policy_version"] = str(policy.get("version") or "unknown")
    decision.setdefault("notes", [])
    decision["notes"] = list(
        dict.fromkeys(
            [*decision["notes"], *policy_notes, *state_notes]
        )
    )
    _write_json(Path(decision_out), decision)

    if not execute:
        _logger.info(
            "Retrain decision generated (dry-run) | decision=%s reasons=%s",
            decision["decision"],
            decision.get("reasons", []),
        )
        return {
            "status": "DRY_RUN",
            "decision": decision,
            "decision_out": _safe_rel(Path(decision_out)),
        }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_logs_dir = Path(logs_dir) / run_id
    run_models_root = (
        Path(models_root) / "runs" / run_id if isolate_run else Path(models_root)
    )
    selection_json = Path(selection_path)
    selection_md = selection_json.with_suffix(".md")

    pre_meta = _safe_read_json(Path(metadata_path)) or {}
    pre_model_version = str(pre_meta.get("model_version") or "unknown")

    run_manifest: dict[str, Any] = {
        "generated_at": _iso_now(),
        "run_id": run_id,
        "status": "PASS",
        "decision": decision,
        "execution": {
            "execute": True,
            "allow_recommended": bool(allow_recommended),
            "allow_warning_promotion": bool(allow_warning_promotion),
            "isolate_run": bool(isolate_run),
        },
        "inputs": {
            "dataset_basename": _dataset_basename(dataset_path),
            "models_root": _safe_rel(Path(models_root)),
            "run_models_root": _safe_rel(run_models_root),
            "selection_path": _safe_rel(selection_json),
            "stage_dir": _safe_rel(Path(stage_dir)),
            "prod_dir": _safe_rel(Path(prod_dir)),
            "reference_dir": _safe_rel(Path(reference_dir)),
        },
        "pre_state": {
            "model_version": pre_model_version,
        },
        "steps": [],
        "notes": [],
    }

    if decision["decision"] == "NOOP":
        run_manifest["status"] = "NOOP"
        run_manifest["notes"].append("no_trigger_fired_execution_skipped")
        _write_json(Path(run_out), run_manifest)
        _logger.info("Retrain orchestration skipped | decision=NOOP")
        return run_manifest

    if (
        decision["decision"] == "RETRAIN_RECOMMENDED"
        and not bool(allow_recommended)
    ):
        run_manifest["status"] = "SKIPPED_RECOMMENDED"
        run_manifest["notes"].append("recommended_trigger_without_override_execution_skipped")
        _write_json(Path(run_out), run_manifest)
        _logger.info("Retrain orchestration skipped | decision=RETRAIN_RECOMMENDED allow_recommended=0")
        return run_manifest

    resolved_dataset = _resolve_dataset_for_execution(dataset_path)
    run_manifest["inputs"]["dataset_basename"] = resolved_dataset.name

    steps, step_notes = _build_retrain_steps(
        python_bin=sys.executable,
        dataset_path=resolved_dataset,
        run_models_root=run_models_root,
        selection_path=selection_json,
        selection_md=selection_md,
        stage_dir=Path(stage_dir),
        prod_dir=Path(prod_dir),
        reference_dir=Path(reference_dir),
        allow_warning_promotion=bool(allow_warning_promotion),
        current_csv=Path(current_csv) if current_csv else None,
        drift_out_html=Path(drift_out_html),
        drift_out_json=Path(drift_out_json),
    )
    run_manifest["notes"].extend(step_notes)

    for step_name, cmd in steps:
        result = _run_step(
            name=step_name,
            command=cmd,
            logs_dir=run_logs_dir,
            dataset_path=resolved_dataset,
        )
        run_manifest["steps"].append(result)
        if int(result["returncode"]) != 0:
            run_manifest["status"] = "FAIL"
            run_manifest["failed_step"] = step_name
            run_manifest["notes"].append(
                f"step_failed={step_name} log={result['log_path']}"
            )
            _write_json(Path(run_out), run_manifest)
            raise RuntimeError(
                f"Retrain orchestration failed at step '{step_name}'. See log: {result['log_path']}"
            )

    post_meta = _safe_read_json(Path(prod_dir) / "metadata.json") or {}
    post_model_version = str(post_meta.get("model_version") or "unknown")
    run_manifest["post_state"] = {
        "model_version": post_model_version,
    }
    if pre_model_version == post_model_version:
        run_manifest["notes"].append("model_version_unchanged_after_retrain")
    else:
        run_manifest["notes"].append("model_version_updated_after_retrain")

    _write_json(Path(run_out), run_manifest)
    _logger.info(
        "Retrain orchestration completed | status=%s run_id=%s",
        run_manifest["status"],
        run_id,
    )
    return run_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local retraining orchestration using time/drift/shift triggers. "
            "Supports dry-run decision mode and end-to-end execute mode."
        )
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to XLSX dataset.")
    parser.add_argument("--policy", type=str, default="docs/retrain_policy.json", help="Retrain policy JSON path.")
    parser.add_argument("--models-root", type=str, default="artifacts/models", help="Root directory for model artifacts.")
    parser.add_argument("--selection-path", type=str, default="artifacts/model_selection.json", help="Selection artifact output path.")
    parser.add_argument("--stage-dir", type=str, default="app/model/staging", help="Staging model directory.")
    parser.add_argument("--prod-dir", type=str, default="app/model", help="Production serving model directory.")
    parser.add_argument("--reference-dir", type=str, default="app/model/reference", help="Reference data directory.")
    parser.add_argument("--drift-summary", type=str, default="artifacts/drift_report_summary.json", help="Drift summary JSON input.")
    parser.add_argument("--shift-report", type=str, default="artifacts/temporal_shift_report.json", help="Temporal shift report JSON input.")
    parser.add_argument("--metadata-path", type=str, default="app/model/metadata.json", help="Serving metadata path.")
    parser.add_argument("--reference-meta-path", type=str, default="app/model/reference/reference_meta.json", help="Reference metadata path.")
    parser.add_argument("--decision-out", type=str, default="artifacts/retrain_decision.json", help="Decision JSON output path.")
    parser.add_argument("--run-out", type=str, default="artifacts/retrain_run.json", help="Run manifest JSON output path.")
    parser.add_argument("--logs-dir", type=str, default="artifacts/retrain_logs", help="Directory to persist per-step logs.")
    parser.add_argument("--execute", type=int, choices=[0, 1], default=0, help="If 1, execute retraining pipeline.")
    parser.add_argument(
        "--allow-recommended",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, allow execution when only recommended triggers fire.",
    )
    parser.add_argument(
        "--allow-warning-promotion",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, allow promotion with WARNING status when policy permits override.",
    )
    parser.add_argument(
        "--isolate-run",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, train under artifacts/models/runs/<run_id> to avoid stale artifact contamination.",
    )
    parser.add_argument("--current-csv", type=str, default=None, help="Optional current MODEL-frame CSV for post-retrain drift refresh.")
    parser.add_argument("--drift-out-html", type=str, default="artifacts/drift_report.html", help="Drift HTML output path.")
    parser.add_argument("--drift-out-json", type=str, default="artifacts/drift_report_summary.json", help="Drift summary JSON output path.")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = _parse_args()
    try:
        payload = run_retrain_orchestration(
            dataset_path=args.dataset_path,
            policy_path=args.policy,
            models_root=args.models_root,
            selection_path=args.selection_path,
            stage_dir=args.stage_dir,
            prod_dir=args.prod_dir,
            reference_dir=args.reference_dir,
            drift_summary_path=args.drift_summary,
            shift_report_path=args.shift_report,
            metadata_path=args.metadata_path,
            reference_meta_path=args.reference_meta_path,
            decision_out=args.decision_out,
            run_out=args.run_out,
            logs_dir=args.logs_dir,
            execute=_to_bool_flag(args.execute),
            allow_recommended=_to_bool_flag(args.allow_recommended),
            allow_warning_promotion=_to_bool_flag(args.allow_warning_promotion),
            isolate_run=_to_bool_flag(args.isolate_run),
            current_csv=args.current_csv,
            drift_out_html=args.drift_out_html,
            drift_out_json=args.drift_out_json,
        )
        status = str(payload.get("status") or "")
        if status == "FAIL":
            return 1
        return 0
    except Exception as exc:
        _logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
