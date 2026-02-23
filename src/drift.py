"""Local drift report generation with Evidently (HTML) on MODEL-frame data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.privacy import SENSITIVE_FIELD_NAMES, find_forbidden_json_keys, is_safe_json_payload
from src.utils import get_logger, log_event, setup_logging

_logger = get_logger(__name__)

_REFERENCE_CSV_NAME = "reference_model_frame.csv"
_REFERENCE_META_NAME = "reference_meta.json"
_DEFAULT_OUT_HTML = "artifacts/drift_report.html"
_DEFAULT_OUT_JSON = "artifacts/drift_report_summary.json"
_DEFAULT_WARN_SHARE = 0.10
_DEFAULT_FAIL_SHARE = 0.30
_DEFAULT_MAX_ROWS = 2000
_DEFAULT_SEED = 42


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload (expected object): {path}")
    return payload


def _resolve_reference_paths(reference_dir: str | Path) -> dict[str, Path]:
    base = Path(reference_dir)
    return {
        "reference_dir": base,
        "reference_csv": base / _REFERENCE_CSV_NAME,
        "reference_meta": base / _REFERENCE_META_NAME,
    }


def _require_evidently() -> tuple[type[Any], type[Any]]:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "Evidently is required to generate drift reports. Install requirements-dev.txt"
        ) from exc
    return Report, DataDriftPreset


def _norm_col_name(value: Any) -> str:
    return str(value).strip()


def _is_sensitive_col_name(name: str) -> bool:
    lowered = name.strip().lower()
    if not lowered:
        return False
    if lowered in {str(x).strip().lower() for x in SENSITIVE_FIELD_NAMES}:
        return True
    return lowered.startswith("avaliador")


def _assert_no_sensitive_columns(columns: Iterable[Any], *, context: str) -> None:
    cols = [_norm_col_name(col) for col in columns]
    forbidden = sorted([col for col in cols if _is_sensitive_col_name(col)])
    if forbidden:
        raise ValueError(
            f"{context} contains sensitive columns (expected MODEL frame without PII): {forbidden}"
        )


def _load_reference_assets(reference_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path], list[str]]:
    paths = _resolve_reference_paths(reference_dir)
    if not paths["reference_csv"].exists() or not paths["reference_meta"].exists():
        raise FileNotFoundError(
            "Reference data not found. Run build_reference_data first. "
            f"Expected files: {paths['reference_csv']} and {paths['reference_meta']}"
        )

    reference_df = pd.read_csv(paths["reference_csv"])
    if reference_df.empty or reference_df.shape[1] == 0:
        raise ValueError(f"Reference MODEL frame is empty or has no columns: {paths['reference_csv']}")

    reference_meta = _safe_read_json(paths["reference_meta"])
    notes: list[str] = []

    _assert_no_sensitive_columns(reference_df.columns, context="reference_model_frame")

    meta_expected = reference_meta.get("expected_model_cols")
    if isinstance(meta_expected, list):
        cleaned = [_norm_col_name(col) for col in meta_expected if _norm_col_name(col)]
        if cleaned and cleaned != [_norm_col_name(col) for col in reference_df.columns]:
            notes.append("reference_meta_expected_model_cols_mismatch_csv_header_using_csv_header")

    return reference_df, reference_meta, paths, notes


def _load_current_csv(current_csv: str | Path) -> pd.DataFrame:
    path = Path(current_csv)
    if not path.exists():
        raise FileNotFoundError(f"Current MODEL frame CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty or df.shape[1] == 0:
        raise ValueError(f"Current MODEL frame CSV is empty or has no columns: {path}")
    _assert_no_sensitive_columns(df.columns, context="current_model_frame")
    return df


def _align_current_to_reference(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ref_cols = [_norm_col_name(col) for col in reference_df.columns]
    cur_cols = [_norm_col_name(col) for col in current_df.columns]
    current_renamed = current_df.copy()
    current_renamed.columns = cur_cols

    missing_cols = [col for col in ref_cols if col not in cur_cols]
    if missing_cols:
        missing_preview = missing_cols[:50]
        raise ValueError(
            "Current MODEL frame is missing required reference columns. "
            f"missing_count={len(missing_cols)} missing_preview={missing_preview}"
        )

    extra_cols = [col for col in cur_cols if col not in ref_cols]
    aligned = current_renamed.drop(columns=extra_cols, errors="ignore").loc[:, ref_cols].copy()
    return aligned, {
        "reference_cols_count": int(len(ref_cols)),
        "missing_cols_count": 0,
        "extra_cols_count": int(len(extra_cols)),
        "extra_cols_preview": extra_cols[:10],
    }


def _sample_frame(
    df: pd.DataFrame,
    *,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_rows = int(len(df))
    if max_rows <= 0:
        raise ValueError("max_rows must be > 0")
    if n_rows <= max_rows:
        return df.reset_index(drop=True), {"original_rows": n_rows, "sampled_rows": n_rows, "sampled": False}
    sampled = df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)
    return sampled, {"original_rows": n_rows, "sampled_rows": int(len(sampled)), "sampled": True}


def _walk_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_dicts(item)


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_evidently_summary(report_dict: Mapping[str, Any]) -> dict[str, Any]:
    candidate: dict[str, Any] | None = None
    for node in _walk_dicts(report_dict):
        keys = set(node.keys())
        if keys & {
            "number_of_drifted_columns",
            "number_of_drifted_features",
            "share_of_drifted_columns",
            "share_of_drifted_features",
            "drift_share",
            "dataset_drift",
            "drift_by_columns",
        }:
            candidate = node
            if "dataset_drift" in keys or "drift_by_columns" in keys:
                break

    if candidate is None:
        return {
            "drifted_features_count": None,
            "share_drifted_features": None,
            "dataset_drift": None,
            "notes": ["evidently_summary_fields_not_found"],
        }

    drifted_count = _to_int_or_none(
        candidate.get("number_of_drifted_columns", candidate.get("number_of_drifted_features"))
    )
    total_cols = _to_int_or_none(candidate.get("number_of_columns", candidate.get("number_of_features")))
    share = _to_float_or_none(
        candidate.get(
            "share_of_drifted_columns",
            candidate.get("share_of_drifted_features", candidate.get("drift_share")),
        )
    )

    drift_by_columns = candidate.get("drift_by_columns")
    if isinstance(drift_by_columns, dict):
        if total_cols is None:
            total_cols = int(len(drift_by_columns))
        if drifted_count is None:
            count = 0
            counted_any = False
            for item in drift_by_columns.values():
                if not isinstance(item, dict):
                    continue
                if "drift_detected" in item:
                    counted_any = True
                    count += 1 if bool(item.get("drift_detected")) else 0
                elif "drifted" in item:
                    counted_any = True
                    count += 1 if bool(item.get("drifted")) else 0
            if counted_any:
                drifted_count = int(count)

    if share is None and drifted_count is not None and total_cols not in (None, 0):
        share = float(drifted_count / total_cols)

    dataset_drift_raw = candidate.get("dataset_drift")
    dataset_drift = bool(dataset_drift_raw) if isinstance(dataset_drift_raw, bool) else None

    notes: list[str] = []
    if share is None:
        notes.append("share_drifted_features_unavailable")
    if drifted_count is None:
        notes.append("drifted_features_count_unavailable")

    return {
        "drifted_features_count": drifted_count,
        "share_drifted_features": share,
        "dataset_drift": dataset_drift,
        "notes": notes,
    }


def _build_evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Any:
    Report, DataDriftPreset = _require_evidently()
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    return report


def _save_evidently_html(report: Any, out_html: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    save_html = getattr(report, "save_html", None)
    if callable(save_html):
        save_html(str(out_html))
        return
    for method_name in ("get_html", "html"):
        method = getattr(report, method_name, None)
        if callable(method):
            html = method()
            if isinstance(html, str) and html.strip():
                out_html.write_text(html, encoding="utf-8")
                return
    raise RuntimeError("Unable to export Evidently HTML report (unsupported API).")


def _report_as_dict(report: Any) -> dict[str, Any]:
    for method_name in ("as_dict", "dict"):
        method = getattr(report, method_name, None)
        if callable(method):
            payload = method()
            if isinstance(payload, dict):
                return payload
    raise RuntimeError("Unable to extract Evidently report as dict (unsupported API).")


def _status_from_drift_share(
    *,
    share_drifted_features: float | None,
    warn_share_threshold: float,
    fail_share_threshold: float,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    if fail_share_threshold < warn_share_threshold:
        raise ValueError("fail_share_threshold must be >= warn_share_threshold")
    if share_drifted_features is None:
        return "FAIL", ["share_drifted_features_unavailable"]
    if share_drifted_features >= float(fail_share_threshold):
        notes.append("drift_share_threshold_fail")
        return "FAIL", notes
    if share_drifted_features >= float(warn_share_threshold):
        notes.append("drift_share_threshold_warning")
        return "WARNING", notes
    return "PASS", notes


def run_drift_report(
    *,
    reference_dir: str | Path = "app/model/reference",
    current_csv: str | Path,
    out_html: str | Path = _DEFAULT_OUT_HTML,
    out_json: str | Path = _DEFAULT_OUT_JSON,
    write_json: bool = True,
    max_rows: int = _DEFAULT_MAX_ROWS,
    seed: int = _DEFAULT_SEED,
    warn_share_threshold: float = _DEFAULT_WARN_SHARE,
    fail_share_threshold: float = _DEFAULT_FAIL_SHARE,
) -> dict[str, Any]:
    reference_df, reference_meta, reference_paths, notes = _load_reference_assets(reference_dir)
    current_df = _load_current_csv(current_csv)
    current_aligned, alignment_info = _align_current_to_reference(reference_df, current_df)

    reference_sample, reference_sample_info = _sample_frame(reference_df, max_rows=int(max_rows), seed=int(seed))
    current_sample, current_sample_info = _sample_frame(current_aligned, max_rows=int(max_rows), seed=int(seed))

    report = _build_evidently_report(reference_sample, current_sample)

    out_html_path = Path(out_html)
    _save_evidently_html(report, out_html_path)
    report_dict = _report_as_dict(report)
    evidently_summary = _extract_evidently_summary(report_dict)

    status, status_notes = _status_from_drift_share(
        share_drifted_features=evidently_summary.get("share_drifted_features"),
        warn_share_threshold=float(warn_share_threshold),
        fail_share_threshold=float(fail_share_threshold),
    )

    winner = reference_meta.get("winner")
    if not isinstance(winner, dict):
        winner = {}

    summary: dict[str, Any] = {
        "status": status,
        "generated_at": _utc_now_iso(),
        "reference": {
            "dir": str(reference_paths["reference_dir"]),
            "csv": str(reference_paths["reference_csv"]),
            "meta": str(reference_paths["reference_meta"]),
            "model_version": reference_meta.get("model_version"),
            "model_family": winner.get("model_family"),
            "variant": winner.get("variant"),
            "n_rows": int(len(reference_df)),
            "n_rows_used": int(reference_sample_info["sampled_rows"]),
        },
        "current": {
            "csv": str(Path(current_csv)),
            "n_rows": int(len(current_df)),
            "n_rows_used": int(current_sample_info["sampled_rows"]),
        },
        "contract": {
            "n_features": int(reference_df.shape[1]),
            "extra_cols_dropped_count": int(alignment_info["extra_cols_count"]),
            "extra_cols_dropped_preview": list(alignment_info["extra_cols_preview"]),
        },
        "sampling": {
            "max_rows": int(max_rows),
            "seed": int(seed),
            "reference_sampled": bool(reference_sample_info["sampled"]),
            "current_sampled": bool(current_sample_info["sampled"]),
        },
        "drift": {
            "drifted_features_count": evidently_summary.get("drifted_features_count"),
            "share_drifted_features": evidently_summary.get("share_drifted_features"),
            "dataset_drift": evidently_summary.get("dataset_drift"),
        },
        "thresholds": {
            "warning_share_drifted_features": float(warn_share_threshold),
            "fail_share_drifted_features": float(fail_share_threshold),
        },
        "notes": list(
            dict.fromkeys(
                [
                    *notes,
                    *evidently_summary.get("notes", []),
                    *status_notes,
                    "drift_report_uses_model_frame_only_no_raw_payload",
                    "temporal_shift_cli_and_evidently_drift_report_are_complementary",
                ]
            )
        ),
        "errors": [],
    }

    forbidden = find_forbidden_json_keys(summary)
    if forbidden:
        raise ValueError(f"Privacy check failed in drift summary: forbidden keys found {forbidden}")
    if not is_safe_json_payload(summary):
        raise ValueError("Privacy check failed in drift summary: unsafe JSON payload detected")

    if bool(write_json):
        out_json_path = Path(out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log_event(
        _logger,
        "drift_report_generated",
        status=summary.get("status"),
        drifted_features_count=summary["drift"]["drifted_features_count"],
        share_drifted_features=summary["drift"]["share_drifted_features"],
        n_features=summary["contract"]["n_features"],
        reference_rows=summary["reference"]["n_rows_used"],
        current_rows=summary["current"]["n_rows_used"],
        model_version=summary["reference"]["model_version"],
        out_html=str(out_html_path),
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local Evidently HTML drift report using MODEL-frame reference/current CSVs."
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="app/model/reference",
        help="Directory containing reference_model_frame.csv and reference_meta.json.",
    )
    parser.add_argument(
        "--current-csv",
        type=str,
        required=True,
        help="Path to current MODEL-frame CSV (same feature schema as reference; extras are ignored).",
    )
    parser.add_argument(
        "--out-html",
        type=str,
        default=_DEFAULT_OUT_HTML,
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=_DEFAULT_OUT_JSON,
        help="Output JSON summary path (use --no-json to disable).",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable writing JSON summary output.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=_DEFAULT_MAX_ROWS,
        help="Maximum rows to use from each dataset (deterministic sampling if larger).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--warn-share",
        type=float,
        default=_DEFAULT_WARN_SHARE,
        help="WARNING threshold for share of drifted features.",
    )
    parser.add_argument(
        "--fail-share",
        type=float,
        default=_DEFAULT_FAIL_SHARE,
        help="FAIL threshold for share of drifted features.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    try:
        summary = run_drift_report(
            reference_dir=args.reference_dir,
            current_csv=args.current_csv,
            out_html=args.out_html,
            out_json=args.out_json,
            write_json=not bool(args.no_json),
            max_rows=int(args.max_rows),
            seed=int(args.seed),
            warn_share_threshold=float(args.warn_share),
            fail_share_threshold=float(args.fail_share),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _logger.error("%s", exc)
        raise SystemExit(1) from exc
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive branch
        _logger.error("Unexpected drift report failure: %s", exc)
        raise SystemExit(1) from exc

    _logger.info(
        "Drift report generated | status=%s out_html=%s out_json=%s",
        summary.get("status"),
        args.out_html,
        None if bool(args.no_json) else args.out_json,
    )


if __name__ == "__main__":
    main()

