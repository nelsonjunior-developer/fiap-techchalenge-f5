"""Utilities for /predict payload normalization and RAW-frame guards."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.request_schemas import RecordModel, RecordsEnvelope
from src.leakage import detect_leakage_columns


def normalize_records(payload: Any) -> list[dict[str, Any]]:
    """Normalize payload into list[dict] supporting dict/list/envelope formats."""
    records: list[dict[str, Any]]
    if isinstance(payload, RecordsEnvelope):
        records = [record.to_dict() for record in payload.records]
        return records

    if isinstance(payload, RecordModel):
        return [payload.to_dict()]

    if isinstance(payload, list):
        resolved: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, RecordModel):
                resolved.append(item.to_dict())
            elif isinstance(item, dict):
                resolved.append(dict(item))
            else:
                raise ValueError("payload records must be objects")
        return resolved

    if isinstance(payload, dict):
        if "records" in payload:
            records_payload = payload["records"]
            if not isinstance(records_payload, list):
                raise ValueError(
                    "payload must be a dict, a list of dicts, or {'records': [...]}"
                )
            if any(not isinstance(item, dict) for item in records_payload):
                raise ValueError("payload records must be objects")
            return [dict(item) for item in records_payload]
        else:
            return [dict(payload)]

    raise ValueError("payload must be a dict, a list of dicts, or {'records': [...]}")


def build_raw_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build RAW dataframe from normalized records."""
    return pd.DataFrame.from_records(records)


def validate_required_columns(
    df: pd.DataFrame, expected_raw_cols: list[str]
) -> tuple[bool, list[str]]:
    """Validate that all expected RAW columns exist in input frame."""
    missing = [col for col in expected_raw_cols if col not in df.columns]
    return len(missing) == 0, missing


def apply_leakage_gate_on_extras(
    df: pd.DataFrame,
    expected_raw_cols: list[str],
) -> None:
    """Block payload extras if they look leakage-like, allow non-suspicious extras."""
    extras = sorted(set(str(col) for col in df.columns) - set(expected_raw_cols))
    if not extras:
        return
    extras_df = df.loc[:, extras].copy()
    report = detect_leakage_columns(
        X=extras_df,
        year_t=None,
        year_t1=None,
        include_year_specific=False,
    )
    if int(report.get("n_suspect", 0)) <= 0:
        return
    suspects = ", ".join(report.get("suspect_columns", []))
    raise ValueError(
        f"[RAW] leakage-like extra columns detected in payload: {suspects}"
    )


__all__ = [
    "apply_leakage_gate_on_extras",
    "build_raw_dataframe",
    "normalize_records",
    "validate_required_columns",
]
