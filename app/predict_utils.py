"""Utilities for /predict payload normalization and RAW-frame guards."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from app.request_schemas import RecordModel, RecordsEnvelope
from src.contracts import Presence, SUPPORTED_YEARS, get_year_contract
from src.leakage import detect_leakage_columns


class MissingColumnsError(ValueError):
    """Typed payload validation error carrying missing column names and stats."""

    def __init__(
        self,
        missing_columns: list[str],
        *,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self.missing_columns = list(missing_columns)
        self.stats = dict(stats or {})
        message = (
            "Missing required columns"
            if not self.missing_columns
            else f"Missing required columns: {self.missing_columns}"
        )
        super().__init__(message)


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


@lru_cache(maxsize=1)
def _structural_optional_columns_union() -> frozenset[str]:
    """Union of contract columns marked as structural_optional across supported years."""
    columns: set[str] = set()
    for year in SUPPORTED_YEARS:
        contract = get_year_contract(int(year))
        for name, spec in contract.columns.items():
            if getattr(spec, "presence", None) == Presence.STRUCTURAL_OPTIONAL:
                columns.add(str(name))
    return frozenset(columns)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize_expected_columns(expected_raw_cols: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in expected_raw_cols:
        col = str(value).strip()
        if not col or col in seen:
            continue
        normalized.append(col)
        seen.add(col)
    return normalized


def compute_missing_stats(df_raw: pd.DataFrame, expected_raw_cols: list[str]) -> dict[str, Any]:
    """Compute aggregated missing-column/value diagnostics without cell-value logging."""
    expected = _normalize_expected_columns(expected_raw_cols)
    observed = [str(col) for col in list(df_raw.columns)]
    observed_set = set(observed)

    missing_cols = [col for col in expected if col not in observed_set]
    present_cols = [col for col in expected if col in observed_set]
    extras = sorted(set(observed) - set(expected))

    X_raw = df_raw.reindex(columns=expected, fill_value=pd.NA)
    total_cells = int(X_raw.shape[0] * X_raw.shape[1])
    missing_values_count = int(X_raw.isna().sum().sum()) if total_cells > 0 else 0

    structural_optional_union = set(_structural_optional_columns_union())
    expected_non_structural = [
        col for col in expected if col not in structural_optional_union
    ]
    missing_non_structural = [
        col for col in missing_cols if col not in structural_optional_union
    ]
    missing_structural_optional = [
        col for col in missing_cols if col in structural_optional_union
    ]

    stats = {
        "expected_cols_count": int(len(expected)),
        "present_cols_count": int(len(present_cols)),
        "missing_cols_count": int(len(missing_cols)),
        "missing_cols_rate": _safe_rate(len(missing_cols), len(expected)),
        "missing_values_count": int(missing_values_count),
        "missing_values_rate": _safe_rate(missing_values_count, total_cells),
        "extra_cols_count": int(len(extras)),
        "top_missing_cols": list(missing_cols[:10]),
        "structural_optional_expected_cols_count": int(
            len(expected) - len(expected_non_structural)
        ),
        "missing_structural_optional_cols_count": int(len(missing_structural_optional)),
        "missing_non_structural_cols_count": int(len(missing_non_structural)),
        "missing_non_structural_cols_rate": _safe_rate(
            len(missing_non_structural), len(expected_non_structural)
        ),
    }
    return stats


def build_model_input_frame(
    df_payload: pd.DataFrame,
    expected_raw_cols: list[str],
    allow_partial: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate RAW payload schema/extras and build model-ready RAW frame via reindex."""
    normalized_expected = _normalize_expected_columns(expected_raw_cols)
    observed_cols_normalized = {str(col) for col in list(df_payload.columns)}

    # Leakage-like extras are always blocked before any partial-payload fallback.
    apply_leakage_gate_on_extras(df_payload, normalized_expected)

    stats = compute_missing_stats(df_payload, normalized_expected)
    missing_cols = [col for col in normalized_expected if col not in observed_cols_normalized]
    allow_partial_enabled = bool(allow_partial)
    allow_partial_used = bool(allow_partial_enabled and len(missing_cols) > 0)

    stats["allow_partial_enabled"] = allow_partial_enabled
    stats["allow_partial_used"] = allow_partial_used

    if missing_cols and not allow_partial_enabled:
        raise MissingColumnsError(missing_cols, stats=stats)

    X_raw = df_payload.reindex(columns=normalized_expected, fill_value=pd.NA)
    return X_raw, stats


__all__ = [
    "MissingColumnsError",
    "apply_leakage_gate_on_extras",
    "build_raw_dataframe",
    "build_model_input_frame",
    "compute_missing_stats",
    "normalize_records",
    "validate_required_columns",
]
