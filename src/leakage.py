"""Explicit data leakage detection helpers for train/inference safeguards."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

DEFAULT_ALLOWLIST: list[str] = [
    "INDE 22",
    "INDE 23",
    "Pedra 20",
    "Pedra 21",
    "Pedra 22",
    "Pedra 23",
    "Pedra_Ano",
]

LEAKAGE_TARGET_COLUMN_NAMES: set[str] = {
    "y",
    "target",
    "label",
    "Defasagem_t1",
    "Defasagem_{t+1}",
    "Defasagem_next",
}

LEAKAGE_SUSPECT_SUFFIXES: tuple[str, ...] = (
    "_t1",
    "_t+1",
    "_next",
    "__t1",
    "__tplus1",
)


def _normalize_column_name(value: object) -> str:
    return str(value).strip()


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def _matches_any_pattern(value: str, compiled: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(value) for pattern in compiled)


def _parse_semantic_year(column_name: str) -> int | None:
    match = re.match(
        r"^(INDE|Pedra)[_\s]*([0-9]{2,4})$",
        column_name,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    token = match.group(2)
    if len(token) == 2:
        return 2000 + int(token)
    return int(token)


def build_blacklist_patterns(
    year_t: int | None = None,
    year_t1: int | None = None,
    include_year_specific: bool = False,
) -> list[str]:
    """Build regex patterns used to detect suspicious future-information columns."""
    patterns = [
        r"(_x$|_y$|_t1$|_t\+1$|_next$|__t1$|__tplus1$)",
        r"\b(t\+1|next[_ ]?year|ano[_ ]?seguinte)\b",
        r"(^y$|^target$|label|target_)",
        r"\bdefasagem\b.*\b(t1|t\+1|next)\b",
    ]

    if include_year_specific and year_t1 is not None:
        # Keep year checks semantic (INDE/Pedra aliases only), never generic year regex.
        patterns.extend(
            [
                rf"^INDE\s*{year_t1}$",
                rf"^Pedra\s*{year_t1}$",
                rf"^INDE[_\s]*{year_t1}$",
                rf"^Pedra[_\s]*{year_t1}$",
            ]
        )

    return sorted(set(patterns))


def detect_leakage_like_columns(
    columns: list[str],
    year_t: int | None = None,
    year_t1: int | None = None,
    extra_blacklist: list[str] | None = None,
    allowlist: list[str] | None = None,
    include_year_specific: bool = False,
) -> dict[str, Any]:
    """Detect leakage-like names (string-level) and provide reasons/allowlist hits."""
    patterns = build_blacklist_patterns(
        year_t=year_t,
        year_t1=year_t1,
        include_year_specific=include_year_specific,
    )
    if extra_blacklist:
        patterns = sorted(set(patterns + list(extra_blacklist)))

    compiled_blacklist = _compile_patterns(patterns)
    allowlist_set = {
        _normalize_column_name(name).lower()
        for name in (allowlist or [])
        if _normalize_column_name(name)
    }
    target_names = {name.lower() for name in LEAKAGE_TARGET_COLUMN_NAMES}

    suspects: list[str] = []
    allowed_hits: list[str] = []
    reasons: dict[str, list[str]] = {}

    for raw_column in columns:
        column = _normalize_column_name(raw_column)
        if not column:
            continue

        column_reasons: list[str] = []
        lower = column.lower()
        if lower in target_names:
            column_reasons.append("target_like_name")

        for suffix in LEAKAGE_SUSPECT_SUFFIXES:
            if lower.endswith(suffix):
                column_reasons.append(f"suffix:{suffix}")

        if _matches_any_pattern(column, compiled_blacklist):
            column_reasons.append("pattern_match")

        semantic_year = _parse_semantic_year(column)
        if year_t is not None and semantic_year is not None and semantic_year > year_t:
            column_reasons.append("semantic_future_year")

        if not column_reasons:
            continue

        if lower in allowlist_set and "semantic_future_year" not in column_reasons:
            allowed_hits.append(column)
            continue

        suspects.append(column)
        reasons[column] = sorted(set(column_reasons))

    return {
        "suspects": sorted(set(suspects)),
        "reasons": {key: reasons[key] for key in sorted(reasons)},
        "allowed_hits": sorted(set(allowed_hits)),
        "patterns_used": sorted(set(patterns)),
    }


def detect_leakage_columns(
    X: pd.DataFrame,
    year_t: int | None = None,
    year_t1: int | None = None,
    extra_blacklist: list[str] | None = None,
    allowlist: list[str] | None = None,
    include_year_specific: bool = False,
) -> dict[str, Any]:
    """Detect suspect leakage columns with backward-compatible fields."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(X)}")

    base_report = detect_leakage_like_columns(
        columns=[_normalize_column_name(column) for column in list(X.columns)],
        year_t=year_t,
        year_t1=year_t1,
        extra_blacklist=extra_blacklist,
        allowlist=allowlist,
        include_year_specific=include_year_specific,
    )
    suspect_columns = base_report["suspects"]

    structural_suspects: list[str] = []
    leakage_real: list[str] = []
    for column in suspect_columns:
        if column not in X.columns:
            continue
        non_null_count = int(X[column].notna().sum())
        if non_null_count == 0:
            structural_suspects.append(column)
        else:
            leakage_real.append(column)

    return {
        "n_columns": int(X.shape[1]),
        "n_suspect": len(suspect_columns),
        "suspect_columns": suspect_columns,
        "patterns_used": base_report["patterns_used"],
        "reasons": base_report["reasons"],
        "allowed_hits": base_report["allowed_hits"],
        "structural_suspects": sorted(set(structural_suspects)),
        "leakage_real": sorted(set(leakage_real)),
    }


def assert_no_leakage(
    X: pd.DataFrame,
    year_t: int | None = None,
    year_t1: int | None = None,
    extra_blacklist: list[str] | None = None,
    allowlist: list[str] | None = None,
    include_year_specific: bool = False,
    context: str = "MODEL",
    tolerate_structural_missing: bool | None = None,
) -> None:
    """Raise ValueError when leakage-like columns are detected in X."""
    report = detect_leakage_columns(
        X=X,
        year_t=year_t,
        year_t1=year_t1,
        extra_blacklist=extra_blacklist,
        allowlist=allowlist,
        include_year_specific=include_year_specific,
    )
    effective_context = context.upper()
    if tolerate_structural_missing is None:
        tolerate_structural_missing = effective_context != "RAW"

    blocking = (
        report["leakage_real"]
        if tolerate_structural_missing
        else report["suspect_columns"]
    )
    if not blocking:
        return

    pair = (
        f"{year_t}->{year_t1}"
        if year_t is not None or year_t1 is not None
        else "unknown->unknown"
    )
    tolerated_note = ""
    if tolerate_structural_missing:
        tolerated_note = f" | tolerated_structural={len(report['structural_suspects'])}"

    suspects = ", ".join(blocking)
    raise ValueError(
        f"[{effective_context}] Leakage detected {pair}: "
        f"{len(blocking)} suspect columns: {suspects}{tolerated_note}"
    )
