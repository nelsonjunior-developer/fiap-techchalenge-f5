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
]


def build_blacklist_patterns(
    year_t: int | None = None,
    year_t1: int | None = None,
    include_year_specific: bool = False,
) -> list[str]:
    """Build regex patterns used to detect suspicious future-information columns."""
    patterns = [
        r"(_x$|_y$|_t1$|_t\+1$)",
        r"(t\+1|next[_ ]?year|ano[_ ]?seguinte)",
        r"(^y$|^target$|label|target_)",
        r"defasagem.*(t\+1|_t1|_y$)",
    ]

    if include_year_specific and year_t1 is not None:
        # Explicit year-t+1 column aliases that should not be features in X(t).
        patterns.extend(
            [
                rf"^INDE\s*{year_t1}$",
                rf"^Pedra\s*{year_t1}$",
                rf"^INDE[_\s]*{year_t1}$",
                rf"^Pedra[_\s]*{year_t1}$",
            ]
        )

    return sorted(set(patterns))


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def _matches_any_pattern(value: str, compiled: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(value) for pattern in compiled)


def detect_leakage_columns(
    X: pd.DataFrame,
    year_t: int | None = None,
    year_t1: int | None = None,
    extra_blacklist: list[str] | None = None,
    allowlist: list[str] | None = None,
    include_year_specific: bool = False,
) -> dict[str, Any]:
    """Detect suspect leakage columns based on blacklist regex patterns."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(X)}")

    patterns = build_blacklist_patterns(
        year_t=year_t,
        year_t1=year_t1,
        include_year_specific=include_year_specific,
    )
    if extra_blacklist:
        patterns = sorted(set(patterns + list(extra_blacklist)))

    compiled_blacklist = _compile_patterns(patterns)
    allowlist_set = {name.strip().lower() for name in (allowlist or []) if name.strip()}

    suspect_columns: list[str] = []
    for column in X.columns:
        normalized = str(column).strip()
        if not normalized:
            continue
        if normalized.lower() in allowlist_set:
            continue
        if _matches_any_pattern(normalized, compiled_blacklist):
            suspect_columns.append(str(column))

    return {
        "n_columns": int(X.shape[1]),
        "n_suspect": len(suspect_columns),
        "suspect_columns": sorted(set(suspect_columns)),
        "patterns_used": sorted(set(patterns)),
    }


def assert_no_leakage(
    X: pd.DataFrame,
    year_t: int | None = None,
    year_t1: int | None = None,
    extra_blacklist: list[str] | None = None,
    allowlist: list[str] | None = None,
    include_year_specific: bool = False,
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
    if report["n_suspect"] > 0:
        suspects = ", ".join(report["suspect_columns"])
        raise ValueError(
            f"Leakage detected {year_t}->{year_t1}: "
            f"{report['n_suspect']} suspect columns: {suspects}"
        )
