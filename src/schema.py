"""Schema harmonization utilities for multi-year PEDE datasets."""

from __future__ import annotations

import re
from typing import Mapping

import pandas as pd

from src.utils import get_logger

_logger = get_logger(__name__)
_DUP_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.(?P<idx>\d+)$")

_INDE_CANDIDATES_BY_YEAR: dict[int, list[str]] = {
    2022: ["INDE 22"],
    2023: ["INDE 2023", "INDE 23", "INDE 22"],
    2024: ["INDE 2024", "INDE 23", "INDE 22"],
}

_PEDRA_CANDIDATES_BY_YEAR: dict[int, list[str]] = {
    2022: ["Pedra 22", "Pedra 21", "Pedra 20"],
    2023: ["Pedra 2023", "Pedra 23", "Pedra 22"],
    2024: ["Pedra 2024", "Pedra 23", "Pedra 22"],
}

_YEAR_SUPPORTED = tuple(sorted(_INDE_CANDIDATES_BY_YEAR))


def _validate_year(year: int) -> None:
    if year not in _INDE_CANDIDATES_BY_YEAR:
        raise ValueError(f"Ano inválido: {year}. Anos suportados: {list(_YEAR_SUPPORTED)}")


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply strip normalization to column headers."""
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    return normalized


def resolve_duplicate_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename duplicated header suffixes (.1/.2/...) into deterministic __dupN names."""
    resolved = df.copy()
    original_cols = [str(col) for col in resolved.columns]
    occupied = set(original_cols)
    new_cols: list[str] = []
    rename_map: dict[str, str] = {}

    for col in original_cols:
        match = _DUP_SUFFIX_RE.match(col)
        if not match:
            if col in new_cols:
                base = col
                suffix = 1
                candidate = f"{base}__dup{suffix}"
                while candidate in occupied or candidate in new_cols:
                    suffix += 1
                    candidate = f"{base}__dup{suffix}"
                new_cols.append(candidate)
                occupied.add(candidate)
                rename_map[col] = candidate
            else:
                new_cols.append(col)
            continue

        base = match.group("base")
        suffix = 1
        candidate = f"{base}__dup{suffix}"
        while candidate in occupied or candidate in new_cols:
            suffix += 1
            candidate = f"{base}__dup{suffix}"

        new_cols.append(candidate)
        occupied.add(candidate)
        rename_map[col] = candidate

    resolved.columns = new_cols
    return resolved, rename_map


def build_canonical_rename_map(year: int) -> dict[str, str]:
    """Return canonical header rename rules for known year schema differences."""
    _validate_year(year)
    return {
        "Defas": "Defasagem",
        "Ano nasc": "Data_Nasc",
        "Data de Nasc": "Data_Nasc",
        "Idade 22": "Idade",
        "Fase ideal": "Fase_Ideal",
        "Fase Ideal": "Fase_Ideal",
        "Matem": "Mat",
        "Portug": "Por",
        "Inglês": "Ing",
        "Nome": "Nome_Anon",
        "Nome Anonimizado": "Nome_Anon",
    }


def _apply_canonical_renames(
    df: pd.DataFrame, rename_map: Mapping[str, str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    renamed = df.copy()
    cols = [str(c) for c in renamed.columns]
    used: set[str] = set()
    new_cols: list[str] = []
    applied: dict[str, str] = {}

    for col in cols:
        target = rename_map.get(col, col)
        if target in used:
            dup_idx = 1
            candidate = f"{target}__dup{dup_idx}"
            while candidate in used:
                dup_idx += 1
                candidate = f"{target}__dup{dup_idx}"
            target = candidate

        new_cols.append(target)
        used.add(target)
        if target != col:
            applied[col] = target

    renamed.columns = new_cols
    return renamed, applied


def select_with_fallback(
    df: pd.DataFrame, candidates: list[str]
) -> tuple[pd.Series | None, str | None]:
    """Select first existing column from candidates, returning series and source name."""
    for col in candidates:
        if col in df.columns:
            return df[col], col
    return None, None


def harmonize_schema_year(
    df: pd.DataFrame,
    year: int,
    logger=None,
) -> pd.DataFrame:
    """Harmonize one year schema and create canonical INDE and Pedra_Ano columns."""
    _validate_year(year)
    log = logger or _logger

    before_shape = df.shape
    normalized = normalize_headers(df)
    deduped, dup_rename_map = resolve_duplicate_headers(normalized)
    canonical_map = build_canonical_rename_map(year)
    harmonized, canonical_rename_map = _apply_canonical_renames(deduped, canonical_map)

    inde_series, inde_source = select_with_fallback(harmonized, _INDE_CANDIDATES_BY_YEAR[year])
    if inde_series is None:
        harmonized["INDE"] = pd.NA
        log.warning("Harmonize schema year=%d | INDE source missing, filled with NaN", year)
        inde_source = None
    else:
        harmonized["INDE"] = inde_series

    pedra_series, pedra_source = select_with_fallback(
        harmonized, _PEDRA_CANDIDATES_BY_YEAR[year]
    )
    if pedra_series is None:
        harmonized["Pedra_Ano"] = pd.NA
        log.warning("Harmonize schema year=%d | Pedra_Ano source missing, filled with NaN", year)
        pedra_source = None
    else:
        harmonized["Pedra_Ano"] = pedra_series

    duplicate_base_names = sorted(
        {
            old_col.rsplit(".", 1)[0]
            for old_col in dup_rename_map
            if _DUP_SUFFIX_RE.match(old_col)
        }
    )
    log.info(
        "Harmonize schema year=%d | shape_before=%s shape_after=%s canonical_renames=%d "
        "duplicates_resolved=%d duplicate_bases=%s INDE_source=%s Pedra_Ano_source=%s",
        year,
        before_shape,
        harmonized.shape,
        len(canonical_rename_map),
        len(dup_rename_map),
        duplicate_base_names,
        inde_source,
        pedra_source,
    )
    return harmonized


def align_years(
    dfs: dict[int, pd.DataFrame],
    years: list[int] | tuple[int, ...] = (2022, 2023, 2024),
    logger=None,
) -> dict[int, pd.DataFrame]:
    """Harmonize and align yearly DataFrames to the same canonical schema."""
    log = logger or _logger
    harmonized: dict[int, pd.DataFrame] = {}
    for year in years:
        if year not in dfs:
            raise ValueError(f"Ano {year} ausente no dicionário de entrada para align_years.")
        harmonized[year] = harmonize_schema_year(dfs[year], year=year, logger=log)

    all_columns: set[str] = set()
    for df in harmonized.values():
        all_columns.update(df.columns)

    if "RA" in all_columns:
        ordered_columns = ["RA"] + sorted(col for col in all_columns if col != "RA")
    else:
        ordered_columns = sorted(all_columns)

    aligned: dict[int, pd.DataFrame] = {}
    for year, df in harmonized.items():
        missing = [col for col in ordered_columns if col not in df.columns]
        aligned_df = df.copy()
        for col in missing:
            aligned_df[col] = pd.NA
        aligned_df = aligned_df.loc[:, ordered_columns]

        log.info(
            "Align schema year=%d | shape_before=%s shape_after=%s added_nan_columns=%d",
            year,
            df.shape,
            aligned_df.shape,
            len(missing),
        )
        aligned[year] = aligned_df

    return aligned

