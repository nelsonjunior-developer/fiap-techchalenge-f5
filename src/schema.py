"""Schema harmonization utilities for multi-year PEDE datasets."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.column_mapping import harmonize_year_columns
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


def select_with_fallback(
    df: pd.DataFrame, candidates: list[str]
) -> tuple[pd.Series | None, str | None]:
    """Select first existing column from candidates, returning series and source name."""
    for col in candidates:
        if col in df.columns:
            return df[col], col
    return None, None


def harmonize_schema_year_with_report(
    df: pd.DataFrame,
    year: int,
    logger=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Harmonize one year schema and create canonical INDE and Pedra_Ano columns."""
    _validate_year(year)
    log = logger or _logger

    before_shape = df.shape
    normalized = normalize_headers(df)
    deduped, dup_rename_map = resolve_duplicate_headers(normalized)
    mapped, mapping_report = harmonize_year_columns(deduped, year=year, strict=False)
    harmonized = mapped.copy()

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
        "Harmonize schema year=%d | shape_before=%s shape_after=%s mapping_renames=%d "
        "mapping_merges=%d duplicates_resolved=%d duplicate_bases=%s INDE_source=%s Pedra_Ano_source=%s",
        year,
        before_shape,
        harmonized.shape,
        len(mapping_report["renamed"]),
        len(mapping_report["merged"]),
        len(dup_rename_map),
        duplicate_base_names,
        inde_source,
        pedra_source,
    )
    mapping_report["header_duplicates_renamed"] = dict(dup_rename_map)
    return harmonized, mapping_report


def harmonize_schema_year(
    df: pd.DataFrame,
    year: int,
    logger=None,
) -> pd.DataFrame:
    """Backward-compatible wrapper returning only harmonized dataframe."""
    harmonized, _ = harmonize_schema_year_with_report(df=df, year=year, logger=logger)
    return harmonized


def align_years(
    dfs: dict[int, pd.DataFrame],
    years: list[int] | tuple[int, ...] = (2022, 2023, 2024),
    logger=None,
) -> dict[int, pd.DataFrame]:
    """Backward-compatible wrapper returning only aligned dataframes."""
    aligned, _ = align_years_with_metadata(dfs=dfs, years=years, logger=logger)
    return aligned


def align_years_with_metadata(
    dfs: dict[int, pd.DataFrame],
    years: list[int] | tuple[int, ...] = (2022, 2023, 2024),
    logger=None,
) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    """Harmonize and align yearly DataFrames to the same canonical schema."""
    log = logger or _logger
    harmonized: dict[int, pd.DataFrame] = {}
    original_columns: dict[int, set[str]] = {}
    mapping_reports: dict[int, dict[str, Any]] = {}
    for year in years:
        if year not in dfs:
            raise ValueError(f"Ano {year} ausente no dicionário de entrada para align_years.")
        harmonized_df, mapping_report = harmonize_schema_year_with_report(
            dfs[year], year=year, logger=log
        )
        harmonized[year] = harmonized_df
        mapping_reports[year] = mapping_report
        # Preserve the pre-alignment schema to distinguish structural vs real missingness.
        original_columns[year] = set(harmonized_df.columns)

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

    schema_identical = len({tuple(df.columns) for df in aligned.values()}) <= 1
    metadata = {
        "original_columns": original_columns,
        "aligned_columns": ordered_columns,
        "schema_identical": schema_identical,
        "column_mapping_report": mapping_reports,
    }
    return aligned, metadata
