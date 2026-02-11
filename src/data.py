"""Data ingestion helpers for PEDE datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import pandas as pd

_SPACES_RE = re.compile(r"\s+")
_DUPLICATE_SUFFIX_RE = re.compile(r"\.\d+$")

DEFAULT_SHEETS_BY_YEAR: dict[int, str] = {
    2022: "PEDE2022",
    2023: "PEDE2023",
    2024: "PEDE2024",
}


def _normalize_header(name: object) -> str:
    text = "" if name is None else str(name)
    text = _SPACES_RE.sub(" ", text).strip()
    return text


def _defas_base_name(column_name: str) -> str | None:
    normalized = _normalize_header(column_name).lower()
    normalized = _DUPLICATE_SUFFIX_RE.sub("", normalized)
    if normalized in {"defas", "defasagem"}:
        return normalized
    return None


def standardize_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Standardize schema differences by renaming Defas (2022) to Defasagem."""
    standardized = df.copy()
    standardized.columns = [_normalize_header(col) for col in standardized.columns]

    candidates: list[tuple[int, str]] = []
    for idx, col in enumerate(standardized.columns):
        base_name = _defas_base_name(col)
        if base_name is not None:
            candidates.append((idx, base_name))

    if not candidates:
        raise ValueError(
            f"Nenhuma coluna de defasagem encontrada para year={year}. "
            f"Colunas disponíveis: {list(standardized.columns)}"
        )

    preferred_candidates = [item for item in candidates if item[1] == "defasagem"]
    chosen_idx = preferred_candidates[0][0] if preferred_candidates else candidates[0][0]

    merged_defasagem = standardized.iloc[:, chosen_idx].copy()
    for idx, _ in candidates:
        if idx == chosen_idx:
            continue
        merged_defasagem = merged_defasagem.where(
            merged_defasagem.notna(), standardized.iloc[:, idx]
        )

    candidate_positions = {idx for idx, _ in candidates}
    keep_positions = [
        idx for idx in range(standardized.shape[1]) if idx not in candidate_positions
    ]
    result = standardized.iloc[:, keep_positions].copy()

    insert_position = sum(1 for idx in keep_positions if idx < chosen_idx)
    result.insert(insert_position, "Defasagem", merged_defasagem)
    return result


def load_year_sheet(
    file_path: str | Path,
    sheet_name: str,
    year: int,
    **read_excel_kwargs: object,
) -> pd.DataFrame:
    """Read one year/sheet and standardize the Defasagem column name."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, **read_excel_kwargs)
    return standardize_columns(df, year=year)


def load_pede_workbook(
    file_path: str | Path,
    sheets_by_year: Mapping[int, str] | None = None,
    **read_excel_kwargs: object,
) -> dict[int, pd.DataFrame]:
    """Read PEDE sheets and return one standardized DataFrame per year."""
    resolved_mapping = dict(sheets_by_year or DEFAULT_SHEETS_BY_YEAR)
    datasets: dict[int, pd.DataFrame] = {}
    for year, sheet_name in resolved_mapping.items():
        datasets[year] = load_year_sheet(
            file_path=file_path,
            sheet_name=sheet_name,
            year=year,
            **read_excel_kwargs,
        )
    return datasets

