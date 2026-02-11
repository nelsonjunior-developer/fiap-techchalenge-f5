"""Data ingestion helpers for PEDE datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import pandas as pd
from src.utils import get_logger

_SPACES_RE = re.compile(r"\s+")
_DUPLICATE_SUFFIX_RE = re.compile(r"\.\d+$")
_logger = get_logger(__name__)

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


def make_target(defasagem_next: pd.Series) -> pd.Series:
    """Build binary target from next-year Defasagem using rule Defasagem < 0."""
    if not pd.api.types.is_numeric_dtype(defasagem_next):
        raise TypeError(
            "make_target exige série numérica. "
            "Converta tokens inválidos para NaN antes de chamar a função."
        )
    if defasagem_next.isna().any():
        raise ValueError(
            "make_target não aceita NaN. "
            "Remova pares com target ausente/inválido antes de calcular y."
        )

    target = (defasagem_next < 0).astype(int)
    target.name = "target"
    return target


def _is_blank_text(value: object) -> bool:
    return isinstance(value, str) and value.strip() == ""


def make_temporal_pairs(
    df_t: pd.DataFrame,
    df_t1: pd.DataFrame,
    year_t: int,
    year_t1: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create temporal pairs X(t) -> y(t+1) with inner cohort by RA."""
    required_cols_t = {"RA"}
    required_cols_t1 = {"RA", "Defasagem"}

    missing_t = required_cols_t - set(df_t.columns)
    if missing_t:
        raise ValueError(
            f"Colunas obrigatórias ausentes em year={year_t}: {sorted(missing_t)}. "
            f"Colunas disponíveis: {list(df_t.columns)}"
        )

    missing_t1 = required_cols_t1 - set(df_t1.columns)
    if missing_t1:
        raise ValueError(
            f"Colunas obrigatórias ausentes em year={year_t1}: {sorted(missing_t1)}. "
            f"Colunas disponíveis: {list(df_t1.columns)}"
        )

    feature_cols_t = [col for col in df_t.columns if col != "RA"]

    next_target_col = "__defasagem_next__"
    while next_target_col in df_t.columns:
        next_target_col = f"_{next_target_col}"

    t1_target = df_t1[["RA", "Defasagem"]].rename(columns={"Defasagem": next_target_col})
    cohort_pairs = df_t.merge(t1_target, on="RA", how="inner")

    expected_after_merge = set(df_t.columns) | {next_target_col}
    observed_after_merge = set(cohort_pairs.columns)
    if observed_after_merge != expected_after_merge:
        extras = sorted(observed_after_merge - expected_after_merge)
        missing = sorted(expected_after_merge - observed_after_merge)
        raise ValueError(
            f"Merge gerou colunas inesperadas em {year_t}->{year_t1}. "
            f"extras={extras}, missing={missing}"
        )

    raw_target = cohort_pairs[next_target_col]
    missing_mask = raw_target.isna() | raw_target.map(_is_blank_text)
    numeric_target = pd.to_numeric(raw_target, errors="coerce")
    invalid_mask = (~missing_mask) & numeric_target.isna()
    valid_mask = numeric_target.notna()

    total_pairs = len(cohort_pairs)
    missing_count = int(missing_mask.sum())
    invalid_count = int(invalid_mask.sum())
    valid_pairs = int(valid_mask.sum())

    filtered = cohort_pairs.loc[valid_mask].copy()
    ids = filtered["RA"].copy()
    ids.name = "RA"

    y = make_target(numeric_target.loc[valid_mask])
    X = filtered.loc[:, feature_cols_t].copy()

    if "RA" in X.columns:
        raise ValueError(f"RA não pode estar presente em X como feature em {year_t}->{year_t1}.")
    if next_target_col in X.columns:
        raise ValueError(
            f"X contém coluna futura do target em {year_t}->{year_t1}: {next_target_col}"
        )
    if list(X.columns) != feature_cols_t:
        raise ValueError(f"X não preservou apenas colunas de year_t em {year_t}->{year_t1}.")
    leaked_t1_only_cols = [col for col in X.columns if col in df_t1.columns and col not in df_t.columns]
    if leaked_t1_only_cols:
        raise ValueError(
            f"X contém colunas exclusivas de year_t1 em {year_t}->{year_t1}: {leaked_t1_only_cols}"
        )
    merge_suffix_cols = [
        col
        for col in X.columns
        if col.endswith("_x") or col.endswith("_y") or col.endswith("_t1")
    ]
    if merge_suffix_cols:
        raise ValueError(
            f"X contém sufixos de merge inesperados em {year_t}->{year_t1}: {merge_suffix_cols}"
        )
    if len(X) != len(y):
        raise ValueError("Inconsistência: len(X) difere de len(y).")
    unique_target_values = set(y.unique().tolist())
    if not unique_target_values.issubset({0, 1}):
        raise ValueError(f"Target inválido; valores encontrados: {sorted(unique_target_values)}")

    prevalence = float(y.mean()) if len(y) else 0.0
    _logger.info(
        "Temporal pairs %s->%s | total_cohort=%d valid=%d excluded_missing=%d "
        "excluded_invalid=%d prevalence=%.4f",
        year_t,
        year_t1,
        total_pairs,
        valid_pairs,
        missing_count,
        invalid_count,
        prevalence,
    )

    return X, y, ids
