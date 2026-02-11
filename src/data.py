"""Data ingestion helpers for PEDE datasets."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

import pandas as pd
from src.dtypes import standardize_dtypes_all
from src.schema import align_years, harmonize_schema_year, normalize_headers
from src.utils import get_logger

_logger = get_logger(__name__)
_DEFAS_SUFFIX_RE = re.compile(r"\.\d+$")

YEAR_TO_SHEET: dict[int, str] = {
    2022: "PEDE2022",
    2023: "PEDE2023",
    2024: "PEDE2024",
}
DEFAULT_SHEETS_BY_YEAR: dict[int, str] = dict(YEAR_TO_SHEET)
_DEFAULT_DATASET_RELATIVE_PATH = Path(
    "dataset/DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
)


def _ensure_dataset_exists(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Arquivo XLSX não encontrado em '{resolved}'. "
            "Defina DATASET_PATH com o caminho correto do dataset."
        )
    return resolved


def _validate_year(year: int) -> None:
    if year not in YEAR_TO_SHEET:
        raise ValueError(
            f"Ano inválido: {year}. "
            f"Anos suportados: {sorted(YEAR_TO_SHEET)}"
        )


def get_default_dataset_path() -> Path:
    """Resolve default dataset path using DATASET_PATH or project fallback path."""
    env_path = os.getenv("DATASET_PATH")
    candidate = Path(env_path) if env_path else _DEFAULT_DATASET_RELATIVE_PATH
    return _ensure_dataset_exists(candidate)


def _load_sheet_from_workbook(
    path: str | Path,
    year: int,
    sheet_name: str,
    *,
    read_excel_kwargs: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    path_obj = _ensure_dataset_exists(path)
    excel_file = pd.ExcelFile(path_obj, engine="openpyxl")
    available_sheets = excel_file.sheet_names
    if sheet_name not in available_sheets:
        raise ValueError(
            f"Aba esperada '{sheet_name}' ausente para year={year}. "
            f"Abas disponíveis: {available_sheets}"
        )

    parse_kwargs = dict(read_excel_kwargs or {})
    df = excel_file.parse(sheet_name=sheet_name, **parse_kwargs)
    _logger.info(
        "Loaded XLSX sheet | file=%s year=%d sheet=%s rows=%d cols=%d",
        path_obj.name,
        year,
        sheet_name,
        df.shape[0],
        df.shape[1],
    )
    return df


def load_year_sheet_raw(path: str | Path, year: int) -> pd.DataFrame:
    """Read raw year sheet from XLSX without any schema standardization."""
    _validate_year(year)
    sheet_name = YEAR_TO_SHEET[year]
    return _load_sheet_from_workbook(path, year, sheet_name)


def load_pede_workbook_raw(path: str | Path) -> dict[int, pd.DataFrame]:
    """Read raw PEDE sheets (2022/2023/2024) without transformations."""
    path_obj = _ensure_dataset_exists(path)
    datasets: dict[int, pd.DataFrame] = {}
    for year in sorted(YEAR_TO_SHEET):
        datasets[year] = load_year_sheet_raw(path_obj, year)
    return datasets


def standardize_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compatibility wrapper delegating schema harmonization for one year."""
    normalized = normalize_headers(df)

    candidates: list[tuple[int, str]] = []
    for idx, col in enumerate(normalized.columns):
        base_name = _DEFAS_SUFFIX_RE.sub("", str(col).strip()).lower()
        if base_name in {"defas", "defasagem"}:
            candidates.append((idx, base_name))

    if not candidates:
        raise ValueError(
            f"Nenhuma coluna de defasagem encontrada para year={year}. "
            f"Colunas disponíveis: {list(normalized.columns)}"
        )

    preferred_candidates = [item for item in candidates if item[1] == "defasagem"]
    chosen_idx = preferred_candidates[0][0] if preferred_candidates else candidates[0][0]
    merged_defasagem = normalized.iloc[:, chosen_idx].copy()
    for idx, _ in candidates:
        if idx == chosen_idx:
            continue
        merged_defasagem = merged_defasagem.where(
            merged_defasagem.notna(), normalized.iloc[:, idx]
        )

    harmonized = harmonize_schema_year(normalized, year=year, logger=_logger)
    harmonized["Defasagem"] = merged_defasagem.values

    drop_defas_dups = [
        col for col in harmonized.columns if str(col).startswith("Defasagem__dup")
    ]
    if drop_defas_dups:
        harmonized = harmonized.drop(columns=drop_defas_dups)

    return harmonized


def load_year_sheet(
    file_path: str | Path,
    sheet_name: str,
    year: int,
    **read_excel_kwargs: object,
) -> pd.DataFrame:
    """Compatibility wrapper: read sheet then apply standardize_columns."""
    _validate_year(year)
    expected_sheet = YEAR_TO_SHEET[year]
    if sheet_name != expected_sheet:
        raise ValueError(
            f"sheet_name inválido para year={year}: '{sheet_name}'. "
            f"Esperado: '{expected_sheet}'"
        )

    if read_excel_kwargs:
        raw_df = _load_sheet_from_workbook(
            file_path,
            year,
            sheet_name,
            read_excel_kwargs=read_excel_kwargs,
        )
    else:
        raw_df = load_year_sheet_raw(file_path, year)
    return standardize_columns(raw_df, year=year)


def load_year(path: str | Path, year: int) -> pd.DataFrame:
    """Read one year and return standardized columns for downstream usage."""
    raw_df = load_year_sheet_raw(path, year)
    return standardize_columns(raw_df, year=year)


def load_pede_workbook(
    file_path: str | Path,
    sheets_by_year: Mapping[int, str] | None = None,
    **read_excel_kwargs: object,
) -> dict[int, pd.DataFrame]:
    """Compatibility wrapper: load workbook raw then harmonize/align yearly schemas."""
    resolved_mapping = dict(sheets_by_year or YEAR_TO_SHEET)
    path_obj = _ensure_dataset_exists(file_path)

    if resolved_mapping == YEAR_TO_SHEET and not read_excel_kwargs:
        raw_datasets = load_pede_workbook_raw(path_obj)
    else:
        raw_datasets = {}
        for year, sheet_name in resolved_mapping.items():
            _validate_year(year)
            raw_datasets[year] = _load_sheet_from_workbook(
                path_obj,
                year,
                sheet_name,
                read_excel_kwargs=read_excel_kwargs,
            )

    standardized: dict[int, pd.DataFrame] = {}
    for year, df in raw_datasets.items():
        standardized[year] = standardize_columns(df, year=year)
    aligned = align_years(standardized, years=tuple(sorted(standardized)), logger=_logger)
    typed_datasets, _ = standardize_dtypes_all(aligned, logger=_logger)
    return typed_datasets


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
