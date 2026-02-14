"""Textual category normalization utilities for yearly PEDE datasets."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_logger

_logger = get_logger(__name__)

CATEGORY_COLUMNS: tuple[str, ...] = (
    "Gênero",
    "Instituição de ensino",
    "Escola",
    "Ativo/ Inativo",
    "Ativo/ Inativo__dup1",
    "Turma",
    "Indicado",
    "Atingiu PV",
    "Fase",
    "Fase_Ideal",
    "Pedra_Ano",
    "Pedra 20",
    "Pedra 21",
    "Pedra 22",
    "Pedra 23",
    "Pedra 2023",
    "Pedra 2024",
)

_PEDRA_COLUMNS: tuple[str, ...] = (
    "Pedra_Ano",
    "Pedra 20",
    "Pedra 21",
    "Pedra 22",
    "Pedra 23",
    "Pedra 2023",
    "Pedra 2024",
)

_FASE_RE = re.compile(r"^FASE\s*([1-8])$", flags=re.IGNORECASE)


def normalize_text_series(series: pd.Series) -> pd.Series:
    """Normalize textual values while preserving missing values as NA."""
    normalized = series.astype("string")
    normalized = normalized.str.normalize("NFC")
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()
    normalized = normalized.mask(normalized == "", pd.NA)
    return normalized


def _casefold_map(series: pd.Series, mapping: dict[str, str | pd.NA]) -> pd.Series:
    normalized = normalize_text_series(series)

    def transform(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        text = str(value)
        return mapping.get(text.casefold(), text)

    transformed = normalized.map(transform)
    return transformed.astype("string")


def _normalize_genero(series: pd.Series) -> pd.Series:
    return _casefold_map(
        series,
        {
            "menina": "Feminino",
            "menino": "Masculino",
            "feminino": "Feminino",
            "masculino": "Masculino",
        },
    )


def _normalize_instituicao(series: pd.Series) -> pd.Series:
    return _casefold_map(
        series,
        {
            "escola pública": "Pública",
            "publica": "Pública",
            "pública": "Pública",
            "privada - programa de apadrinhamento": "Privada - Programa de Apadrinhamento",
        },
    )


def _normalize_pedra(series: pd.Series) -> pd.Series:
    return _casefold_map(
        series,
        {
            "agata": "Ágata",
            "ágata": "Ágata",
            "incluir": pd.NA,
        },
    )


def _normalize_turma(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    return normalized.str.upper()


def _normalize_fase(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)

    def transform(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        text = str(value)
        if text.casefold() == "alfa":
            return "ALFA"
        match = _FASE_RE.fullmatch(text)
        if match:
            return f"Fase {match.group(1)}"
        return text

    transformed = normalized.map(transform)
    return transformed.astype("string")


def _normalize_fase_ideal(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    normalized = normalized.str.replace("°", "º", regex=False)
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()
    return normalized


def _top_counts(series: pd.Series, limit: int = 10) -> list[dict[str, int | str]]:
    counts = series.astype("string").value_counts(dropna=False).head(limit)
    result: list[dict[str, int | str]] = []
    for label, count in counts.items():
        item_label = "<NA>" if pd.isna(label) else str(label)
        result.append({"label": item_label, "count": int(count)})
    return result


def _count_changed(before: pd.Series, after: pd.Series) -> int:
    before_s = before.astype("string")
    after_s = after.astype("string")
    equal = (before_s == after_s) | (before_s.isna() & after_s.isna())
    return int((~equal.fillna(False)).sum())


def normalize_categories(
    df: pd.DataFrame,
    year: int,
    logger=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize configured category columns for a single year dataframe."""
    log = logger or _logger
    normalized_df = df.copy()
    columns_report: dict[str, dict[str, Any]] = {}
    total_changed = 0

    for column in CATEGORY_COLUMNS:
        if column not in normalized_df.columns:
            continue

        before = normalized_df[column].copy()

        if column == "Gênero":
            after = _normalize_genero(before)
        elif column == "Instituição de ensino":
            after = _normalize_instituicao(before)
        elif column in _PEDRA_COLUMNS:
            after = _normalize_pedra(before)
        elif column == "Turma":
            after = _normalize_turma(before)
        elif column == "Fase":
            after = _normalize_fase(before)
        elif column == "Fase_Ideal":
            after = _normalize_fase_ideal(before)
        else:
            after = normalize_text_series(before)

        normalized_df[column] = after

        n_changed = _count_changed(before, after)
        total_changed += n_changed
        columns_report[column] = {
            "n_changed": n_changed,
            "top_before": _top_counts(before, limit=10),
            "top_after": _top_counts(after, limit=10),
        }

    year_report = {
        "year": year,
        "total_changed": total_changed,
        "columns": columns_report,
    }

    top_changed = sorted(
        ((col, info["n_changed"]) for col, info in columns_report.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    log.info(
        "Category normalization year=%d | columns=%d total_changed=%d top_changed=%s",
        year,
        len(columns_report),
        total_changed,
        top_changed,
    )

    return normalized_df, year_report


def normalize_categories_all(
    dfs: dict[int, pd.DataFrame],
    logger=None,
) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, Any]]]:
    """Normalize category columns for all yearly dataframes."""
    log = logger or _logger
    normalized: dict[int, pd.DataFrame] = {}
    report: dict[int, dict[str, Any]] = {}

    for year in sorted(dfs):
        normalized[year], report[year] = normalize_categories(
            dfs[year],
            year=year,
            logger=log,
        )

    return normalized, report


def _build_markdown(report: dict[int, dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Relatorio de Normalizacao de Categorias")
    lines.append("")

    for year in sorted(report):
        year_report = report[year]
        lines.append(f"## Ano {year}")
        lines.append(f"- Total alteracoes: `{year_report['total_changed']}`")

        top_changed = sorted(
            (
                (column, details["n_changed"])
                for column, details in year_report["columns"].items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        lines.append("- Top colunas alteradas:")
        for column, count in top_changed:
            lines.append(f"  - `{column}`: {count}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def persist_category_normalization_report(
    report: dict[int, dict[str, Any]],
    output_dir: str | Path = "artifacts",
    write_markdown: bool = False,
) -> Path:
    """Persist category normalization report to artifacts directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "years": report,
    }

    json_path = output_path / "category_normalization_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_markdown:
        md_path = output_path / "category_normalization_report.md"
        md_path.write_text(_build_markdown(report), encoding="utf-8")

    return json_path
