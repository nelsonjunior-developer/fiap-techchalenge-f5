"""Crosswalk and harmonization utilities for equivalent yearly columns."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

import pandas as pd

# Canonical -> aliases ordered by priority for each year.
# Keep this as the single source of truth for equivalent columns across years.
COLUMN_EQUIVALENCES: dict[str, dict[int, list[str]]] = {
    "Defasagem": {
        2022: ["Defasagem", "Defasagem__dup1", "Defasagem.1", "Defas"],
        2023: ["Defasagem", "Defasagem__dup1", "Defasagem.1", "Defas"],
        2024: ["Defasagem", "Defasagem__dup1", "Defasagem.1", "Defas"],
    },
    "Mat": {
        2022: ["Matem", "Mat"],
        2023: ["Mat"],
        2024: ["Mat"],
    },
    "Por": {
        2022: ["Portug", "Por"],
        2023: ["Por"],
        2024: ["Por"],
    },
    "Ing": {
        2022: ["Inglês", "Ing"],
        2023: ["Ing"],
        2024: ["Ing"],
    },
    "Data_Nasc": {
        2022: ["Ano nasc", "Data_Nasc"],
        2023: ["Data de Nasc", "Data_Nasc"],
        2024: ["Data de Nasc", "Data_Nasc"],
    },
    "Fase_Ideal": {
        2022: ["Fase ideal", "Fase_Ideal"],
        2023: ["Fase Ideal", "Fase_Ideal"],
        2024: ["Fase Ideal", "Fase_Ideal"],
    },
    "Nome_Anon": {
        2022: ["Nome", "Nome_Anon"],
        2023: ["Nome Anonimizado", "Nome_Anon"],
        2024: ["Nome Anonimizado", "Nome_Anon"],
    },
    "Idade": {
        2022: ["Idade 22", "Idade"],
        2023: ["Idade"],
        2024: ["Idade"],
    },
}


def _aliases_for_year(canonical: str, year: int) -> list[str]:
    aliases = list(COLUMN_EQUIVALENCES.get(canonical, {}).get(year, []))
    if canonical not in aliases:
        aliases.append(canonical)
    # Preserve order while removing duplicates.
    return list(OrderedDict.fromkeys(aliases))


def _matched_columns(alias: str, columns: list[str], already_selected: set[str]) -> list[str]:
    """Find columns matching alias, including duplicate suffixes."""
    pattern = re.compile(rf"^{re.escape(alias)}(?:__dup\d+|\.\d+)?$")
    matched: list[str] = []
    for col in columns:
        if col in already_selected:
            continue
        if pattern.fullmatch(col):
            matched.append(col)
    return matched


def harmonize_year_columns(
    df: pd.DataFrame,
    year: int,
    *,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply canonical crosswalk mapping to one yearly dataframe.

    The output keeps only canonical names for mapped concepts. When multiple
    aliases are found, values are merged by priority via `combine_first`.
    """
    harmonized = df.copy()
    report: dict[str, Any] = {
        "year": year,
        "renamed": {},
        "merged": {},
        "missing_aliases": {},
        "collisions": [],
    }

    all_columns_ordered = [str(c) for c in harmonized.columns]
    for canonical in sorted(COLUMN_EQUIVALENCES):
        aliases = _aliases_for_year(canonical, year)
        selected: list[str] = []
        selected_set: set[str] = set()

        # Expand aliases with duplicate suffix variants while preserving alias priority.
        for alias in aliases:
            selected.extend(_matched_columns(alias, all_columns_ordered, selected_set))
            selected_set.update(selected)

        if not selected:
            report["missing_aliases"][canonical] = aliases
            if strict:
                raise ValueError(
                    f"Nenhum alias encontrado para canonical='{canonical}' em year={year}. "
                    f"Aliases esperados: {aliases}. Colunas disponíveis: {all_columns_ordered}"
                )
            continue

        merged = harmonized[selected[0]].copy()
        for source_col in selected[1:]:
            # Prefer first non-null value by priority without creating merge suffix artifacts.
            merged = merged.where(merged.notna(), harmonized[source_col])

        if len(selected) == 1 and selected[0] != canonical:
            report["renamed"][selected[0]] = canonical
        elif len(selected) > 1:
            report["merged"][canonical] = {
                "sources_used": selected,
                "strategy": "combine_first",
                "n_sources_found": len(selected),
            }
            report["collisions"].append(
                {
                    "canonical": canonical,
                    "found": selected,
                    "resolved": "merged",
                }
            )

        harmonized[canonical] = merged

        # Keep only canonical column for mapped aliases to preserve stable schema.
        drop_candidates = [col for col in selected if col != canonical]
        if drop_candidates:
            harmonized = harmonized.drop(columns=drop_candidates, errors="ignore")
            all_columns_ordered = [c for c in all_columns_ordered if c not in drop_candidates]
            if canonical not in all_columns_ordered:
                all_columns_ordered.append(canonical)
        elif canonical not in all_columns_ordered:
            all_columns_ordered.append(canonical)

    return harmonized, report
