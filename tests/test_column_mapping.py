from pathlib import Path

import pandas as pd

import src.data as data
import src.schema as schema
from src.column_mapping import harmonize_year_columns


def _write_workbook(tmp_path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    workbook_path = tmp_path / "pede_mapping_test.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return workbook_path


def test_harmonize_year_columns_renames_2022_core_aliases() -> None:
    df = pd.DataFrame(
        {
            "Matem": [8.0],
            "Portug": [7.0],
            "Inglês": [6.0],
            "Defas": [-1],
        }
    )

    result, report = harmonize_year_columns(df, year=2022)

    assert {"Mat", "Por", "Ing", "Defasagem"}.issubset(set(result.columns))
    assert not {"Matem", "Portug", "Inglês", "Defas"}.intersection(set(result.columns))
    assert "Matem" in report["renamed"]
    assert report["renamed"].get("Defas") == "Defasagem"


def test_harmonize_year_columns_keeps_2023_when_already_canonical() -> None:
    df = pd.DataFrame({"Mat": [8.0], "Por": [7.0], "Ing": [6.0], "Defasagem": [0]})

    result, report = harmonize_year_columns(df, year=2023)

    assert list(result.columns) == ["Mat", "Por", "Ing", "Defasagem"]
    assert report["renamed"] == {}
    assert report["collisions"] == []


def test_harmonize_year_columns_merges_duplicate_defasagem_without_loss() -> None:
    df = pd.DataFrame(
        {
            "Defasagem": [pd.NA, -2],
            "Defasagem__dup1": [-1, pd.NA],
        }
    )

    result, report = harmonize_year_columns(df, year=2023)

    assert "Defasagem" in result.columns
    assert "Defasagem__dup1" not in result.columns
    assert result["Defasagem"].tolist() == [-1, -2]
    assert report["merged"]["Defasagem"]["n_sources_found"] == 2


def test_schema_harmonization_keeps_only_expected_dup_suffixes() -> None:
    df = pd.DataFrame(
        {
            "RA": [1],
            "Defasagem": [-1],
            "Defasagem.1": [pd.NA],
            "Ativo/ Inativo": ["Cursando"],
            "Ativo/ Inativo.1": ["Cursando"],
            "INDE 2024": [7.2],
            "Pedra 2024": ["Ametista"],
        }
    )

    result = schema.harmonize_schema_year(df, year=2024)

    assert not any(col.endswith(".1") for col in result.columns)
    allowed_dup_columns = {"Ativo/ Inativo__dup1", "Destaque IPV__dup1"}
    dup_columns = {col for col in result.columns if "__dup" in col}
    assert dup_columns.issubset(allowed_dup_columns)


def test_harmonize_year_columns_report_has_no_cell_values_or_ra() -> None:
    df = pd.DataFrame(
        {
            "RA": ["A001"],
            "Matem": [9.5],
            "Defas": [-2],
        }
    )

    _, report = harmonize_year_columns(df, year=2022)
    dumped = str(report)

    assert "A001" not in dumped
    assert "9.5" not in dumped
    assert "RA" not in dumped


def test_load_pede_workbook_with_metadata_contains_column_mapping_report(
    tmp_path: Path,
) -> None:
    workbook_path = _write_workbook(
        tmp_path,
        {
            "PEDE2022": pd.DataFrame({"RA": [1], "Defas": [-1], "Matem": [8.0], "INDE 22": [7.0]}),
            "PEDE2023": pd.DataFrame({"RA": [1], "Defasagem": [0], "Mat": [7.5], "INDE 2023": [7.2]}),
            "PEDE2024": pd.DataFrame({"RA": [1], "Defasagem": [1], "Mat": [7.8], "INDE 2024": [7.4]}),
        },
    )

    _, metadata, _ = data.load_pede_workbook_with_metadata(workbook_path)

    assert "column_mapping_report" in metadata
    assert set(metadata["column_mapping_report"].keys()) == {2022, 2023, 2024}
    for year in (2022, 2023, 2024):
        report = metadata["column_mapping_report"][year]
        assert {"year", "renamed", "merged", "missing_aliases", "collisions"}.issubset(
            set(report.keys())
        )
