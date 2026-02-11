import pandas as pd
import pytest

import src.schema as schema


def test_harmonize_schema_year_renames_core_columns_2022() -> None:
    df = pd.DataFrame(
        {
            "RA": [1],
            "Defas": [-1],
            "Idade 22": [13],
            "Fase ideal": ["Fase 3"],
            "Matem": [7.0],
            "Portug": [8.0],
            "Inglês": [9.0],
            "INDE 22": [6.5],
            "Pedra 22": ["Topázio"],
            "Nome": ["Aluno A"],
            "Ano nasc": [2011],
        }
    )

    result = schema.harmonize_schema_year(df, year=2022)

    expected_columns = {
        "RA",
        "Defasagem",
        "Idade",
        "Fase_Ideal",
        "Mat",
        "Por",
        "Ing",
        "INDE",
        "Pedra_Ano",
        "Nome_Anon",
        "Data_Nasc",
    }
    assert expected_columns.issubset(set(result.columns))


def test_harmonize_schema_year_uses_inde_fallback_for_2023(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame({"RA": [1], "INDE 23": [5.1], "Pedra 2023": ["Ametista"]})

    with caplog.at_level("INFO"):
        result = schema.harmonize_schema_year(df, year=2023)

    assert result["INDE"].tolist() == [5.1]
    assert "INDE_source=INDE 23" in caplog.text


def test_harmonize_schema_year_uses_pedra_fallback_for_2024(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = pd.DataFrame({"RA": [1], "INDE 2024": [5.9], "Pedra 23": ["Quartzo"]})

    with caplog.at_level("INFO"):
        result = schema.harmonize_schema_year(df, year=2024)

    assert result["Pedra_Ano"].tolist() == ["Quartzo"]
    assert "Pedra_Ano_source=Pedra 23" in caplog.text


def test_harmonize_schema_year_creates_nan_when_no_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = pd.DataFrame({"RA": [1], "Turma": ["A"]})

    with caplog.at_level("WARNING"):
        result = schema.harmonize_schema_year(df, year=2024)

    assert "INDE" in result.columns
    assert "Pedra_Ano" in result.columns
    assert result["INDE"].isna().all()
    assert result["Pedra_Ano"].isna().all()
    assert "INDE source missing" in caplog.text
    assert "Pedra_Ano source missing" in caplog.text


def test_resolve_duplicate_headers_handles_existing_dup_collision() -> None:
    df = pd.DataFrame(
        [[1, "Ativo", "Inativo"]],
        columns=["Ativo/ Inativo", "Ativo/ Inativo.1", "Ativo/ Inativo__dup1"],
    )

    resolved, rename_map = schema.resolve_duplicate_headers(df)

    assert list(resolved.columns) == [
        "Ativo/ Inativo",
        "Ativo/ Inativo__dup2",
        "Ativo/ Inativo__dup1",
    ]
    assert rename_map["Ativo/ Inativo.1"] == "Ativo/ Inativo__dup2"


def test_align_years_returns_same_columns_and_order() -> None:
    df_2022 = pd.DataFrame({"RA": [1], "Matem": [7.0], "Defas": [-1], "INDE 22": [5.0]})
    df_2023 = pd.DataFrame({"RA": [2], "Mat": [8.0], "Defasagem": [0], "INDE 2023": [6.0]})

    aligned = schema.align_years({2022: df_2022, 2023: df_2023}, years=(2022, 2023))

    cols_2022 = list(aligned[2022].columns)
    cols_2023 = list(aligned[2023].columns)
    assert cols_2022 == cols_2023
    assert cols_2022[0] == "RA"
    assert "Mat" in cols_2022
