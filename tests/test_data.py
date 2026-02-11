import pandas as pd
import pytest

import src.data as data


def test_standardize_columns_renames_defas_for_2022() -> None:
    df = pd.DataFrame(
        {
            "RA": [101, 102],
            "Defas": [-1, 0],
            "OutraColuna": [10, 20],
        }
    )

    result = data.standardize_columns(df, year=2022)

    assert "Defasagem" in result.columns
    assert "Defas" not in result.columns
    assert result["Defasagem"].tolist() == [-1, 0]
    assert result["RA"].tolist() == [101, 102]


def test_standardize_columns_handles_whitespace_in_defasagem_header() -> None:
    df = pd.DataFrame(
        {
            "RA": [201],
            " Defasagem ": [-2],
        }
    )

    result = data.standardize_columns(df, year=2023)

    assert list(result.columns) == ["RA", "Defasagem"]
    assert result["Defasagem"].tolist() == [-2]


def test_standardize_columns_raises_error_when_defas_column_is_missing() -> None:
    df = pd.DataFrame({"RA": [301], "Nome": ["Aluno A"]})

    with pytest.raises(ValueError, match="year=2024") as error:
        data.standardize_columns(df, year=2024)

    assert "Colunas disponíveis" in str(error.value)


def test_standardize_columns_prefers_defasagem_when_multiple_candidates_exist() -> None:
    df = pd.DataFrame(
        {
            "RA": [401, 402],
            "Defas": [-1, -3],
            "Defasagem.1": [None, -2],
            "Outra": [7, 8],
        }
    )

    result = data.standardize_columns(df, year=2022)

    assert list(result.columns).count("Defasagem") == 1
    assert result["Defasagem"].tolist() == [-1, -2]
    assert "Defas" not in result.columns
    assert "Defasagem.1" not in result.columns


def test_load_year_sheet_applies_standardization(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_excel(*args, **kwargs) -> pd.DataFrame:
        assert kwargs["sheet_name"] == "PEDE2022"
        return pd.DataFrame({"RA": [501], " Defas ": [-4]})

    monkeypatch.setattr(data.pd, "read_excel", fake_read_excel)

    result = data.load_year_sheet(
        file_path="dataset/DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx",
        sheet_name="PEDE2022",
        year=2022,
    )

    assert "Defasagem" in result.columns
    assert "Defas" not in result.columns
    assert result["Defasagem"].tolist() == [-4]

