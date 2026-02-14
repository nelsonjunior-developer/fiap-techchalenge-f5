from pathlib import Path

import pandas as pd

from src.categories import normalize_categories, normalize_categories_all, normalize_text_series
from src.dtypes import standardize_dtypes_all
from src.validate import validate_yearly_frames


def test_normalize_text_series_is_na_safe() -> None:
    series = pd.Series([pd.NA, None, " Agata "])

    normalized = normalize_text_series(series)

    assert pd.isna(normalized.iloc[0])
    assert pd.isna(normalized.iloc[1])
    assert normalized.iloc[2] == "Agata"


def test_normalize_categories_genero_mapping() -> None:
    df = pd.DataFrame(
        {
            "RA": ["1", "2", "3", "4"],
            "Gênero": ["Menina", "Menino", "Feminino", "Masculino"],
        }
    )

    normalized, _ = normalize_categories(df, year=2022)

    assert normalized["Gênero"].tolist() == [
        "Feminino",
        "Masculino",
        "Feminino",
        "Masculino",
    ]


def test_normalize_categories_instituicao_mapping() -> None:
    df = pd.DataFrame(
        {
            "RA": ["1", "2", "3"],
            "Instituição de ensino": ["Escola Pública", "Publica", "Pública"],
        }
    )

    normalized, _ = normalize_categories(df, year=2023)

    assert normalized["Instituição de ensino"].tolist() == ["Pública", "Pública", "Pública"]


def test_normalize_categories_pedra_mapping() -> None:
    df = pd.DataFrame(
        {
            "RA": ["1", "2", "3", "4"],
            "Pedra_Ano": ["Agata", "Ágata", "INCLUIR", pd.NA],
        }
    )

    normalized, _ = normalize_categories(df, year=2024)

    assert normalized.loc[0, "Pedra_Ano"] == "Ágata"
    assert normalized.loc[1, "Pedra_Ano"] == "Ágata"
    assert pd.isna(normalized.loc[2, "Pedra_Ano"])
    assert pd.isna(normalized.loc[3, "Pedra_Ano"])


def test_normalize_categories_turma_uppercase() -> None:
    df = pd.DataFrame({"RA": ["1", "2"], "Turma": ["7e", " 8D "]})

    normalized, _ = normalize_categories(df, year=2024)

    assert normalized["Turma"].tolist() == ["7E", "8D"]


def test_normalize_categories_fase_conservative() -> None:
    df = pd.DataFrame(
        {
            "RA": ["1", "2", "3", "4", "5"],
            "Fase": ["FASE 2", "Fase 3", "ALFA", "7E", "4M"],
        }
    )

    normalized, _ = normalize_categories(df, year=2024)

    assert normalized["Fase"].tolist() == ["Fase 2", "Fase 3", "ALFA", "7E", "4M"]
    assert str(normalized["Fase"].dtype) == "string"


def test_pipeline_dtypes_categories_and_validate_skip_fase_numeric_coercion(
    tmp_path: Path,
) -> None:
    dfs = {
        2024: pd.DataFrame(
            {
                "RA": ["1", "2", "3"],
                "Defasagem": [-1, 0, 1],
                "Fase": ["FASE 2", "ALFA", "7E"],
                "Fase_Ideal": ["Fase 2 (5° e 6° ano)", "ALFA (1° e 2° ano)", "Fase 3 (7° e 8° ano)"],
                "INDE": ["0.5", "0.7", "0.9"],
            }
        )
    }

    typed, coercion = standardize_dtypes_all(dfs)
    assert str(typed[2024]["Fase"].dtype) == "string"
    assert str(typed[2024]["Fase_Ideal"].dtype) == "string"
    assert "Fase" not in coercion[2024]["numeric_columns"]
    assert "Fase_Ideal" not in coercion[2024]["numeric_columns"]

    normalized, _ = normalize_categories_all(typed)
    assert str(normalized[2024]["Fase"].dtype) == "string"
    assert str(normalized[2024]["Fase_Ideal"].dtype) == "string"

    report = validate_yearly_frames(
        dfs=normalized,
        original_columns={2024: set(normalized[2024].columns)},
        coercion_report=coercion,
        strict=False,
        output_dir=tmp_path,
        write_markdown=False,
    )

    coercion_columns = [item["column"] for item in report["years"]["2024"]["coercion_summary"]]
    assert "Fase" not in coercion_columns
    assert "Fase_Ideal" not in coercion_columns
