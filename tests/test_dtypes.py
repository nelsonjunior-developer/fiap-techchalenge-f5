import pandas as pd

from src.dtypes import standardize_dtypes


def test_standardize_dtypes_data_nasc_disambiguates_year_and_excel_serial() -> None:
    df = pd.DataFrame({"RA": ["1", "2"], "Data_Nasc": [2011, 45000]})

    result, report = standardize_dtypes(df, year=2022)

    assert str(result["Data_Nasc"].dtype).startswith("datetime64")
    assert result.loc[0, "Data_Nasc"] == pd.Timestamp("2011-01-01")
    assert pd.notna(result.loc[1, "Data_Nasc"])
    assert report["data_nasc_sources"]["year"] == 1
    assert report["data_nasc_sources"]["excel_serial"] == 1


def test_standardize_dtypes_data_nasc_string_to_datetime() -> None:
    df = pd.DataFrame({"RA": ["1"], "Data_Nasc": ["2008-05-10"]})

    result, _ = standardize_dtypes(df, year=2023)

    assert result.loc[0, "Data_Nasc"] == pd.Timestamp("2008-05-10")


def test_standardize_dtypes_idade_removes_datetime_values_and_keeps_int64() -> None:
    df = pd.DataFrame(
        {
            "RA": ["1", "2", "3"],
            "Idade": [pd.Timestamp("1900-01-11"), "1900-01-11 00:00:00", "12"],
        }
    )

    result, report = standardize_dtypes(df, year=2024)

    assert str(result["Idade"].dtype) == "Int64"
    assert pd.isna(result.loc[0, "Idade"])
    assert pd.isna(result.loc[1, "Idade"])
    assert result.loc[2, "Idade"] == 12
    assert report["idade"]["datetime_object_to_nan"] == 1
    assert report["idade"]["datetime_string_to_nan"] == 1


def test_standardize_dtypes_replaces_incluir_in_numeric_columns() -> None:
    df = pd.DataFrame({"RA": ["1", "2"], "INDE": ["INCLUIR", "0.55"]})

    result, report = standardize_dtypes(df, year=2023)

    assert str(result["INDE"].dtype) == "Float64"
    assert pd.isna(result.loc[0, "INDE"])
    assert result.loc[1, "INDE"] == 0.55
    assert report["invalid_tokens_replaced"]["INDE"] == 1


def test_standardize_dtypes_strips_categorical_columns_as_string_dtype() -> None:
    df = pd.DataFrame({"RA": ["1", "2"], "Turma": [" A1 ", None]})

    result, _ = standardize_dtypes(df, year=2022)

    assert str(result["Turma"].dtype) == "string"
    assert result.loc[0, "Turma"] == "A1"
    assert pd.isna(result.loc[1, "Turma"])
