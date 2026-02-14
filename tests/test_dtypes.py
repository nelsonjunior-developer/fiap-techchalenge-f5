import pandas as pd

from src.dtypes import parse_age_series, standardize_dtypes


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


def test_parse_age_series_parses_common_valid_age_formats() -> None:
    series = pd.Series(["12", " 12 ", "12.0", "12,0", "12 anos"])

    parsed, report = parse_age_series(series, year=2024)

    assert parsed.tolist() == [12, 12, 12, 12, 12]
    assert str(parsed.dtype) == "Int64"
    assert report["n_parsed_numeric_ok"] == 5
    assert report["n_coerced_to_na"] == 0


def test_parse_age_series_handles_invalid_values_as_na() -> None:
    series = pd.Series(
        ["INCLUIR", "ALFA", "1900-01-11 00:00:00", pd.Timestamp("1900-01-11")]
    )

    parsed, report = parse_age_series(series, year=2024)

    assert parsed.isna().all()
    assert report["n_invalid_tokens_replaced"] == 2
    assert report["n_invalid_datetime_like"] == 2
    assert report["n_coerced_to_na"] == 4


def test_parse_age_series_applies_age_plausibility_range() -> None:
    series = pd.Series(["2", "31", "10"])

    parsed, report = parse_age_series(series, year=2024)

    assert pd.isna(parsed.iloc[0])
    assert pd.isna(parsed.iloc[1])
    assert parsed.iloc[2] == 10
    assert report["n_out_of_range"] == 2


def test_standardize_dtypes_idade_keeps_na_and_int64() -> None:
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
    assert report["idade"]["n_invalid_datetime_like"] == 2


def test_standardize_dtypes_idade_recovers_excel_date_artifacts_for_2023() -> None:
    df = pd.DataFrame(
        {
            "RA": ["1", "2", "3"],
            "Idade": [pd.Timestamp("1900-01-11"), "1900-01-09 00:00:00", 12],
        }
    )

    result, report = standardize_dtypes(df, year=2023)

    assert result["Idade"].tolist() == [11, 9, 12]
    assert report["idade"]["n_recovered_excel_date"] == 2


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
