"""Utilities to standardize yearly PEDE dataframe dtypes."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.utils import get_logger

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a pandas dependency but keep defensive import
    np = None

_logger = get_logger(__name__)

_DUP_COL_RE = re.compile(r"__dup\d+$")
_NUMERIC_TEXT_RE = re.compile(r"^[+-]?\d+(?:[\.,]\d+)?$")
_YEAR_FIRST_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_AGE_NUMBER_RE = re.compile(r"(\d{1,3})(?:[.,]\d+)?")

_INVALID_NUMERIC_TOKENS = {"INCLUIR"}
_INVALID_AGE_TOKENS = {"INCLUIR", "ALFA", "#N/A", "#DIV/0!", "N/A"}
_INVALID_AGE_TOKENS_CASEFOLD = {token.casefold() for token in _INVALID_AGE_TOKENS}

_INTEGER_BASE_COLUMNS = {
    "Ano ingresso",
    "Nº Av",
    "Defasagem",
    "Idade",
}

_FLOAT_BASE_COLUMNS = {
    "IAA",
    "IAN",
    "IDA",
    "IEG",
    "IPS",
    "IPP",
    "IPV",
    "INDE",
    "INDE 22",
    "INDE 23",
    "INDE 2023",
    "INDE 2024",
    "Mat",
    "Por",
    "Ing",
    "Cg",
    "Cf",
    "Ct",
    "Rec Psicologia",
}

_FORCE_STRING_COLUMNS = {
    "Fase",
    "Fase_Ideal",
}


def _base_column_name(column: str) -> str:
    return _DUP_COL_RE.sub("", column)


def _is_datetime_scalar(value: object) -> bool:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return True
    if np is not None and isinstance(value, np.datetime64):
        return True
    return False


def _normalize_blank_strings(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    string_mask = cleaned.map(lambda value: isinstance(value, str))
    if string_mask.any():
        normalized = cleaned.loc[string_mask].map(lambda value: value.strip())
        cleaned.loc[string_mask] = normalized
        blank_mask = normalized == ""
        if blank_mask.any():
            cleaned.loc[normalized.index[blank_mask]] = pd.NA
    return cleaned


def _convert_numeric_data_nasc(value: float, source_counts: dict[str, int]) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT

    numeric_value = float(value)
    if 1900 <= numeric_value <= 2100 and float(numeric_value).is_integer():
        source_counts["year"] += 1
        return pd.Timestamp(int(numeric_value), 1, 1)

    source_counts["excel_serial"] += 1
    return pd.to_datetime(numeric_value, unit="D", origin="1899-12-30", errors="coerce")


def _coerce_data_nasc_series(series: pd.Series) -> tuple[pd.Series, dict[str, int], int]:
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    source_counts = {
        "year": 0,
        "excel_serial": 0,
        "string": 0,
        "datetime": 0,
    }

    for idx, raw_value in series.items():
        if pd.isna(raw_value):
            continue

        if _is_datetime_scalar(raw_value):
            parsed = pd.to_datetime(raw_value, errors="coerce")
            result.loc[idx] = parsed
            if pd.notna(parsed):
                source_counts["datetime"] += 1
            continue

        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            result.loc[idx] = _convert_numeric_data_nasc(raw_value, source_counts)
            continue

        if isinstance(raw_value, str):
            text_value = raw_value.strip()
            if text_value == "":
                continue

            numeric_candidate = pd.to_numeric(text_value.replace(",", "."), errors="coerce")
            if pd.notna(numeric_candidate):
                result.loc[idx] = _convert_numeric_data_nasc(float(numeric_candidate), source_counts)
                continue

            parsed = _parse_datetime_text(text_value)
            result.loc[idx] = parsed
            if pd.notna(parsed):
                source_counts["string"] += 1
            continue

        parsed = pd.to_datetime(raw_value, errors="coerce")
        result.loc[idx] = parsed
        if pd.notna(parsed):
            source_counts["string"] += 1

    n_nat_data_nasc = int(result.isna().sum())
    return result, source_counts, n_nat_data_nasc


def _is_datetime_like_string(value: object) -> bool:
    if not isinstance(value, str):
        return False

    text = value.strip()
    if text == "":
        return False
    if "1900-01-" in text:
        return True

    numeric_text = text.replace(",", ".")
    if _NUMERIC_TEXT_RE.fullmatch(numeric_text):
        return False

    parsed = _parse_datetime_text(text)
    return pd.notna(parsed)


def _parse_datetime_text(text: str) -> pd.Timestamp | pd.NaT:
    normalized = text.strip()
    if _YEAR_FIRST_DATE_RE.match(normalized):
        return pd.to_datetime(normalized, errors="coerce")
    return pd.to_datetime(normalized, errors="coerce", dayfirst=True)


def _recover_age_from_datetime(value: object, *, year: int | None) -> int | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None

    # In 2023, age came partially serialized as Excel dates (1900-01-DD).
    # Recovering day-of-month avoids large information loss for this specific year.
    if (
        year == 2023
        and parsed.year == 1900
        and parsed.month == 1
        and parsed.hour == 0
        and parsed.minute == 0
        and parsed.second == 0
    ):
        return int(parsed.day)
    return None


def parse_age_series(
    series: pd.Series,
    *,
    year: int | None = None,
) -> tuple[pd.Series, dict[str, int]]:
    """Parse age values robustly while preserving missingness semantics."""
    original = series.copy()
    normalized = _normalize_blank_strings(original)

    parsed = pd.Series(pd.NA, index=series.index, dtype="Int64")
    report = {
        "n_original_non_null": int(original.notna().sum()),
        "n_parsed_numeric_ok": 0,
        "n_invalid_datetime_like": 0,
        "n_invalid_tokens": 0,
        "n_out_of_range": 0,
        "n_recovered_excel_date": 0,
        "n_final_na": 0,
        "n_coerced_to_na": 0,
        "n_invalid_tokens_replaced": 0,
        # Backward-compatible aliases for previous report keys.
        "datetime_object_to_nan": 0,
        "datetime_string_to_nan": 0,
        "non_numeric_to_nan": 0,
        "fractional_to_nan": 0,
    }

    for idx, value in normalized.items():
        if pd.isna(value):
            continue

        age_value: int | None = None

        if _is_datetime_scalar(value):
            recovered = _recover_age_from_datetime(value, year=year)
            if recovered is not None:
                age_value = recovered
                report["n_recovered_excel_date"] += 1
            else:
                report["n_invalid_datetime_like"] += 1
                report["datetime_object_to_nan"] += 1
        elif isinstance(value, str):
            text = value.strip()
            if text == "":
                report["n_invalid_tokens"] += 1
                report["non_numeric_to_nan"] += 1
                continue

            if text.casefold() in _INVALID_AGE_TOKENS_CASEFOLD:
                report["n_invalid_tokens"] += 1
                report["n_invalid_tokens_replaced"] += 1
                report["non_numeric_to_nan"] += 1
                continue

            parsed_dt = pd.to_datetime(text, errors="coerce")
            if pd.notna(parsed_dt):
                recovered = _recover_age_from_datetime(parsed_dt, year=year)
                if recovered is not None:
                    age_value = recovered
                    report["n_recovered_excel_date"] += 1
                else:
                    report["n_invalid_datetime_like"] += 1
                    report["datetime_string_to_nan"] += 1
                    continue
            else:
                match = _AGE_NUMBER_RE.search(text)
                if not match:
                    report["n_invalid_tokens"] += 1
                    report["non_numeric_to_nan"] += 1
                    continue
                age_value = int(match.group(1))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_value = float(value)
            rounded = round(numeric_value)
            if abs(numeric_value - rounded) <= 1e-9:
                age_value = int(rounded)
            else:
                report["fractional_to_nan"] += 1
                continue
        else:
            report["n_invalid_tokens"] += 1
            report["non_numeric_to_nan"] += 1
            continue

        if age_value is None:
            continue

        report["n_parsed_numeric_ok"] += 1
        if age_value < 3 or age_value > 30:
            report["n_out_of_range"] += 1
            continue

        parsed.loc[idx] = age_value

    non_null_original_mask = normalized.notna()
    final_na_non_null = int(parsed[non_null_original_mask].isna().sum())
    report["n_final_na"] = int(parsed.isna().sum())
    report["n_coerced_to_na"] = final_na_non_null
    return parsed.astype("Int64"), report


def _coerce_numeric_series(
    series: pd.Series,
    *,
    as_integer: bool,
) -> tuple[pd.Series, dict[str, int]]:
    original_non_null = int(series.notna().sum())
    cleaned = _normalize_blank_strings(series)

    token_mask = cleaned.map(
        lambda value: isinstance(value, str) and value.strip().upper() in _INVALID_NUMERIC_TOKENS
    )
    invalid_tokens_replaced = int(token_mask.sum())
    cleaned = cleaned.mask(token_mask, pd.NA)

    numeric = pd.to_numeric(cleaned, errors="coerce")
    coerced_to_nan = int((cleaned.notna() & numeric.isna()).sum())

    if as_integer:
        fractional_mask = numeric.notna() & (numeric % 1 != 0)
        coerced_to_nan += int(fractional_mask.sum())
        numeric = numeric.mask(fractional_mask, pd.NA)
        standardized = numeric.astype("Int64")
    else:
        standardized = numeric.astype("Float64")

    return standardized, {
        "n_original_non_null": original_non_null,
        "invalid_tokens_replaced": invalid_tokens_replaced,
        "coerced_to_nan": coerced_to_nan,
    }


def _to_clean_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def standardize_dtypes(
    df: pd.DataFrame,
    year: int,
    logger=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize yearly dataframe dtypes using explicit PEDE data contracts."""
    log = logger or _logger
    standardized = df.copy()
    report: dict[str, Any] = {
        "year": year,
        "coercions": {},
        "invalid_tokens_replaced": {},
        "numeric_columns": {},
        "n_nat_data_nasc": 0,
        "data_nasc_sources": {
            "year": 0,
            "excel_serial": 0,
            "string": 0,
            "datetime": 0,
        },
        "idade": {
            "n_original_non_null": 0,
            "n_parsed_numeric_ok": 0,
            "n_invalid_datetime_like": 0,
            "n_invalid_tokens": 0,
            "n_out_of_range": 0,
            "n_recovered_excel_date": 0,
            "n_final_na": 0,
            "n_coerced_to_na": 0,
            "n_invalid_tokens_replaced": 0,
            "datetime_object_to_nan": 0,
            "datetime_string_to_nan": 0,
            "non_numeric_to_nan": 0,
            "fractional_to_nan": 0,
        },
        "dtypes_final": {},
    }

    if "RA" in standardized.columns:
        standardized["RA"] = _to_clean_string(standardized["RA"])

    for column in list(standardized.columns):
        base_column = _base_column_name(column)

        if base_column == "RA":
            continue

        if base_column == "Data_Nasc":
            converted, source_counts, n_nat = _coerce_data_nasc_series(standardized[column])
            standardized[column] = converted
            report["n_nat_data_nasc"] += n_nat
            for key, value in source_counts.items():
                report["data_nasc_sources"][key] += value
            continue

        if base_column == "Idade":
            converted_idade, idade_report = parse_age_series(
                standardized[column], year=year
            )
            standardized[column] = converted_idade
            report["idade"] = {
                key: report["idade"][key] + idade_report[key]
                for key in report["idade"]
            }
            report["coercions"][column] = idade_report["n_coerced_to_na"]
            report["numeric_columns"][column] = {
                "n_original_non_null": idade_report["n_original_non_null"],
                "n_coerced_to_na": idade_report["n_coerced_to_na"],
                "n_invalid_tokens_replaced": idade_report["n_invalid_tokens_replaced"],
                "dtype_final": str(converted_idade.dtype),
            }
            continue

        if base_column in _INTEGER_BASE_COLUMNS:
            converted, numeric_report = _coerce_numeric_series(
                standardized[column],
                as_integer=True,
            )
            standardized[column] = converted
            report["coercions"][column] = numeric_report["coerced_to_nan"]
            if numeric_report["invalid_tokens_replaced"]:
                report["invalid_tokens_replaced"][column] = numeric_report[
                    "invalid_tokens_replaced"
                ]
            report["numeric_columns"][column] = {
                "n_original_non_null": numeric_report["n_original_non_null"],
                "n_coerced_to_na": numeric_report["coerced_to_nan"],
                "n_invalid_tokens_replaced": numeric_report["invalid_tokens_replaced"],
                "dtype_final": str(converted.dtype),
            }
            continue

        if base_column in _FLOAT_BASE_COLUMNS:
            converted, numeric_report = _coerce_numeric_series(
                standardized[column],
                as_integer=False,
            )
            standardized[column] = converted
            report["coercions"][column] = numeric_report["coerced_to_nan"]
            if numeric_report["invalid_tokens_replaced"]:
                report["invalid_tokens_replaced"][column] = numeric_report[
                    "invalid_tokens_replaced"
                ]
            report["numeric_columns"][column] = {
                "n_original_non_null": numeric_report["n_original_non_null"],
                "n_coerced_to_na": numeric_report["coerced_to_nan"],
                "n_invalid_tokens_replaced": numeric_report["invalid_tokens_replaced"],
                "dtype_final": str(converted.dtype),
            }
            continue

    for column in list(standardized.columns):
        base_column = _base_column_name(column)
        if base_column in _FORCE_STRING_COLUMNS:
            standardized[column] = _to_clean_string(standardized[column])
            continue
        if base_column in _INTEGER_BASE_COLUMNS | _FLOAT_BASE_COLUMNS | {"Data_Nasc", "RA"}:
            continue
        if pd.api.types.is_datetime64_any_dtype(standardized[column]):
            continue
        standardized[column] = _to_clean_string(standardized[column])

    report["dtypes_final"] = {
        column: str(dtype)
        for column, dtype in standardized.dtypes.items()
    }

    dtype_counts = {
        "int": sum(dtype == "Int64" for dtype in report["dtypes_final"].values()),
        "float": sum(dtype == "Float64" for dtype in report["dtypes_final"].values()),
        "string": sum(dtype == "string" for dtype in report["dtypes_final"].values()),
        "datetime": sum(dtype.startswith("datetime64") for dtype in report["dtypes_final"].values()),
    }

    top_coercions = sorted(
        (
            (column, count)
            for column, count in report["coercions"].items()
            if count > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    top_invalid_tokens = sorted(
        report["invalid_tokens_replaced"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    log.info(
        "Dtype standardization year=%d | nat_data_nasc=%d data_nasc_year=%d "
        "data_nasc_excel_serial=%d data_nasc_string=%d idade_original_non_null=%d "
        "idade_parsed_ok=%d idade_recovered_excel_date=%d idade_invalid_datetime_like=%d "
        "idade_invalid_tokens=%d idade_out_of_range=%d idade_final_na=%d "
        "idade_coerced_to_na=%d invalid_tokens_top=%s top_coercions=%s "
        "dtype_counts=int:%d float:%d string:%d datetime:%d",
        year,
        report["n_nat_data_nasc"],
        report["data_nasc_sources"]["year"],
        report["data_nasc_sources"]["excel_serial"],
        report["data_nasc_sources"]["string"],
        report["idade"]["n_original_non_null"],
        report["idade"]["n_parsed_numeric_ok"],
        report["idade"]["n_recovered_excel_date"],
        report["idade"]["n_invalid_datetime_like"],
        report["idade"]["n_invalid_tokens"],
        report["idade"]["n_out_of_range"],
        report["idade"]["n_final_na"],
        report["idade"]["n_coerced_to_na"],
        top_invalid_tokens,
        top_coercions,
        dtype_counts["int"],
        dtype_counts["float"],
        dtype_counts["string"],
        dtype_counts["datetime"],
    )

    return standardized, report


def standardize_dtypes_all(
    dfs: dict[int, pd.DataFrame],
    logger=None,
) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, Any]]]:
    """Apply dtype standardization to all yearly datasets."""
    log = logger or _logger
    standardized: dict[int, pd.DataFrame] = {}
    reports: dict[int, dict[str, Any]] = {}

    for year in sorted(dfs):
        standardized[year], reports[year] = standardize_dtypes(
            dfs[year],
            year=year,
            logger=log,
        )

    return standardized, reports
