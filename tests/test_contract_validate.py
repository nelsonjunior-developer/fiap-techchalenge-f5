from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.contract_validate import load_year_contract, validate_frame_against_contract


def _mini_contract() -> dict:
    return {
        "year": 2023,
        "columns": {
            "RA": {
                "name": "RA",
                "dtype": "string",
                "presence": "original",
                "pii": True,
                "rules": [
                    {"rule_type": "dtype", "enforcement": "error", "spec": {"expected_dtype": "string"}},
                    {"rule_type": "missing", "enforcement": "error", "spec": {"allow_missing": False}},
                    {"rule_type": "domain", "enforcement": "info", "spec": {"kind": "none"}},
                ],
            },
            "Idade": {
                "name": "Idade",
                "dtype": "Int64",
                "presence": "original",
                "pii": False,
                "rules": [
                    {"rule_type": "dtype", "enforcement": "error", "spec": {"expected_dtype": "Int64"}},
                    {"rule_type": "missing", "enforcement": "error", "spec": {"allow_missing": False}},
                    {
                        "rule_type": "domain",
                        "enforcement": "error",
                        "spec": {"kind": "range", "min": 3, "max": 30},
                    },
                ],
            },
            "Gênero": {
                "name": "Gênero",
                "dtype": "string",
                "presence": "original",
                "pii": False,
                "rules": [
                    {"rule_type": "dtype", "enforcement": "error", "spec": {"expected_dtype": "string"}},
                    {"rule_type": "missing", "enforcement": "error", "spec": {"allow_missing": False}},
                    {
                        "rule_type": "domain",
                        "enforcement": "error",
                        "spec": {"kind": "set", "allowed": ["Feminino", "Masculino"]},
                    },
                ],
            },
            "Data_Nasc": {
                "name": "Data_Nasc",
                "dtype": "datetime64[ns]",
                "presence": "original",
                "pii": False,
                "rules": [
                    {
                        "rule_type": "dtype",
                        "enforcement": "error",
                        "spec": {"expected_dtype": "datetime64[ns]"},
                    },
                    {"rule_type": "missing", "enforcement": "warning", "spec": {"allow_missing": False}},
                    {
                        "rule_type": "domain",
                        "enforcement": "warning",
                        "spec": {
                            "kind": "date_range",
                            "start": "1990-01-01",
                            "end": "2030-12-31",
                        },
                    },
                ],
            },
            "OptionalX": {
                "name": "OptionalX",
                "dtype": "string",
                "presence": "structural_optional",
                "pii": False,
                "rules": [
                    {"rule_type": "dtype", "enforcement": "error", "spec": {"expected_dtype": "string"}},
                    {"rule_type": "missing", "enforcement": "error", "spec": {"allow_missing": False}},
                    {
                        "rule_type": "domain",
                        "enforcement": "error",
                        "spec": {"kind": "set", "allowed": ["A", "B"]},
                    },
                ],
            },
        },
        "metadata": {},
    }


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RA": pd.Series(["1", "2"], dtype="string"),
            "Idade": pd.Series([12, 13], dtype="Int64"),
            "Gênero": pd.Series(["Feminino", "Masculino"], dtype="string"),
            "Data_Nasc": pd.to_datetime(["2011-01-01", "2010-02-03"]),
            "OptionalX": pd.Series([pd.NA, pd.NA], dtype="string"),
        }
    )


def test_contract_validation_pass_basic() -> None:
    report = validate_frame_against_contract(_base_df(), 2023, _mini_contract())
    assert report["status"] == "PASS"
    assert report["errors_count"] == 0


def test_contract_validation_dtype_mismatch_error() -> None:
    df = _base_df().copy()
    df["Idade"] = df["Idade"].astype("string")
    report = validate_frame_against_contract(df, 2023, _mini_contract())
    assert report["status"] == "FAIL"
    assert any(
        f["column"] == "Idade" and f["rule_type"] == "dtype" and f["enforcement"] == "error"
        for f in report["findings"]
    )


def test_contract_validation_missing_violation_error() -> None:
    df = _base_df().copy()
    df.loc[0, "RA"] = pd.NA
    report = validate_frame_against_contract(df, 2023, _mini_contract())
    assert report["status"] == "FAIL"
    assert any(
        f["column"] == "RA" and f["rule_type"] == "missing" and f["enforcement"] == "error"
        for f in report["findings"]
    )


def test_contract_validation_range_violation_error() -> None:
    df = _base_df().copy()
    df.loc[0, "Idade"] = 50
    report = validate_frame_against_contract(df, 2023, _mini_contract())
    assert report["status"] == "FAIL"
    assert any(
        f["column"] == "Idade" and f["kind"] == "range" and f["enforcement"] == "error"
        for f in report["findings"]
    )


def test_contract_validation_set_violation_error() -> None:
    df = _base_df().copy()
    df.loc[0, "Gênero"] = "Outro"
    report = validate_frame_against_contract(df, 2023, _mini_contract())
    assert report["status"] == "FAIL"
    assert any(
        f["column"] == "Gênero" and f["kind"] == "set" and f["enforcement"] == "error"
        for f in report["findings"]
    )


def test_contract_validation_date_range_warning_only() -> None:
    df = _base_df().copy()
    df.loc[0, "Data_Nasc"] = pd.Timestamp("1980-01-01")
    report = validate_frame_against_contract(df, 2023, _mini_contract())
    assert report["status"] == "PASS"
    assert any(
        f["column"] == "Data_Nasc"
        and f["kind"] == "date_range"
        and f["enforcement"] == "warning"
        for f in report["findings"]
    )


def test_structural_optional_missing_does_not_fail() -> None:
    df = _base_df().copy()
    report = validate_frame_against_contract(df, 2023, _mini_contract())
    assert report["status"] == "PASS"
    assert any(
        f["column"] == "OptionalX"
        and f["rule_type"] in {"missing", "domain"}
        and f["enforcement"] == "info"
        for f in report["findings"]
    )


def test_load_year_contract_real_file_contains_expected_columns() -> None:
    contract = load_year_contract(2023, contracts_dir=Path("docs/contracts"))
    assert len(contract["columns"]) == 57
    assert "RA" in contract["columns"]
