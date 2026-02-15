import json
from pathlib import Path

import pandas as pd
import pytest

from src.features import DEFAULT_EXCLUDE_COLUMNS, get_feature_columns
from src.imputation import (
    DEFAULT_MISSING_POLICY,
    build_imputation_plan,
    find_all_missing_columns,
    make_imputers,
    persist_imputation_plan,
)

ALL_MISSING_2022_2023 = [
    "Ativo/ Inativo",
    "Ativo/ Inativo__dup1",
    "Destaque IPV__dup1",
    "Escola",
    "INDE 2023",
    "INDE 2024",
    "INDE 23",
    "IPP",
    "Pedra 2023",
    "Pedra 2024",
    "Pedra 23",
    "Rec Psicologia",
]


def test_find_all_missing_columns_returns_sorted_list() -> None:
    X = pd.DataFrame(
        {
            "b_col": [pd.NA, pd.NA],
            "a_col": [1.0, pd.NA],
            "c_col": [pd.NA, pd.NA],
        }
    )

    result = find_all_missing_columns(X, ["b_col", "a_col", "c_col"])

    assert result == ["b_col", "c_col"]


def test_build_imputation_plan_drops_real_all_missing_columns_and_excludes_datetime() -> None:
    X_train = pd.DataFrame(
        {
            "Mat": pd.Series([7.0, 8.0, pd.NA], dtype="Float64"),
            "INDE": pd.Series([6.0, 5.5, pd.NA], dtype="Float64"),
            "Idade": pd.Series([11, 12, 13], dtype="Int64"),
            "Turma": pd.Series(["7A", "8B", "9C"], dtype="string"),
            "Gênero": pd.Series(["Feminino", "Masculino", "Feminino"], dtype="string"),
            "Data_Nasc": pd.to_datetime(["2011-01-01", "2012-01-01", "2013-01-01"]),
            "INDE 2023": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Float64"),
            "INDE 2024": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Float64"),
            "INDE 23": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Float64"),
            "IPP": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Float64"),
            "Rec Psicologia": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Float64"),
            "Ativo/ Inativo": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
            "Ativo/ Inativo__dup1": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
            "Destaque IPV__dup1": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
            "Escola": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
            "Pedra 2023": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
            "Pedra 2024": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
            "Pedra 23": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
        }
    )

    numeric_cols = ["Mat", "INDE", "Idade", "INDE 2023", "INDE 2024", "INDE 23", "IPP", "Rec Psicologia"]
    categorical_cols = [
        "Turma",
        "Gênero",
        "Ativo/ Inativo",
        "Ativo/ Inativo__dup1",
        "Destaque IPV__dup1",
        "Escola",
        "Pedra 2023",
        "Pedra 2024",
        "Pedra 23",
    ]
    datetime_cols = ["Data_Nasc"]

    plan = build_imputation_plan(
        X_train=X_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        policy={"drop_all_missing_columns": True},
    )

    assert plan["datetime_cols_excluded"] == ["Data_Nasc"]
    assert set(plan["dropped_all_missing_cols"]) == set(ALL_MISSING_2022_2023)
    for column in ALL_MISSING_2022_2023:
        assert column not in plan["numeric_cols_used"]
        assert column not in plan["categorical_cols_used"]
    assert plan["numeric_cols_used"] == sorted(plan["numeric_cols_used"])
    assert plan["categorical_cols_used"] == sorted(plan["categorical_cols_used"])
    assert "add_indicator=True increases transformed feature dimensionality after fit." in plan["notes"]


def test_build_imputation_plan_requires_disjoint_feature_families() -> None:
    X_train = pd.DataFrame({"Mat": pd.Series([1.0, 2.0], dtype="Float64")})

    with pytest.raises(ValueError, match="não são disjuntas"):
        build_imputation_plan(
            X_train=X_train,
            numeric_cols=["Mat"],
            categorical_cols=["Mat"],
            datetime_cols=[],
        )


def test_feature_count_after_default_pii_exclusion_matches_context() -> None:
    excluded = sorted(DEFAULT_EXCLUDE_COLUMNS)
    assert len(excluded) == 7

    all_columns = [f"f_{idx}" for idx in range(49)] + excluded
    X = pd.DataFrame([list(range(len(all_columns)))], columns=all_columns)

    feature_cols = get_feature_columns(X)

    assert len(all_columns) == 56
    assert len(feature_cols) == 49


def test_make_imputers_uses_plan_policy() -> None:
    plan = {
        "policy": {
            **DEFAULT_MISSING_POLICY,
            "numeric_strategy": "median",
            "categorical_strategy": "most_frequent",
            "add_missing_indicators": True,
        }
    }

    numeric_imputer, categorical_imputer = make_imputers(plan)

    assert numeric_imputer.strategy == "median"
    assert categorical_imputer.strategy == "most_frequent"
    assert numeric_imputer.add_indicator is True
    assert categorical_imputer.add_indicator is True


def _has_suspicious_keys(payload: object) -> bool:
    suspicious_keys = {"rows", "values", "ra_list", "student_ids", "ids"}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in suspicious_keys:
                return True
            if _has_suspicious_keys(value):
                return True
    elif isinstance(payload, list):
        for value in payload:
            if _has_suspicious_keys(value):
                return True
    return False


def _has_probable_id_lists(payload: object) -> bool:
    if isinstance(payload, list):
        if len(payload) > 20 and all(
            isinstance(item, str) and item.strip().isdigit() for item in payload
        ):
            return True
        return any(_has_probable_id_lists(item) for item in payload)
    if isinstance(payload, dict):
        return any(_has_probable_id_lists(value) for value in payload.values())
    return False


def test_privacy_of_persisted_imputation_plan(tmp_path: Path) -> None:
    plan = {
        "policy": DEFAULT_MISSING_POLICY,
        "numeric_cols_used": ["Mat", "INDE"],
        "categorical_cols_used": ["Turma", "Gênero"],
        "datetime_cols_excluded": ["Data_Nasc"],
        "dropped_all_missing_cols": ["Rec Psicologia"],
        "excluded_cols": ["Nome_Anon"],
        "counts": {
            "n_numeric_used": 2,
            "n_categorical_used": 2,
            "n_dropped_all_missing": 1,
            "n_datetime_excluded": 1,
        },
        "notes": ["safe report"],
    }
    report_path = tmp_path / "imputation_plan.json"
    persist_imputation_plan(plan, path=report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert not _has_suspicious_keys(payload)
    assert not _has_probable_id_lists(payload)

