import json
from pathlib import Path

import pandas as pd

import src.data as data
from src.features import (
    add_engineered_features,
    DEFAULT_EXCLUDE_COLUMNS,
    get_engineered_feature_names,
    get_feature_columns,
    persist_feature_split_report,
    split_numeric_categorical_datetime,
)


def test_get_feature_columns_excludes_default_pii_columns() -> None:
    X = pd.DataFrame(
        {
            "Nome_Anon": pd.Series(["a", "b"], dtype="string"),
            "Avaliador1": pd.Series(["x", "y"], dtype="string"),
            "Mat": pd.Series([7.5, 8.0], dtype="Float64"),
        }
    )

    feature_cols = get_feature_columns(X)

    assert "Nome_Anon" in DEFAULT_EXCLUDE_COLUMNS
    assert "Avaliador1" in DEFAULT_EXCLUDE_COLUMNS
    assert "Nome_Anon" not in feature_cols
    assert "Avaliador1" not in feature_cols
    assert feature_cols == ["Mat"]


def test_split_disjoint_and_complete() -> None:
    X = pd.DataFrame(
        {
            "Mat": pd.Series([7.5, 8.0], dtype="Float64"),
            "Idade": pd.Series([11, 12], dtype="Int64"),
            "Gênero": pd.Series(["Feminino", "Masculino"], dtype="string"),
            "FlagAtivo": pd.Series([True, False], dtype="boolean"),
            "Data_Nasc": pd.to_datetime(["2011-01-01", "2012-01-01"]),
        }
    )
    feature_cols = list(X.columns)

    numeric, categorical, datetime_cols, report = split_numeric_categorical_datetime(X, feature_cols)

    assert set(numeric).isdisjoint(categorical)
    assert set(numeric).isdisjoint(datetime_cols)
    assert set(categorical).isdisjoint(datetime_cols)
    assert set(numeric) | set(categorical) | set(datetime_cols) == set(feature_cols)
    assert "FlagAtivo" in categorical
    assert report["n_total_features"] == len(feature_cols)


def test_datetime_classification() -> None:
    X = pd.DataFrame(
        {
            "Data_Nasc": pd.to_datetime(["2008-05-10"]),
            "Mat": pd.Series([5.2], dtype="Float64"),
        }
    )
    feature_cols = list(X.columns)

    _, _, datetime_cols, _ = split_numeric_categorical_datetime(X, feature_cols)

    assert "Data_Nasc" in datetime_cols


def _has_suspicious_id_keys(payload: object) -> bool:
    suspicious_keys = {"ra_list", "student_ids", "ids", "rows"}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in suspicious_keys:
                return True
            if _has_suspicious_id_keys(value):
                return True
        return False
    if isinstance(payload, list):
        return any(_has_suspicious_id_keys(item) for item in payload)
    return False


def _has_numeric_id_list(payload: object) -> bool:
    if isinstance(payload, list):
        if payload and all(isinstance(item, str) and item.strip().isdigit() for item in payload):
            return True
        return any(_has_numeric_id_list(item) for item in payload)
    if isinstance(payload, dict):
        return any(_has_numeric_id_list(value) for value in payload.values())
    return False


def test_privacy_report_no_ids(tmp_path: Path) -> None:
    report = {
        "n_total_features": 3,
        "numeric_cols": ["Mat"],
        "categorical_cols": ["Turma"],
        "datetime_cols": ["Data_Nasc"],
        "excluded_cols": ["Nome_Anon", "Avaliador1", "RA"],
    }
    report_path = tmp_path / "feature_split_report.json"
    persist_feature_split_report(report, path=report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert not _has_suspicious_id_keys(payload)
    assert not _has_numeric_id_list(payload)


def test_make_temporal_pairs_applies_feature_split_and_optional_persist(
    tmp_path: Path,
) -> None:
    df_t = pd.DataFrame(
        {
            "RA": ["1", "2"],
            "Nome_Anon": ["Aluno A", "Aluno B"],
            "Avaliador1": ["Av1", "Av2"],
            "Mat": [7.0, 8.0],
            "Turma": ["7A", "8B"],
        }
    )
    df_t1 = pd.DataFrame(
        {
            "RA": ["1", "2"],
            "Defasagem": [-1, 0],
        }
    )

    no_persist_path = tmp_path / "no_persist.json"
    X_no_persist, _, _ = data.make_temporal_pairs(
        df_t,
        df_t1,
        year_t=2022,
        year_t1=2023,
        persist_feature_split=False,
        feature_split_report_path=no_persist_path,
    )
    assert not no_persist_path.exists()

    persist_path = tmp_path / "feature_split_report.json"
    X, y, ids = data.make_temporal_pairs(
        df_t,
        df_t1,
        year_t=2022,
        year_t1=2023,
        persist_feature_split=True,
        feature_split_report_path=persist_path,
    )

    assert "Nome_Anon" not in X.columns
    assert "Avaliador1" not in X.columns
    assert "feature_split" in X.attrs
    assert X.attrs["feature_split"]["year_t"] == 2022
    assert X.attrs["feature_split"]["year_t1"] == 2023
    assert persist_path.exists()

    payload = json.loads(persist_path.read_text(encoding="utf-8"))
    assert "Nome_Anon" in payload["excluded_cols"]
    assert "Avaliador1" in payload["excluded_cols"]
    assert ids.tolist() == ["1", "2"]
    assert y.tolist() == [1, 0]


def test_add_engineered_features_basic() -> None:
    X = pd.DataFrame(
        {
            "Mat": pd.Series([7.0, 8.0, pd.NA], dtype="Float64"),
            "Por": pd.Series([6.0, 7.5, 9.0], dtype="Float64"),
            "Ing": pd.Series([pd.NA, 8.0, 7.0], dtype="Float64"),
            "Defasagem": pd.Series([-1, 0, 2], dtype="Int64"),
            "Idade": pd.Series([10, 15, pd.NA], dtype="Int64"),
            "IAA": pd.Series([7.0, pd.NA, 8.0], dtype="Float64"),
            "IAN": pd.Series([6.0, 7.0, pd.NA], dtype="Float64"),
            "IDA": pd.Series([5.0, 6.0, 7.0], dtype="Float64"),
            "IEG": pd.Series([8.0, 8.5, 9.0], dtype="Float64"),
            "IPS": pd.Series([7.5, pd.NA, 6.5], dtype="Float64"),
            "IPV": pd.Series([8.0, 8.0, 8.0], dtype="Float64"),
            "INDE": pd.Series([7.2, 7.8, pd.NA], dtype="Float64"),
        }
    )

    X_out, report = add_engineered_features(X, enable_age_bucket=True)
    engineered = get_engineered_feature_names(enable_age_bucket=True)["all"]
    for feature in engineered:
        assert feature in X_out.columns
    assert set(report["features_added"]).issuperset(set(engineered))


def test_strict_false_missing_columns_does_not_break() -> None:
    X = pd.DataFrame(
        {
            "Mat": pd.Series([7.0, pd.NA], dtype="Float64"),
            "Por": pd.Series([8.0, 6.0], dtype="Float64"),
            "Defasagem": pd.Series([-1, 1], dtype="Int64"),
        }
    )
    X_out, report = add_engineered_features(X, strict=False, enable_age_bucket=False)
    assert "avg_grades" in X_out.columns
    assert "missing_indicators_count" not in X_out.columns
    assert report["enable_age_bucket"] is False


def test_no_mutation_when_adding_engineered_features() -> None:
    X = pd.DataFrame(
        {
            "Mat": pd.Series([7.0], dtype="Float64"),
            "Por": pd.Series([8.0], dtype="Float64"),
            "Defasagem": pd.Series([0], dtype="Int64"),
            "Idade": pd.Series([12], dtype="Int64"),
        }
    )
    original_columns = list(X.columns)
    _ = add_engineered_features(X)
    assert list(X.columns) == original_columns


def test_age_bucket_bins() -> None:
    X = pd.DataFrame(
        {
            "Idade": pd.Series([9, 15, pd.NA, 22], dtype="Int64"),
        }
    )
    X_out, _ = add_engineered_features(X, enable_age_bucket=True)
    expected = pd.Series(["07_10", "15_18", pd.NA, "19_plus"], dtype="string")
    assert X_out["age_bucket"].fillna("<NA>").tolist() == expected.fillna("<NA>").tolist()


def test_report_contains_only_names() -> None:
    X = pd.DataFrame(
        {
            "Mat": pd.Series([7.0], dtype="Float64"),
            "Por": pd.Series([8.0], dtype="Float64"),
            "Defasagem": pd.Series([0], dtype="Int64"),
        }
    )
    _, report = add_engineered_features(X)
    assert all(isinstance(item, str) for item in report["features_added"])
    assert all(isinstance(item, str) for item in report["base_columns_used"])
