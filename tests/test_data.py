from pathlib import Path

import pandas as pd
import pytest

import src.data as data


def _write_workbook(tmp_path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    workbook_path = tmp_path / "pede_test.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return workbook_path


def test_year_to_sheet_mapping_is_correct() -> None:
    assert data.YEAR_TO_SHEET == {2022: "PEDE2022", 2023: "PEDE2023", 2024: "PEDE2024"}


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

    assert "RA" in result.columns
    assert "Defasagem" in result.columns
    assert " Defasagem " not in result.columns
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


def test_load_year_sheet_applies_standardization(tmp_path: Path) -> None:
    workbook_path = _write_workbook(
        tmp_path,
        {"PEDE2022": pd.DataFrame({"RA": [501], " Defas ": [-4]})},
    )
    result = data.load_year_sheet(
        file_path=workbook_path,
        sheet_name="PEDE2022",
        year=2022,
    )

    assert "Defasagem" in result.columns
    assert "Defas" not in result.columns
    assert result["Defasagem"].tolist() == [-4]


def test_make_target_binary_rule() -> None:
    values = pd.Series([-0.1, 0.0, 2.0], dtype=float)
    result = data.make_target(values)
    assert result.tolist() == [1, 0, 0]
    assert result.dtype == int
    assert result.name == "target"


def test_make_target_rejects_non_numeric_series() -> None:
    values = pd.Series(["-1", "0"], dtype=object)
    with pytest.raises(TypeError, match="série numérica"):
        data.make_target(values)


def test_make_target_rejects_nan_values() -> None:
    values = pd.Series([1.0, None], dtype=float)
    with pytest.raises(ValueError, match="não aceita NaN"):
        data.make_target(values)


def test_make_temporal_pairs_builds_cohort_and_excludes_invalid_target(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df_t = pd.DataFrame(
        {
            "RA": [1, 2, 3, 4],
            "Defasagem": [-1, 0, 1, -2],
            "Nota": [10, 20, 30, 40],
        }
    )
    df_t1 = pd.DataFrame(
        {
            "RA": [1, 2, 3, 9],
            "Defasagem": [-2, None, "INCLUIR", 0],
            "NotaFutura": [100, 200, 300, 900],
        }
    )

    with caplog.at_level("INFO"):
        X, y, ids = data.make_temporal_pairs(df_t, df_t1, year_t=2022, year_t1=2023)

    assert len(X) == len(y) == len(ids) == 1
    assert ids.tolist() == [1]
    assert y.tolist() == [1]
    assert "RA" not in X.columns
    assert "NotaFutura" not in X.columns
    assert "__defasagem_next__" not in X.columns
    assert not any(col.endswith("_x") or col.endswith("_y") for col in X.columns)
    assert set(y.unique().tolist()).issubset({0, 1})
    assert "excluded_missing=1" in caplog.text
    assert "excluded_invalid=1" in caplog.text
    assert "Temporal pairs 2022->2023" in caplog.text
    assert "RA=" not in caplog.text


def test_make_temporal_pairs_raises_when_target_column_is_missing() -> None:
    df_t = pd.DataFrame({"RA": [1], "Defasagem": [0], "Nota": [10]})
    df_t1 = pd.DataFrame({"RA": [1], "OutraColuna": [1]})

    with pytest.raises(ValueError, match="year=2023") as error:
        data.make_temporal_pairs(df_t, df_t1, year_t=2022, year_t1=2023)

    assert "Colunas disponíveis" in str(error.value)


def test_make_temporal_pairs_keeps_only_columns_from_t_when_names_overlap() -> None:
    df_t = pd.DataFrame(
        {
            "RA": [10, 11],
            "Nota": [7.5, 8.0],
            "__defasagem_next__": [111, 222],
        }
    )
    df_t1 = pd.DataFrame(
        {
            "RA": [10, 11],
            "Defasagem": [-1.0, 1.0],
            "Nota": [70, 80],
            "ExtraT1": ["a", "b"],
        }
    )

    X, y, ids = data.make_temporal_pairs(df_t, df_t1, year_t=2023, year_t1=2024)

    assert list(X.columns) == ["Nota", "__defasagem_next__"]
    assert "ExtraT1" not in X.columns
    assert not any(col.endswith("_x") or col.endswith("_y") for col in X.columns)
    assert ids.tolist() == [10, 11]
    assert y.tolist() == [1, 0]


def test_make_temporal_pairs_raises_on_unexpected_merge_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_merge = pd.DataFrame.merge

    def fake_merge(self: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        merged = original_merge(self, *args, **kwargs)
        merged["Mat_y"] = [999] * len(merged)
        return merged

    monkeypatch.setattr(pd.DataFrame, "merge", fake_merge)

    df_t = pd.DataFrame({"RA": [1], "Mat": [7.0]})
    df_t1 = pd.DataFrame({"RA": [1], "Defasagem": [-1.0]})

    with pytest.raises(ValueError, match=r"2022->2023") as error:
        data.make_temporal_pairs(df_t, df_t1, year_t=2022, year_t1=2023)

    assert "extras=['Mat_y']" in str(error.value)


def test_load_pede_workbook_raw_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="DATASET_PATH"):
        data.load_pede_workbook_raw("arquivo_inexistente.xlsx")


def test_load_year_sheet_raw_raises_clear_error_when_sheet_is_missing(tmp_path: Path) -> None:
    workbook_path = _write_workbook(
        tmp_path,
        {"PEDE2022": pd.DataFrame({"RA": [1], "Defas": [-1]})},
    )

    with pytest.raises(ValueError, match="PEDE2023") as error:
        data.load_year_sheet_raw(workbook_path, 2023)

    message = str(error.value)
    assert "year=2023" in message
    assert "PEDE2022" in message


def test_load_pede_workbook_raw_loads_three_years_without_standardization(
    tmp_path: Path,
) -> None:
    workbook_path = _write_workbook(
        tmp_path,
        {
            "PEDE2022": pd.DataFrame({"RA": [1], "Defas": [-1]}),
            "PEDE2023": pd.DataFrame({"RA": [1], "Defasagem": [0]}),
            "PEDE2024": pd.DataFrame({"RA": [1], "Defasagem": [1]}),
        },
    )

    datasets = data.load_pede_workbook_raw(workbook_path)
    assert set(datasets.keys()) == {2022, 2023, 2024}
    assert "Defas" in datasets[2022].columns
    assert "Defasagem" not in datasets[2022].columns


def test_load_pede_workbook_wrapper_keeps_standardization_contract(tmp_path: Path) -> None:
    workbook_path = _write_workbook(
        tmp_path,
        {
            "PEDE2022": pd.DataFrame({"RA": [1], "Defas": [-1], "INDE 22": ["INCLUIR"]}),
            "PEDE2023": pd.DataFrame({"RA": [1], "Defasagem": [0], "INDE 2023": [0.5]}),
            "PEDE2024": pd.DataFrame({"RA": [1], "Defasagem": [1], "INDE 2024": [0.7]}),
        },
    )

    datasets = data.load_pede_workbook(workbook_path)

    assert set(datasets.keys()) == {2022, 2023, 2024}
    for year in (2022, 2023, 2024):
        assert "Defasagem" in datasets[year].columns
        assert str(datasets[year]["RA"].dtype) == "string"
    assert str(datasets[2022]["INDE"].dtype) == "Float64"
    assert datasets[2022]["INDE"].isna().all()
