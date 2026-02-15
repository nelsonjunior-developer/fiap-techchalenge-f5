import pandas as pd
import pytest

from src.leakage import DEFAULT_ALLOWLIST, assert_no_leakage, detect_leakage_columns


def test_detects_merge_suffixes() -> None:
    X = pd.DataFrame({"Mat": [7.0], "Defasagem_y": [1.0]})
    report = detect_leakage_columns(X)
    assert "Defasagem_y" in report["suspect_columns"]
    assert report["n_suspect"] == 1


def test_detects_t1_marker() -> None:
    X = pd.DataFrame({"INDE": [7.0], "Defasagem_t1": [-1.0]})
    report = detect_leakage_columns(X)
    assert "Defasagem_t1" in report["suspect_columns"]


def test_detects_year_specific_when_given() -> None:
    X = pd.DataFrame({"INDE 2024": [6.5], "Mat": [7.2]})
    with pytest.raises(ValueError, match=r"Leakage detected 2023->2024"):
        assert_no_leakage(
            X,
            year_t=2023,
            year_t1=2024,
            include_year_specific=True,
        )


def test_year_specific_disabled_does_not_flag_year_named_columns() -> None:
    X = pd.DataFrame({"INDE 2024": [6.5], "Mat": [7.2]})
    report = detect_leakage_columns(
        X,
        year_t=2023,
        year_t1=2024,
        include_year_specific=False,
    )
    assert report["n_suspect"] == 0


def test_allowlist_prevents_false_positive_for_historical_columns() -> None:
    X = pd.DataFrame({"INDE 23": [7.2], "Pedra 23": ["Quartzo"]})
    report = detect_leakage_columns(
        X,
        year_t=2023,
        year_t1=2024,
        include_year_specific=True,
        allowlist=DEFAULT_ALLOWLIST,
    )
    assert report["n_suspect"] == 0
    assert report["suspect_columns"] == []


def test_returns_only_names_no_values() -> None:
    X = pd.DataFrame({"Defasagem_y": [999], "Mat": [7.5]})
    report = detect_leakage_columns(X)
    assert "suspect_columns" in report
    assert "patterns_used" in report
    assert all(isinstance(name, str) for name in report["suspect_columns"])
    # Ensure cell values are not exposed in report payload.
    assert "999" not in str(report)
