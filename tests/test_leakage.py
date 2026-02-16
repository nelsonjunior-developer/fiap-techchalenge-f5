import pandas as pd
import pytest

from src.leakage import (
    DEFAULT_ALLOWLIST,
    assert_no_leakage,
    detect_leakage_columns,
    detect_leakage_like_columns,
)


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
    with pytest.raises(ValueError, match=r"\[MODEL\] Leakage detected 2023->2024"):
        assert_no_leakage(
            X,
            year_t=2023,
            year_t1=2024,
            include_year_specific=True,
        )


def test_year_specific_disabled_still_flags_semantic_future_columns() -> None:
    X = pd.DataFrame({"INDE 2024": [6.5], "Mat": [7.2]})
    report = detect_leakage_columns(
        X,
        year_t=2023,
        year_t1=2024,
        include_year_specific=False,
    )
    assert report["n_suspect"] == 1
    assert "INDE 2024" in report["suspect_columns"]
    assert "semantic_future_year" in report["reasons"]["INDE 2024"]


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


def test_structural_future_column_is_tolerated_when_all_missing() -> None:
    X = pd.DataFrame({"INDE 2024": [pd.NA, pd.NA], "Mat": [7.2, 8.1]})
    report = detect_leakage_columns(
        X,
        year_t=2023,
        year_t1=2024,
        include_year_specific=True,
        allowlist=DEFAULT_ALLOWLIST,
    )
    assert "INDE 2024" in report["suspect_columns"]
    assert "INDE 2024" in report["structural_suspects"]
    assert report["leakage_real"] == []

    assert_no_leakage(
        X,
        year_t=2023,
        year_t1=2024,
        include_year_specific=True,
        allowlist=DEFAULT_ALLOWLIST,
        context="TRAIN",
        tolerate_structural_missing=True,
    )


def test_future_column_with_signal_raises() -> None:
    X = pd.DataFrame({"INDE 2024": [7.0, pd.NA], "Mat": [7.2, 8.1]})
    with pytest.raises(ValueError, match=r"\[TRAIN\] Leakage detected 2023->2024"):
        assert_no_leakage(
            X,
            year_t=2023,
            year_t1=2024,
            include_year_specific=True,
            allowlist=DEFAULT_ALLOWLIST,
            context="TRAIN",
            tolerate_structural_missing=True,
        )


def test_semantic_future_year_beyond_t1_is_detected() -> None:
    X = pd.DataFrame({"INDE 2024": [7.0], "Mat": [7.2]})
    report = detect_leakage_columns(
        X,
        year_t=2022,
        year_t1=2023,
        include_year_specific=True,
        allowlist=DEFAULT_ALLOWLIST,
    )
    assert "INDE 2024" in report["suspect_columns"]
    assert "semantic_future_year" in report["reasons"]["INDE 2024"]


def test_detect_like_columns_reports_allowed_hits() -> None:
    report = detect_leakage_like_columns(
        columns=["INDE 23", "target", "Pedra 23"],
        year_t=2023,
        year_t1=2024,
        include_year_specific=True,
        extra_blacklist=[r"^INDE\s*23$", r"^Pedra\s*23$"],
        allowlist=DEFAULT_ALLOWLIST,
    )
    assert report["allowed_hits"] == ["INDE 23", "Pedra 23"]
    assert report["suspects"] == ["target"]


def test_returns_only_names_no_values() -> None:
    X = pd.DataFrame({"Defasagem_y": [999], "Mat": [7.5]})
    report = detect_leakage_columns(X)
    assert "suspect_columns" in report
    assert "patterns_used" in report
    assert all(isinstance(name, str) for name in report["suspect_columns"])
    # Ensure cell values are not exposed in report payload.
    assert "999" not in str(report)
