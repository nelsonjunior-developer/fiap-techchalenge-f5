from pathlib import Path

import pandas as pd
import pytest

from src.validate import validate_yearly_frames


def test_validate_yearly_frames_flags_duplicate_ra_as_error(tmp_path: Path) -> None:
    dfs = {
        2022: pd.DataFrame(
            {
                "RA": ["1", "1"],
                "Defasagem": [0, 1],
            }
        )
    }
    original_columns = {2022: {"RA", "Defasagem"}}

    report = validate_yearly_frames(
        dfs=dfs,
        original_columns=original_columns,
        strict=False,
        output_dir=tmp_path,
        write_markdown=False,
    )

    assert report["overall"]["status"] == "FAIL"
    assert any("RA possui duplicados" in msg for msg in report["years"]["2022"]["errors"])
    assert (tmp_path / "data_quality_report.json").exists()


def test_validate_yearly_frames_distinguishes_structural_vs_real_missing(
    tmp_path: Path,
) -> None:
    dfs = {
        2022: pd.DataFrame(
            {
                "RA": ["1", "2"],
                "Defasagem": [pd.NA, pd.NA],
                "Coluna_Estrutural": [pd.NA, pd.NA],
            }
        )
    }
    original_columns = {2022: {"RA", "Defasagem"}}

    report = validate_yearly_frames(
        dfs=dfs,
        original_columns=original_columns,
        strict=False,
        output_dir=tmp_path,
        write_markdown=False,
    )

    summary = report["years"]["2022"]["missing_summary"]
    structural_entry = next(item for item in summary if item["column"] == "Coluna_Estrutural")
    assert structural_entry["missing_type"] == "structural_100_missing"
    assert all("Coluna_Estrutural" not in msg for msg in report["years"]["2022"]["errors"])
    assert any("Defasagem" in msg and "100% missing real" in msg for msg in report["years"]["2022"]["errors"])


def test_validate_yearly_frames_uses_original_non_null_denominator_for_coercion(
    tmp_path: Path,
) -> None:
    dfs = {
        2022: pd.DataFrame(
            {
                "RA": [str(i) for i in range(12)],
                "Defasagem": [0] * 12,
                "INDE": [0.1] * 12,
            }
        )
    }
    original_columns = {2022: {"RA", "Defasagem", "INDE"}}
    coercion_report = {
        2022: {
            "numeric_columns": {
                "INDE": {
                    "n_original_non_null": 10,
                    "n_coerced_to_na": 4,
                    "n_invalid_tokens_replaced": 0,
                    "dtype_final": "Float64",
                }
            }
        }
    }

    report = validate_yearly_frames(
        dfs=dfs,
        original_columns=original_columns,
        coercion_report=coercion_report,
        strict=False,
        output_dir=tmp_path,
        write_markdown=False,
    )

    coercion_summary = report["years"]["2022"]["coercion_summary"]
    inde_entry = next(item for item in coercion_summary if item["column"] == "INDE")
    assert inde_entry["coerced_rate"] == 0.4
    assert report["overall"]["status"] == "FAIL"


def test_validate_yearly_frames_strict_mode_behavior(tmp_path: Path) -> None:
    dfs = {2022: pd.DataFrame({"RA": ["1", None], "Defasagem": [0, 1]})}
    original_columns = {2022: {"RA", "Defasagem"}}

    report = validate_yearly_frames(
        dfs=dfs,
        original_columns=original_columns,
        strict=False,
        output_dir=tmp_path / "non_strict",
        write_markdown=False,
    )
    assert report["overall"]["status"] == "FAIL"

    with pytest.raises(RuntimeError, match="validation failed"):
        validate_yearly_frames(
            dfs=dfs,
            original_columns=original_columns,
            strict=True,
            output_dir=tmp_path / "strict",
            write_markdown=False,
        )
