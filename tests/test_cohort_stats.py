import json
from pathlib import Path

import pandas as pd
import pytest

import src.cohort_stats as cohort_stats
from src.cohort_stats import (
    compute_intersections,
    compute_ra_sets,
    persist_ra_intersections,
)


def test_compute_intersections_basic_counts_and_percentages() -> None:
    dfs = {
        2022: pd.DataFrame({"RA": ["A", "B", "C"]}),
        2023: pd.DataFrame({"RA": ["B", "C", "D", "E"]}),
        2024: pd.DataFrame({"RA": ["C", "E", "F"]}),
    }

    ra_sets, invalid_counts = compute_ra_sets(dfs)
    report = compute_intersections(ra_sets, invalid_counts)

    pair_22_23 = report["pairs"]["2022_2023"]
    assert pair_22_23["intersection"] == 2
    assert pair_22_23["union"] == 5
    assert pair_22_23["pct_of_2022"] == pytest.approx(2 / 3)
    assert pair_22_23["pct_of_2023"] == pytest.approx(2 / 4)
    assert pair_22_23["jaccard"] == pytest.approx(2 / 5)


def test_compute_ra_sets_discards_null_and_blank_ra() -> None:
    dfs = {
        2022: pd.DataFrame({"RA": [None, " ", " A ", "A"]}),
        2023: pd.DataFrame({"RA": ["A"]}),
        2024: pd.DataFrame({"RA": ["A"]}),
    }

    ra_sets, invalid_counts = compute_ra_sets(dfs)

    assert ra_sets[2022] == {"A"}
    assert invalid_counts[2022] == 2


def test_compute_intersections_handles_zero_division() -> None:
    ra_sets = {2022: set(), 2023: {"A"}, 2024: set()}
    invalid_counts = {2022: 0, 2023: 0, 2024: 0}

    report = compute_intersections(ra_sets, invalid_counts)
    pair_22_23 = report["pairs"]["2022_2023"]
    pair_22_24 = report["pairs"]["2022_2024"]

    assert pair_22_23["pct_of_2022"] == 0.0
    assert pair_22_23["pct_of_2023"] == 0.0
    assert pair_22_23["jaccard"] == 0.0
    assert pair_22_24["union"] == 0
    assert pair_22_24["jaccard"] == 0.0


def test_compute_ra_sets_requires_ra_column() -> None:
    dfs = {
        2022: pd.DataFrame({"SemRA": [1]}),
        2023: pd.DataFrame({"RA": ["A"]}),
        2024: pd.DataFrame({"RA": ["B"]}),
    }

    with pytest.raises(ValueError, match="RA ausente"):
        compute_ra_sets(dfs)


def test_intersection_report_does_not_expose_ra_lists(tmp_path: Path) -> None:
    dfs = {
        2022: pd.DataFrame({"RA": ["A", "B", "C"]}),
        2023: pd.DataFrame({"RA": ["B", "C", "D"]}),
        2024: pd.DataFrame({"RA": ["C", "D", "E"]}),
    }
    ra_sets, invalid_counts = compute_ra_sets(dfs)
    report = compute_intersections(ra_sets, invalid_counts)
    json_path, _ = persist_ra_intersections(report, output_dir=tmp_path, write_markdown=False)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    dumped = json.dumps(payload, ensure_ascii=False)

    for leaked_ra in ("\"A\"", "\"B\"", "\"C\"", "\"D\"", "\"E\""):
        assert leaked_ra not in dumped

    suspicious_keys = {"ras", "ra_list", "students", "ra_values"}
    payload_keys = set(dumped.casefold().replace('"', " ").replace(":", " ").split())
    assert suspicious_keys.isdisjoint(payload_keys)

    # Ensure report keeps aggregate-only structures.
    assert isinstance(payload["counts"], dict)
    assert isinstance(payload["pairs"], dict)
    assert all(isinstance(v, int) for v in payload["counts"].values())


def test_compute_intersections_raises_for_invalid_pair() -> None:
    ra_sets = {2022: {"A"}, 2023: {"A"}, 2024: {"B"}}
    invalid_counts = {2022: 0, 2023: 0, 2024: 0}

    with pytest.raises(ValueError, match="Par inválido"):
        compute_intersections(ra_sets, invalid_counts, pairs=[(2022, 2025)])


def test_persist_ra_intersections_writes_markdown(tmp_path: Path) -> None:
    ra_sets = {2022: {"A"}, 2023: {"A"}, 2024: {"A"}}
    invalid_counts = {2022: 0, 2023: 0, 2024: 0}
    report = compute_intersections(ra_sets, invalid_counts)

    json_path, md_path = persist_ra_intersections(report, output_dir=tmp_path, write_markdown=True)

    assert json_path.exists()
    assert md_path is not None
    assert md_path.exists()
    markdown = md_path.read_text(encoding="utf-8")
    assert "| Par | Interseção | % ano A | % ano B | União | Jaccard |" in markdown


def test_run_from_loaded_data_and_main_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_dfs = {
        2022: pd.DataFrame({"RA": ["A", "B"]}),
        2023: pd.DataFrame({"RA": ["B"]}),
        2024: pd.DataFrame({"RA": ["C"]}),
    }

    def _fake_get_default_dataset_path() -> Path:
        return tmp_path / "fake.xlsx"

    def _fake_load_pede_workbook_with_metadata(path: Path):
        return fake_dfs, {}, {}

    monkeypatch.setattr("src.data.get_default_dataset_path", _fake_get_default_dataset_path)
    monkeypatch.setattr(
        "src.data.load_pede_workbook_with_metadata",
        _fake_load_pede_workbook_with_metadata,
    )

    report = cohort_stats.run_from_loaded_data(output_dir=tmp_path, write_markdown=False)
    assert report["pairs"]["2022_2023"]["intersection"] == 1
    assert (tmp_path / "ra_intersections.json").exists()

    monkeypatch.setattr(
        cohort_stats,
        "run_from_loaded_data",
        lambda output_dir, write_markdown: report,
    )
    monkeypatch.setattr(
        cohort_stats,
        "setup_logging",
        lambda: None,
    )
    monkeypatch.setattr("sys.argv", ["python", "--output-dir", str(tmp_path), "--no-markdown"])
    cohort_stats.main()
