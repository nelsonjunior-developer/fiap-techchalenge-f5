from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_cli_skip_dataset_writes_skipped_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "validate_out"
    result = _run_cli(
        "-m",
        "src.validate",
        "--no-markdown",
        "--skip-dataset",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr

    json_path = output_dir / "data_quality_report.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "SKIPPED"
    assert payload["reason"] == "dataset_not_available_in_ci"
    assert "ran_in_ci_skip_mode" in payload["notes"]
    assert "generated_at" in payload


def test_cohort_stats_cli_skip_dataset_writes_skipped_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "cohort_out"
    result = _run_cli(
        "-m",
        "src.cohort_stats",
        "--no-markdown",
        "--skip-dataset",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr

    json_path = output_dir / "ra_intersections.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "SKIPPED"
    assert payload["reason"] == "dataset_not_available_in_ci"
    assert payload["default_pairs"] == [[2022, 2023], [2023, 2024], [2022, 2024]]
    assert "ran_in_ci_skip_mode" in payload["notes"]
    assert "generated_at" in payload
