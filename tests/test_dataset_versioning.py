from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset_versioning import (
    compute_file_sha256,
    get_dataset_fingerprint,
    persist_dataset_version,
    persist_dataset_version_event,
    safe_path_hint,
)


def test_compute_file_sha256_is_deterministic_for_known_content(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.xlsx"
    dataset_path.write_bytes(b"abc")

    digest = compute_file_sha256(dataset_path)

    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_get_dataset_fingerprint_returns_expected_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "folder" / "dataset_v1.xlsx"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_bytes(b"payload")

    fingerprint = get_dataset_fingerprint(dataset_path)

    assert fingerprint["path_hint"] == "dataset_v1.xlsx"
    assert fingerprint["basename"] == "dataset_v1.xlsx"
    assert fingerprint["bytes"] == len(b"payload")
    assert isinstance(fingerprint["mtime_utc"], str)
    assert len(fingerprint["sha256"]) == 64


def test_persist_dataset_version_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "artifacts" / "dataset_versions" / "dataset_version.json"
    payload = {
        "generated_at": "2026-02-20T12:00:00+00:00",
        "context": "unit_test",
        "dataset": {
            "path_hint": "dataset.xlsx",
            "basename": "dataset.xlsx",
            "bytes": 123,
            "mtime_utc": "2026-02-20T12:00:00+00:00",
            "sha256": "0" * 64,
        },
    }

    persist_dataset_version(payload, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == payload


def test_persist_dataset_version_event_creates_timestamped_file(tmp_path: Path) -> None:
    fingerprint = {
        "path_hint": "dataset.xlsx",
        "basename": "dataset.xlsx",
        "bytes": 456,
        "mtime_utc": "2026-02-20T12:00:00+00:00",
        "sha256": "f" * 64,
    }

    out_path = persist_dataset_version_event(
        context="train_baseline",
        dataset_fingerprint=fingerprint,
        output_dir=tmp_path / "artifacts" / "dataset_versions",
    )

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["context"] == "train_baseline"
    assert payload["dataset"]["sha256"] == "f" * 64
    assert out_path.name.endswith("_dataset.xlsx.json")


def test_safe_path_hint_returns_basename_only() -> None:
    assert safe_path_hint("/tmp/sensitive/local/path/dataset.xlsx") == "dataset.xlsx"
    assert safe_path_hint(Path("nested/dir/dataset.xlsx")) == "dataset.xlsx"


def test_compute_file_sha256_missing_path_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        compute_file_sha256(tmp_path / "missing.xlsx")
