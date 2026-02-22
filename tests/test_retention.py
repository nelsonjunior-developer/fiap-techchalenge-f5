from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.retention import delete_paths, keep_n_newest_dirs, list_files_older_than, run_retention


def _write_file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _set_age_days(path: Path, age_days: int) -> None:
    ts = time.time() - (age_days * 24 * 60 * 60)
    os.utime(path, (ts, ts))


def test_list_files_older_than_ignores_missing_root() -> None:
    paths = list_files_older_than(
        root="this/path/does/not/exist",
        ttl_days=14,
        patterns=("*.jsonl", "*.log"),
    )
    assert paths == []


def test_delete_paths_dry_run_and_execute_skip_protected_and_symlink(tmp_path: Path) -> None:
    file_path = _write_file(tmp_path / "logs" / "online_metrics.jsonl", "abc")
    dir_path = tmp_path / "app" / "model" / "backups" / "20260222T000000Z"
    _write_file(dir_path / "metadata.json", "{}")
    gitkeep = _write_file(tmp_path / "logs" / ".gitkeep", "")
    symlink_path = tmp_path / "logs" / "link.log"
    symlink_created = False
    try:
        symlink_path.symlink_to(file_path)
        symlink_created = True
    except OSError:
        symlink_created = False

    dry_summary = delete_paths(
        [str(file_path), str(dir_path), str(gitkeep), str(symlink_path)],
        dry_run=True,
    )
    assert file_path.exists()
    assert dir_path.exists()
    assert gitkeep.exists()
    assert dry_summary["n_candidate_files"] == 1
    assert dry_summary["n_candidate_dirs"] == 1
    assert dry_summary["n_deleted_files"] == 0
    assert dry_summary["n_deleted_dirs"] == 0
    assert dry_summary["n_skipped_protected"] >= 1
    if symlink_created:
        assert dry_summary["n_skipped_symlinks"] >= 1
        assert symlink_path.exists()

    execute_summary = delete_paths(
        [str(file_path), str(dir_path), str(gitkeep), str(symlink_path)],
        dry_run=False,
    )
    assert not file_path.exists()
    assert not dir_path.exists()
    assert gitkeep.exists()
    assert execute_summary["n_deleted_files"] == 1
    assert execute_summary["n_deleted_dirs"] == 1
    if symlink_created:
        assert execute_summary["n_skipped_symlinks"] >= 1
        assert symlink_path.is_symlink()


def test_keep_n_newest_dirs_keeps_latest_by_name(tmp_path: Path) -> None:
    root = tmp_path / "app" / "model" / "backups"
    root.mkdir(parents=True, exist_ok=True)
    names = [f"20260222T0000{i:02d}Z" for i in range(12)]
    for idx, name in enumerate(names):
        d = root / name
        d.mkdir()
        _write_file(d / "model.joblib", f"dummy-{idx}")
        # Intentionally scramble mtimes; sorting should still prefer timestamp-ish names.
        _set_age_days(d, age_days=(12 - idx))

    dry = keep_n_newest_dirs(str(root), keep=10, dry_run=True)
    assert dry["n_found_dirs"] == 12
    assert dry["n_candidate_dirs"] == 2
    assert (root / names[0]).exists()
    assert (root / names[-1]).exists()

    run = keep_n_newest_dirs(str(root), keep=10, dry_run=False)
    assert run["n_candidate_dirs"] == 2
    remaining = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert len(remaining) == 10
    assert names[0] not in remaining
    assert names[1] not in remaining
    assert names[-1] in remaining


def test_run_retention_policy_dry_run_and_execute(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    artifacts_dir = tmp_path / "artifacts"
    model_backups_dir = tmp_path / "app" / "model" / "backups"
    reference_backups_dir = tmp_path / "app" / "model" / "reference" / "backups"
    model_releases_dir = artifacts_dir / "models" / "releases"

    old_log = _write_file(logs_dir / "online_metrics.jsonl", "event")
    recent_log = _write_file(logs_dir / "app.log", "log")
    _write_file(logs_dir / ".gitkeep", "")
    _set_age_days(old_log, 20)
    _set_age_days(recent_log, 1)

    old_report = _write_file(artifacts_dir / "offline_metrics_2023_2024.json", "{}")
    recent_report = _write_file(artifacts_dir / "model_selection.md", "# keep")
    old_dataset_version = _write_file(
        artifacts_dir / "dataset_versions" / "20200101T000000Z_dataset.xlsx.json",
        "{}",
    )
    old_model_metadata = _write_file(
        model_releases_dir / "2026-02-20T17-20-08Z__abc" / "metadata.json",
        "{}",
    )
    _set_age_days(old_report, 40)
    _set_age_days(recent_report, 2)
    _set_age_days(old_dataset_version, 40)
    _set_age_days(old_model_metadata, 90)

    for idx in range(12):
        d = model_backups_dir / f"20260222T0000{idx:02d}Z"
        d.mkdir(parents=True, exist_ok=True)
        _write_file(d / "model.joblib", "x")
    for idx in range(7):
        d = reference_backups_dir / f"20260222T0100{idx:02d}Z"
        d.mkdir(parents=True, exist_ok=True)
        _write_file(d / "reference.parquet", "x")

    dry = run_retention(
        dry_run=True,
        logs_dir=str(logs_dir),
        artifacts_dir=str(artifacts_dir),
        model_backups_dir=str(model_backups_dir),
        reference_backups_dir=str(reference_backups_dir),
        model_releases_dir=str(model_releases_dir),
        verbose=False,
    )
    assert dry["status"] == "ok"
    assert dry["dry_run"] is True
    assert old_log.exists()
    assert old_report.exists()
    assert old_dataset_version.exists()
    assert old_model_metadata.exists()  # automatic cleanup disabled for models
    assert dry["summary"]["n_candidate_files"] >= 3
    assert dry["summary"]["n_candidate_dirs"] >= 4  # 2 + 2 backup dirs to trim
    assert dry["summary"]["n_deleted_files"] == 0
    assert dry["summary"]["n_deleted_dirs"] == 0
    assert dry["policy"]["model_releases_cleanup_enabled"] is False

    executed = run_retention(
        dry_run=False,
        logs_dir=str(logs_dir),
        artifacts_dir=str(artifacts_dir),
        model_backups_dir=str(model_backups_dir),
        reference_backups_dir=str(reference_backups_dir),
        model_releases_dir=str(model_releases_dir),
        verbose=False,
    )
    assert executed["status"] == "ok"
    assert not old_log.exists()
    assert recent_log.exists()
    assert not old_report.exists()
    assert recent_report.exists()
    assert not old_dataset_version.exists()
    assert old_model_metadata.exists()  # still preserved under artifacts/models/**
    assert len([p for p in model_backups_dir.iterdir() if p.is_dir()]) == 10
    assert len([p for p in reference_backups_dir.iterdir() if p.is_dir()]) == 5
    assert executed["summary"]["n_deleted_files"] >= 3
    assert executed["summary"]["n_deleted_dirs"] >= 4


def test_run_retention_handles_all_missing_dirs(tmp_path: Path) -> None:
    summary = run_retention(
        dry_run=True,
        logs_dir=str(tmp_path / "missing_logs"),
        artifacts_dir=str(tmp_path / "missing_artifacts"),
        model_backups_dir=str(tmp_path / "missing_model_backups"),
        reference_backups_dir=str(tmp_path / "missing_reference_backups"),
        model_releases_dir=str(tmp_path / "missing_releases"),
        verbose=False,
    )
    assert summary["status"] == "ok"
    assert summary["summary"]["n_skipped_missing_dirs"] >= 4
    assert summary["summary"]["n_deleted_files"] == 0
    assert summary["summary"]["n_deleted_dirs"] == 0
