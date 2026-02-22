"""Local retention/cleanup policy for logs, reports, and rollback backups.

Privacy note: this module never inspects file contents. It operates only on filesystem
metadata (path, mtime, size, type).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_SECONDS_PER_DAY = 24 * 60 * 60
_PROTECTED_FILENAMES = {".gitkeep"}


def _parse_bool_flag(value: int) -> bool:
    return bool(int(value))


def _safe_rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _safe_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _is_protected_file(path: Path) -> bool:
    return path.name in _PROTECTED_FILENAMES


def _estimate_path_bytes(path: Path) -> int:
    """Estimate reclaimable bytes using metadata only (no file content reads)."""
    if not path.exists() or path.is_symlink():
        return 0
    if path.is_file():
        return _safe_size(path)
    if not path.is_dir():
        return 0

    total = 0
    for root, dirnames, filenames in os.walk(path, topdown=True, followlinks=False):
        root_path = Path(root)
        kept_dirnames: list[str] = []
        for dirname in dirnames:
            child = root_path / dirname
            if child.is_symlink():
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames
        for filename in filenames:
            child = root_path / filename
            if child.is_symlink():
                continue
            total += _safe_size(child)
    return int(total)


def list_files_older_than(
    root: str,
    ttl_days: int,
    patterns: tuple[str, ...],
) -> list[str]:
    """List matching files older than TTL using `mtime`.

    Missing roots are ignored (warning only).
    """
    if int(ttl_days) < 0:
        raise ValueError("ttl_days must be >= 0")

    root_path = Path(root)
    if not root_path.exists():
        _logger.warning("retention root missing (skip) | root=%s", root)
        return []
    if not root_path.is_dir():
        _logger.warning("retention root is not a directory (skip) | root=%s", root)
        return []

    pattern_list = tuple(str(p) for p in patterns if str(p).strip())
    if not pattern_list:
        return []

    cutoff_epoch = time.time() - (int(ttl_days) * _SECONDS_PER_DAY)
    matches: list[str] = []

    for current_root, dirnames, filenames in os.walk(root_path, topdown=True, followlinks=False):
        current_root_path = Path(current_root)

        # Never walk symlink directories.
        dirnames[:] = [
            d for d in dirnames if not (current_root_path / d).is_symlink()
        ]

        for filename in filenames:
            path = current_root_path / filename
            if path.is_symlink():
                continue
            if _is_protected_file(path):
                continue
            if not any(fnmatch.fnmatch(filename, pattern) for pattern in pattern_list):
                continue
            if _safe_mtime(path) < cutoff_epoch:
                matches.append(str(path))

    matches.sort()
    return matches


def delete_paths(paths: list[str], dry_run: bool) -> dict[str, int | bool]:
    """Delete files/dirs (or simulate), skipping symlinks and protected files."""
    summary: dict[str, int | bool] = {
        "dry_run": bool(dry_run),
        "n_candidate_files": 0,
        "n_candidate_dirs": 0,
        "n_deleted_files": 0,
        "n_deleted_dirs": 0,
        "bytes_freed_estimated": 0,
        "n_skipped_missing": 0,
        "n_skipped_symlinks": 0,
        "n_skipped_protected": 0,
        "n_errors": 0,
    }

    seen: set[str] = set()
    for raw in paths:
        path = Path(raw)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)

        if path.is_symlink():
            summary["n_skipped_symlinks"] = int(summary["n_skipped_symlinks"]) + 1
            _logger.info("retention skip symlink | path=%s", _safe_rel_path(path))
            continue
        if not path.exists():
            summary["n_skipped_missing"] = int(summary["n_skipped_missing"]) + 1
            continue
        if path.is_file() and _is_protected_file(path):
            summary["n_skipped_protected"] = int(summary["n_skipped_protected"]) + 1
            continue

        bytes_est = _estimate_path_bytes(path)
        summary["bytes_freed_estimated"] = int(summary["bytes_freed_estimated"]) + int(bytes_est)

        if path.is_file():
            summary["n_candidate_files"] = int(summary["n_candidate_files"]) + 1
            if bool(dry_run):
                _logger.info("retention dry-run file | path=%s | bytes=%s", _safe_rel_path(path), bytes_est)
                continue
            try:
                path.unlink()
                summary["n_deleted_files"] = int(summary["n_deleted_files"]) + 1
                _logger.info("retention deleted file | path=%s | bytes=%s", _safe_rel_path(path), bytes_est)
            except OSError:
                summary["n_errors"] = int(summary["n_errors"]) + 1
                _logger.warning("retention delete file failed | path=%s", _safe_rel_path(path))
            continue

        if path.is_dir():
            summary["n_candidate_dirs"] = int(summary["n_candidate_dirs"]) + 1
            if bool(dry_run):
                _logger.info("retention dry-run dir | path=%s | bytes=%s", _safe_rel_path(path), bytes_est)
                continue
            try:
                shutil.rmtree(path)
                summary["n_deleted_dirs"] = int(summary["n_deleted_dirs"]) + 1
                _logger.info("retention deleted dir | path=%s | bytes=%s", _safe_rel_path(path), bytes_est)
            except OSError:
                summary["n_errors"] = int(summary["n_errors"]) + 1
                _logger.warning("retention delete dir failed | path=%s", _safe_rel_path(path))

    return summary


def _backup_dir_sort_key(path: Path) -> tuple[str, float]:
    # Backup dirs in this repo are timestamped (`YYYYMMDDTHHMMSSZ[_NN]`), so name is
    # the primary key. `mtime` is a fallback if names are custom/touched later.
    return (path.name, _safe_mtime(path))


def keep_n_newest_dirs(root: str, keep: int, dry_run: bool) -> dict[str, Any]:
    """Keep only the N newest direct subdirectories under `root`."""
    if int(keep) < 0:
        raise ValueError("keep must be >= 0")

    root_path = Path(root)
    if not root_path.exists():
        _logger.warning("retention backup root missing (skip) | root=%s", root)
        return {
            "root": str(root),
            "keep": int(keep),
            "n_found_dirs": 0,
            "n_candidate_dirs": 0,
            "n_skipped_missing_dirs": 1,
            "delete_summary": delete_paths([], dry_run=bool(dry_run)),
        }
    if not root_path.is_dir():
        _logger.warning("retention backup root is not a directory (skip) | root=%s", root)
        return {
            "root": str(root),
            "keep": int(keep),
            "n_found_dirs": 0,
            "n_candidate_dirs": 0,
            "n_skipped_missing_dirs": 1,
            "delete_summary": delete_paths([], dry_run=bool(dry_run)),
        }

    dirs: list[Path] = []
    skipped_symlinks = 0
    for child in root_path.iterdir():
        if child.is_symlink():
            skipped_symlinks += 1
            continue
        if child.is_dir():
            dirs.append(child)

    dirs_sorted = sorted(dirs, key=_backup_dir_sort_key, reverse=True)
    candidates = dirs_sorted[int(keep) :] if int(keep) < len(dirs_sorted) else []
    delete_summary = delete_paths([str(p) for p in candidates], dry_run=bool(dry_run))
    delete_summary["n_skipped_symlinks"] = int(delete_summary["n_skipped_symlinks"]) + int(
        skipped_symlinks
    )

    return {
        "root": str(root),
        "keep": int(keep),
        "n_found_dirs": int(len(dirs_sorted)),
        "n_candidate_dirs": int(len(candidates)),
        "n_skipped_missing_dirs": 0,
        "delete_summary": delete_summary,
    }


def _filter_artifact_report_candidates(paths: list[str], artifacts_dir: str) -> list[str]:
    """Allow TTL cleanup only for report-like outputs, never `artifacts/models/**`."""
    root = Path(artifacts_dir)
    dataset_versions_root = root / "dataset_versions"
    models_root = root / "models"
    allowed: list[str] = []

    for raw in paths:
        path = Path(raw)
        try:
            relative = path.relative_to(root)
        except Exception:
            continue

        if path == root / ".gitkeep":
            continue
        if path.is_relative_to(models_root):
            continue
        if path.parent == root:
            allowed.append(str(path))
            continue
        if path.is_relative_to(dataset_versions_root):
            allowed.append(str(path))
            continue

    allowed = sorted(dict.fromkeys(allowed))
    return allowed


def _merge_delete_summaries(items: list[dict[str, int | bool]]) -> dict[str, int | bool]:
    merged: dict[str, int | bool] = {
        "dry_run": bool(items[0]["dry_run"]) if items else True,
        "n_candidate_files": 0,
        "n_candidate_dirs": 0,
        "n_deleted_files": 0,
        "n_deleted_dirs": 0,
        "bytes_freed_estimated": 0,
        "n_skipped_missing": 0,
        "n_skipped_symlinks": 0,
        "n_skipped_protected": 0,
        "n_errors": 0,
    }
    for item in items:
        for key in (
            "n_candidate_files",
            "n_candidate_dirs",
            "n_deleted_files",
            "n_deleted_dirs",
            "bytes_freed_estimated",
            "n_skipped_missing",
            "n_skipped_symlinks",
            "n_skipped_protected",
            "n_errors",
        ):
            merged[key] = int(merged[key]) + int(item.get(key, 0))
    return merged


def run_retention(
    *,
    dry_run: bool = True,
    logs_ttl_days: int = 14,
    artifacts_ttl_days: int = 30,
    keep_model_backups: int = 10,
    keep_reference_backups: int = 5,
    keep_model_releases: int = 0,
    logs_dir: str = "logs",
    artifacts_dir: str = "artifacts",
    model_backups_dir: str = "app/model/backups",
    reference_backups_dir: str = "app/model/reference/backups",
    model_releases_dir: str = "artifacts/models/releases",
    verbose: bool = True,
) -> dict[str, Any]:
    """Apply local retention policy and return a structured summary."""
    previous_logger_level = _logger.level
    if not bool(verbose):
        _logger.setLevel("WARNING")

    try:
        logs_missing_root = 0 if Path(logs_dir).is_dir() else 1
        artifacts_missing_root = 0 if Path(artifacts_dir).is_dir() else 1

        logs_candidates = list_files_older_than(
            root=logs_dir,
            ttl_days=int(logs_ttl_days),
            patterns=("*.jsonl", "*.log"),
        )
        logs_delete = delete_paths(logs_candidates, dry_run=bool(dry_run))

        artifact_candidates_all = list_files_older_than(
            root=artifacts_dir,
            ttl_days=int(artifacts_ttl_days),
            patterns=("*.json", "*.md"),
        )
        artifact_candidates = _filter_artifact_report_candidates(artifact_candidates_all, artifacts_dir)
        artifacts_delete = delete_paths(artifact_candidates, dry_run=bool(dry_run))

        model_backups = keep_n_newest_dirs(
            root=model_backups_dir,
            keep=int(keep_model_backups),
            dry_run=bool(dry_run),
        )
        reference_backups = keep_n_newest_dirs(
            root=reference_backups_dir,
            keep=int(keep_reference_backups),
            dry_run=bool(dry_run),
        )

        model_releases_cleanup_enabled = int(keep_model_releases) > 0
        if model_releases_cleanup_enabled:
            model_releases = keep_n_newest_dirs(
                root=model_releases_dir,
                keep=int(keep_model_releases),
                dry_run=bool(dry_run),
            )
        else:
            model_releases = {
                "root": str(model_releases_dir),
                "keep": int(keep_model_releases),
                "n_found_dirs": 0,
                "n_candidate_dirs": 0,
                "n_skipped_missing_dirs": 0,
                "enabled": False,
                "delete_summary": delete_paths([], dry_run=bool(dry_run)),
            }

        all_delete_summaries = [
            logs_delete,
            artifacts_delete,
            dict(model_backups["delete_summary"]),
            dict(reference_backups["delete_summary"]),
            dict(model_releases["delete_summary"]),
        ]
        aggregate_delete = _merge_delete_summaries(all_delete_summaries)
        aggregate_delete["dry_run"] = bool(dry_run)

        n_skipped_missing_dirs = int(logs_missing_root) + int(artifacts_missing_root)
        n_skipped_missing_dirs += int(model_backups.get("n_skipped_missing_dirs", 0))
        n_skipped_missing_dirs += int(reference_backups.get("n_skipped_missing_dirs", 0))
        n_skipped_missing_dirs += int(model_releases.get("n_skipped_missing_dirs", 0))

        summary: dict[str, Any] = {
            "status": "ok",
            "dry_run": bool(dry_run),
            "policy": {
                "logs_ttl_days": int(logs_ttl_days),
                "artifacts_ttl_days": int(artifacts_ttl_days),
                "keep_model_backups": int(keep_model_backups),
                "keep_reference_backups": int(keep_reference_backups),
                "keep_model_releases": int(keep_model_releases),
                "model_releases_cleanup_enabled": bool(model_releases_cleanup_enabled),
            },
            "paths": {
                "logs_dir": str(logs_dir),
                "artifacts_dir": str(artifacts_dir),
                "model_backups_dir": str(model_backups_dir),
                "reference_backups_dir": str(reference_backups_dir),
                "model_releases_dir": str(model_releases_dir),
            },
            "sections": {
                "logs": {
                    "ttl_days": int(logs_ttl_days),
                    "patterns": ["*.jsonl", "*.log"],
                    "n_candidates": int(len(logs_candidates)),
                    "delete_summary": logs_delete,
                },
                "artifacts_reports": {
                    "ttl_days": int(artifacts_ttl_days),
                    "patterns": ["*.json", "*.md"],
                    "scope": [
                        "artifacts/*.json",
                        "artifacts/*.md",
                        "artifacts/dataset_versions/*.json",
                    ],
                    "excludes": ["artifacts/models/** (automatic cleanup disabled by default)"],
                    "n_candidates_total_scanned": int(len(artifact_candidates_all)),
                    "n_candidates_filtered": int(len(artifact_candidates)),
                    "delete_summary": artifacts_delete,
                },
                "model_backups": model_backups,
                "reference_backups": reference_backups,
                "model_releases": model_releases,
            },
            "summary": {
                "n_deleted_files": int(aggregate_delete["n_deleted_files"]),
                "n_deleted_dirs": int(aggregate_delete["n_deleted_dirs"]),
                "bytes_freed_estimated": int(aggregate_delete["bytes_freed_estimated"]),
                "n_candidate_files": int(aggregate_delete["n_candidate_files"]),
                "n_candidate_dirs": int(aggregate_delete["n_candidate_dirs"]),
                "n_errors": int(aggregate_delete["n_errors"]),
                "n_skipped_missing_dirs": int(n_skipped_missing_dirs),
                "n_skipped_missing_paths": int(aggregate_delete["n_skipped_missing"]),
                "n_skipped_symlinks": int(aggregate_delete["n_skipped_symlinks"]),
                "n_skipped_protected": int(aggregate_delete["n_skipped_protected"]),
            },
            "notes": [
                "uses_mtime_only_no_file_content_reads",
                "skips_symlinks_and_.gitkeep",
                "artifacts_models_not_deleted_automatically_by_default",
                "online_metrics_jsonl_ttl_is_file_level_not_line_level",
            ],
        }
        return summary
    finally:
        _logger.setLevel(previous_logger_level)


def _print_summary(summary: dict[str, Any]) -> None:
    top = summary.get("summary", {})
    print(
        (
            "Retention summary | dry_run={dry_run} | deleted_files={deleted_files} | "
            "deleted_dirs={deleted_dirs} | bytes_freed_estimated={bytes_freed} | "
            "skipped_missing_dirs={skipped_missing_dirs}"
        ).format(
            dry_run=bool(summary.get("dry_run", True)),
            deleted_files=int(top.get("n_deleted_files", 0)),
            deleted_dirs=int(top.get("n_deleted_dirs", 0)),
            bytes_freed=int(top.get("bytes_freed_estimated", 0)),
            skipped_missing_dirs=int(top.get("n_skipped_missing_dirs", 0)),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply local retention policy for logs/artifacts/backups. "
            "Dry-run is enabled by default."
        )
    )
    parser.add_argument("--dry-run", type=int, default=1, choices=[0, 1])
    parser.add_argument("--logs-ttl-days", type=int, default=14)
    parser.add_argument("--artifacts-ttl-days", type=int, default=30)
    parser.add_argument("--keep-model-backups", type=int, default=10)
    parser.add_argument("--keep-reference-backups", type=int, default=5)
    parser.add_argument(
        "--keep-model-releases",
        type=int,
        default=0,
        help="Optional keep-N cleanup for artifacts/models/releases (0 disables automatic cleanup).",
    )
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--model-backups-dir", type=str, default="app/model/backups")
    parser.add_argument(
        "--reference-backups-dir",
        type=str,
        default="app/model/reference/backups",
    )
    parser.add_argument("--model-releases-dir", type=str, default="artifacts/models/releases")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    summary = run_retention(
        dry_run=_parse_bool_flag(args.dry_run),
        logs_ttl_days=int(args.logs_ttl_days),
        artifacts_ttl_days=int(args.artifacts_ttl_days),
        keep_model_backups=int(args.keep_model_backups),
        keep_reference_backups=int(args.keep_reference_backups),
        keep_model_releases=int(args.keep_model_releases),
        logs_dir=str(args.logs_dir),
        artifacts_dir=str(args.artifacts_dir),
        model_backups_dir=str(args.model_backups_dir),
        reference_backups_dir=str(args.reference_backups_dir),
        model_releases_dir=str(args.model_releases_dir),
        verbose=_parse_bool_flag(args.verbose),
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
