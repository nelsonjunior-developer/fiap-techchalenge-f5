"""Dataset fingerprint/versioning helpers for train/evaluation traceability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 in streaming mode to avoid loading full file in memory."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be > 0")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path_hint(path: str | Path) -> str:
    """Return basename-only hint to avoid leaking local directory structure."""
    return Path(path).name


def get_dataset_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return metadata-only dataset fingerprint (no row-level content)."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    stats = file_path.stat()
    mtime_utc = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat()
    return {
        "path_hint": safe_path_hint(file_path),
        "basename": file_path.name,
        "bytes": int(stats.st_size),
        "mtime_utc": mtime_utc,
        "sha256": compute_file_sha256(file_path),
    }


def persist_dataset_version(fingerprint: dict[str, Any], out_path: str | Path) -> None:
    """Persist dataset version payload as JSON (indented)."""
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_dataset_version_event(
    *,
    context: str,
    dataset_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "context": str(context),
        "dataset": dict(dataset_fingerprint),
    }


def persist_dataset_version_event(
    *,
    context: str,
    dataset_fingerprint: dict[str, Any],
    output_dir: str | Path = "artifacts/dataset_versions",
) -> Path:
    """Persist one execution-scoped dataset version event under artifacts/dataset_versions."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    basename = str(dataset_fingerprint.get("basename") or "dataset").replace(" ", "_")
    event_path = Path(output_dir) / f"{timestamp}_{basename}.json"
    event_payload = build_dataset_version_event(
        context=context,
        dataset_fingerprint=dataset_fingerprint,
    )
    persist_dataset_version(event_payload, event_path)
    return event_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset fingerprint/version payload.")
    parser.add_argument(
        "--path",
        type=str,
        default=os.getenv("DATASET_PATH"),
        help="Dataset file path (XLSX).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/dataset_versions/dataset_version.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="dataset_versioning_cli",
        help="Context label to include in payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.path:
        raise SystemExit(1)
    fingerprint = get_dataset_fingerprint(args.path)
    payload = build_dataset_version_event(
        context=args.context,
        dataset_fingerprint=fingerprint,
    )
    persist_dataset_version(payload, args.out)


if __name__ == "__main__":
    main()
