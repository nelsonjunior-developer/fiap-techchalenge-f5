"""Local release versioning helpers for trained model artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.dataset_versioning import compute_file_sha256
from src.privacy import find_forbidden_json_keys


def compute_sha256(path: str | Path) -> str:
    """Compute SHA-256 of an artifact in streaming mode."""
    return compute_file_sha256(path)


def _parse_iso_utc_or_none(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_utc_timestamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def make_model_version(trained_at_iso: str, model_joblib_sha256: str) -> str:
    """Build release version string from train timestamp + joblib hash prefix."""
    parsed = _parse_iso_utc_or_none(trained_at_iso)
    if parsed is None:
        parsed = datetime.now(timezone.utc)
    sha = str(model_joblib_sha256 or "").strip().lower()
    sha_prefix = (sha[:8] if len(sha) >= 8 else sha) or "unknown000"
    return f"{_compact_utc_timestamp(parsed)}__{sha_prefix}"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload (expected object): {path}")
    return payload


def _safe_rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _infer_identity(source_metadata: dict[str, Any], source_metadata_path: Path) -> dict[str, str]:
    model_family = str(source_metadata.get("model_family") or "").strip()
    variant = str(source_metadata.get("variant") or "").strip()
    if model_family and variant:
        return {"model_family": model_family, "variant": variant}

    parts = source_metadata_path.parts
    if len(parts) >= 4:
        # .../artifacts/models/<family>/<variant>/metadata.json
        try:
            idx = parts.index("models")
            if len(parts) > idx + 2:
                inferred_family = str(parts[idx + 1]).strip()
                inferred_variant = str(parts[idx + 2]).strip()
                return {
                    "model_family": inferred_family or "unknown",
                    "variant": inferred_variant or "unknown",
                }
        except ValueError:
            pass
    return {"model_family": "unknown", "variant": "unknown"}


def _resolve_trained_at(source_metadata: dict[str, Any], source_model_path: Path) -> str:
    for key in ("trained_at", "created_at"):
        parsed = _parse_iso_utc_or_none(source_metadata.get(key))
        if parsed is not None:
            return parsed.isoformat()
    try:
        mtime = datetime.fromtimestamp(source_model_path.stat().st_mtime, tz=timezone.utc)
        return mtime.isoformat()
    except OSError:
        return _utc_now_iso()


def _enrich_release_metadata(
    *,
    source_metadata: dict[str, Any],
    source_metadata_path: Path,
    source_model_sha256: str,
    model_version: str,
    trained_at: str,
) -> dict[str, Any]:
    payload = dict(source_metadata)
    identity = _infer_identity(source_metadata, source_metadata_path)

    payload["model_version"] = str(payload.get("model_version") or model_version)
    payload["trained_at"] = str(payload.get("trained_at") or payload.get("created_at") or trained_at)
    if not str(payload.get("model_family") or "").strip():
        payload["model_family"] = identity["model_family"]
    if not str(payload.get("variant") or "").strip():
        payload["variant"] = identity["variant"]

    artifact_hashes_raw = payload.get("artifact_hashes")
    artifact_hashes = dict(artifact_hashes_raw) if isinstance(artifact_hashes_raw, dict) else {}
    artifact_hashes["model_joblib_sha256"] = str(
        artifact_hashes.get("model_joblib_sha256") or source_model_sha256
    )
    artifact_hashes.setdefault("metadata_sha256", None)
    payload["artifact_hashes"] = artifact_hashes

    notes_raw = payload.get("notes")
    notes = [str(item) for item in notes_raw] if isinstance(notes_raw, list) else []
    if "release artifact metadata copy (source metadata preserved separately at build-artifact path)" not in notes:
        notes.append(
            "release artifact metadata copy (source metadata preserved separately at build-artifact path)"
        )
    payload["notes"] = list(dict.fromkeys(notes))
    return payload


def create_release(
    source_model_path: str | Path,
    source_metadata_path: str | Path,
    out_root: str | Path = "artifacts/models/releases",
    model_version: str | None = None,
    backup: bool = False,
) -> dict[str, Any]:
    """Create immutable local release directory with model + metadata + release manifest."""
    src_model = Path(source_model_path)
    src_meta = Path(source_metadata_path)
    if not src_model.exists():
        raise FileNotFoundError(f"Source model.joblib not found: {src_model}")
    if not src_meta.exists():
        raise FileNotFoundError(f"Source metadata.json not found: {src_meta}")

    source_metadata = _safe_read_json(src_meta)
    model_sha = compute_sha256(src_model)
    trained_at = _resolve_trained_at(source_metadata, src_model)

    metadata_model_version = str(source_metadata.get("model_version") or "").strip()
    resolved_model_version = (
        str(model_version).strip()
        if model_version is not None and str(model_version).strip()
        else (metadata_model_version or make_model_version(trained_at, model_sha))
    )

    out_root_path = Path(out_root)
    ensure_dir(out_root_path)
    out_dir = out_root_path / resolved_model_version
    if out_dir.exists():
        raise ValueError(f"Release directory already exists: {out_dir}")

    temp_dir = out_root_path / f".{resolved_model_version}.tmp"
    suffix = 1
    while temp_dir.exists():
        temp_dir = out_root_path / f".{resolved_model_version}.tmp_{suffix:02d}"
        suffix += 1
    ensure_dir(temp_dir)

    dest_model = temp_dir / "model.joblib"
    dest_meta = temp_dir / "metadata.json"
    shutil.copy2(src_model, dest_model)

    metadata_copy = _enrich_release_metadata(
        source_metadata=source_metadata,
        source_metadata_path=src_meta,
        source_model_sha256=model_sha,
        model_version=resolved_model_version,
        trained_at=trained_at,
    )
    dest_meta.write_text(
        json.dumps(metadata_copy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata_sha = compute_sha256(dest_meta)
    identity = _infer_identity(metadata_copy, src_meta)
    release_payload = {
        "model_version": resolved_model_version,
        "created_at": _utc_now_iso(),
        "source": {
            "model_path": _safe_rel_path(src_model),
            "metadata_path": _safe_rel_path(src_meta),
        },
        "release": {
            "dir": _safe_rel_path(out_dir),
            "model_path": _safe_rel_path(out_dir / "model.joblib"),
            "metadata_path": _safe_rel_path(out_dir / "metadata.json"),
            "backup_enabled": bool(backup),
        },
        "sha256": {
            "model_joblib": model_sha,
            "metadata": metadata_sha,
        },
        "identity": identity,
        "trained_at": str(metadata_copy.get("trained_at") or trained_at),
        "notes": [
            "Local release is immutable and should not overwrite an existing model_version directory.",
            "Build artifacts remain under artifacts/models/<family>/<variant>/; releases are copies for rollback/rastreability.",
        ],
    }
    forbidden = find_forbidden_json_keys(release_payload)
    if forbidden:
        raise ValueError(f"Privacy check failed in release.json payload: {forbidden}")

    (temp_dir / "release.json").write_text(
        json.dumps(release_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Atomic-ish finalize on same filesystem.
    temp_dir.rename(out_dir)
    return release_payload


__all__ = [
    "compute_sha256",
    "create_release",
    "ensure_dir",
    "make_model_version",
]
