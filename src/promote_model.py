"""Promote the selected champion model artifact to a fixed serving path."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_FORBIDDEN_KEYS = {"ra", "ra_list", "ids", "student_ids", "students", "records"}


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload (expected object): {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys |= _collect_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_keys(item)
    return keys


def _parse_bool_flag(value: int) -> bool:
    return bool(int(value))


def _resolve_winner_paths(
    *,
    selection_payload: dict[str, Any],
    models_root: Path,
) -> dict[str, Any]:
    winner = selection_payload.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("Selection artifact missing winner block.")

    model_family = str(winner.get("model_family") or "").strip()
    variant = str(winner.get("variant") or "").strip()
    if not model_family or not variant:
        raise ValueError("Selection winner must include model_family and variant.")

    src_model_raw = str(winner.get("path_model") or "").strip()
    src_meta_raw = str(winner.get("path_metadata") or "").strip()
    src_model = (
        Path(src_model_raw)
        if src_model_raw
        else models_root / model_family / variant / "model.joblib"
    )
    src_meta = (
        Path(src_meta_raw)
        if src_meta_raw
        else models_root / model_family / variant / "metadata.json"
    )

    if not src_model.exists():
        raise FileNotFoundError(f"Source model.joblib not found: {src_model}")
    if not src_meta.exists():
        raise FileNotFoundError(f"Source metadata.json not found: {src_meta}")

    return {
        "winner": {
            "model_family": model_family,
            "variant": variant,
        },
        "src_model": src_model,
        "src_meta": src_meta,
    }


def _backup_existing_destination(
    *,
    out_dir: Path,
    dest_model: Path,
    dest_meta: Path,
) -> Path | None:
    has_existing = dest_model.exists() or dest_meta.exists()
    if not has_existing:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = out_dir / "backups" / timestamp
    suffix = 1
    while backup_dir.exists():
        backup_dir = out_dir / "backups" / f"{timestamp}_{suffix:02d}"
        suffix += 1
    backup_dir.mkdir(parents=True, exist_ok=False)

    if dest_model.exists():
        shutil.copy2(dest_model, backup_dir / "model.joblib")
    if dest_meta.exists():
        shutil.copy2(dest_meta, backup_dir / "metadata.json")
    return backup_dir


def run_model_promotion(
    *,
    selection_path: str | Path = "artifacts/model_selection.json",
    models_root: str | Path = "artifacts/models",
    out_dir: str | Path = "app/model",
    force: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    selection_path_obj = Path(selection_path)
    if not selection_path_obj.exists():
        raise FileNotFoundError(f"Selection artifact not found: {selection_path_obj}")
    selection_payload = _safe_read_json(selection_path_obj)
    resolved = _resolve_winner_paths(
        selection_payload=selection_payload,
        models_root=Path(models_root),
    )

    src_model = Path(resolved["src_model"])
    src_meta = Path(resolved["src_meta"])
    winner = dict(resolved["winner"])

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    dest_model = out_dir_path / "model.joblib"
    dest_meta = out_dir_path / "metadata.json"

    if dest_model.exists() and not bool(force):
        raise ValueError(
            "Destination exists. Use --force 1 to overwrite (backup enabled by default)."
        )

    backup_path: Path | None = None
    if bool(backup):
        backup_path = _backup_existing_destination(
            out_dir=out_dir_path,
            dest_model=dest_model,
            dest_meta=dest_meta,
        )

    shutil.copy2(src_model, dest_model)
    shutil.copy2(src_meta, dest_meta)

    promoted_payload = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "winner": {
            "model_family": winner["model_family"],
            "variant": winner["variant"],
        },
        "source_paths": {
            "model": str(src_model),
            "metadata": str(src_meta),
        },
        "dest_paths": {
            "model": str(dest_model),
            "metadata": str(dest_meta),
        },
        "sha256": {
            "model": _sha256(dest_model),
            "metadata": _sha256(dest_meta),
        },
        "backup": {
            "enabled": bool(backup),
            "path": str(backup_path) if backup_path is not None else None,
        },
        "notes": [
            "promotion copies the winning pipeline artifact for serving",
            "app/model/model.joblib is the fixed serving path for future API loading",
        ],
    }

    keys_found = _collect_keys(promoted_payload)
    forbidden = sorted(_FORBIDDEN_KEYS & keys_found)
    if forbidden:
        raise ValueError(
            f"Privacy check failed: forbidden keys found in promotion payload: {forbidden}"
        )

    promoted_path = out_dir_path / "promoted_model.json"
    promoted_path.write_text(
        json.dumps(promoted_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return promoted_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote winner model artifact to fixed serving path app/model."
    )
    parser.add_argument(
        "--selection-path",
        type=str,
        default="artifacts/model_selection.json",
        help="Path to model_selection artifact.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="artifacts/models",
        help="Root directory containing trained model variants.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="app/model",
        help="Serving destination directory.",
    )
    parser.add_argument(
        "--force",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, allow overwrite in destination.",
    )
    parser.add_argument(
        "--backup",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, backup existing destination model/metadata before overwrite.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    try:
        promoted = run_model_promotion(
            selection_path=args.selection_path,
            models_root=args.models_root,
            out_dir=args.out_dir,
            force=_parse_bool_flag(args.force),
            backup=_parse_bool_flag(args.backup),
        )
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("%s", exc)
        raise SystemExit(1) from exc

    _logger.info(
        "Model promotion completed | winner=%s/%s dest=%s",
        promoted["winner"]["model_family"],
        promoted["winner"]["variant"],
        promoted["dest_paths"]["model"],
    )


if __name__ == "__main__":
    main()
