"""Create versioned local release artifacts from the formally selected winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.model_versioning import create_release
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload (expected object): {path}")
    return payload


def _resolve_source_paths(
    *,
    selection_payload: dict[str, Any],
    models_root: Path,
) -> tuple[Path, Path]:
    winner = selection_payload.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("Selection artifact missing winner block.")

    model_family = str(winner.get("model_family") or "").strip()
    variant = str(winner.get("variant") or "").strip()
    if not model_family or not variant:
        raise ValueError("Selection winner must include model_family and variant.")

    path_model_raw = str(winner.get("path_model") or "").strip()
    path_meta_raw = str(winner.get("path_metadata") or "").strip()

    src_model = (
        Path(path_model_raw)
        if path_model_raw
        else models_root / model_family / variant / "model.joblib"
    )
    src_meta = (
        Path(path_meta_raw)
        if path_meta_raw
        else models_root / model_family / variant / "metadata.json"
    )
    return src_model, src_meta


def run_create_release(
    *,
    selection_path: str | Path = "artifacts/model_selection.json",
    models_root: str | Path = "artifacts/models",
    out_root: str | Path = "artifacts/models/releases",
    model_version: str | None = None,
) -> dict[str, Any]:
    selection_path_obj = Path(selection_path)
    if not selection_path_obj.exists():
        raise FileNotFoundError(f"Selection artifact not found: {selection_path_obj}")

    selection_payload = _safe_read_json(selection_path_obj)
    src_model, src_meta = _resolve_source_paths(
        selection_payload=selection_payload,
        models_root=Path(models_root),
    )

    return create_release(
        source_model_path=src_model,
        source_metadata_path=src_meta,
        out_root=out_root,
        model_version=model_version,
        backup=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create immutable versioned release directory from selected winner."
    )
    parser.add_argument(
        "--selection-path",
        type=str,
        default="artifacts/model_selection.json",
        help="Path to model_selection.json.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="artifacts/models",
        help="Root directory for family/variant build artifacts.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="artifacts/models/releases",
        help="Root directory where release folders are created.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Optional explicit model_version to use for the release directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    try:
        release = run_create_release(
            selection_path=args.selection_path,
            models_root=args.models_root,
            out_root=args.out_root,
            model_version=args.model_version,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _logger.error("%s", exc)
        raise SystemExit(1) from exc

    _logger.info(
        "Release created | model_version=%s dir=%s",
        release["model_version"],
        release["release"]["dir"],
    )


if __name__ == "__main__":
    main()
