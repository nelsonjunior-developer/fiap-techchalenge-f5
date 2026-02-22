from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.create_release import run_create_release
from src.model_versioning import create_release, make_model_version


def _collect_keys(payload):
    keys = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys |= _collect_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_keys(item)
    return keys


def _write_source_artifacts(
    root: Path,
    *,
    family: str = "nonlinear_hgb",
    variant: str = "default",
    with_model_version: bool = False,
) -> tuple[Path, Path]:
    target = root / family / variant
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model.joblib"
    metadata_path = target / "metadata.json"
    model_path.write_bytes(b"dummy-model-artifact")

    payload = {
        "variant": variant,
        "created_at": "2026-02-22T14:05:33+00:00",
        "threshold_policy": {"operational_fixed_threshold": 0.30},
    }
    if with_model_version:
        payload["model_version"] = "2026-02-22T14-05-33Z__deadbeef"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, metadata_path


def test_make_model_version_is_stable_with_sha_prefix() -> None:
    version = make_model_version(
        "2026-02-22T14:05:33+00:00",
        "a1b2c3d4e5f67890",
    )
    assert version == "2026-02-22T14-05-33Z__a1b2c3d4"


def test_create_release_creates_dir_and_copies_artifacts(tmp_path: Path) -> None:
    src_model, src_meta = _write_source_artifacts(tmp_path / "artifacts" / "models")

    manifest = create_release(
        source_model_path=src_model,
        source_metadata_path=src_meta,
        out_root=tmp_path / "artifacts" / "models" / "releases",
    )

    release_dir = tmp_path / "artifacts" / "models" / "releases" / manifest["model_version"]
    assert release_dir.exists()
    assert (release_dir / "model.joblib").exists()
    assert (release_dir / "metadata.json").exists()
    assert (release_dir / "release.json").exists()

    copied_metadata = json.loads((release_dir / "metadata.json").read_text(encoding="utf-8"))
    assert copied_metadata["model_version"] == manifest["model_version"]
    assert isinstance(copied_metadata.get("trained_at"), str)
    assert copied_metadata["artifact_hashes"]["model_joblib_sha256"] == manifest["sha256"]["model_joblib"]
    assert copied_metadata["model_family"] == "nonlinear_hgb"
    assert copied_metadata["variant"] == "default"

    release_payload = json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
    assert release_payload["model_version"] == manifest["model_version"]
    assert release_payload["identity"]["model_family"] == "nonlinear_hgb"
    assert release_payload["identity"]["variant"] == "default"
    assert len(release_payload["sha256"]["model_joblib"]) == 64
    assert len(release_payload["sha256"]["metadata"]) == 64


def test_create_release_fails_when_directory_already_exists(tmp_path: Path) -> None:
    src_model, src_meta = _write_source_artifacts(tmp_path / "artifacts" / "models")
    out_root = tmp_path / "artifacts" / "models" / "releases"

    first = create_release(
        source_model_path=src_model,
        source_metadata_path=src_meta,
        out_root=out_root,
        model_version="v-fixed",
    )
    assert first["model_version"] == "v-fixed"

    with pytest.raises(ValueError, match="Release directory already exists"):
        create_release(
            source_model_path=src_model,
            source_metadata_path=src_meta,
            out_root=out_root,
            model_version="v-fixed",
        )


def test_release_json_does_not_contain_pii_keys(tmp_path: Path) -> None:
    src_model, src_meta = _write_source_artifacts(tmp_path / "artifacts" / "models")
    manifest = create_release(
        source_model_path=src_model,
        source_metadata_path=src_meta,
        out_root=tmp_path / "artifacts" / "models" / "releases",
    )
    release_path = (
        tmp_path
        / "artifacts"
        / "models"
        / "releases"
        / manifest["model_version"]
        / "release.json"
    )
    payload = json.loads(release_path.read_text(encoding="utf-8"))
    forbidden = {"ra", "ra_list", "ids", "students", "records"}
    assert forbidden.isdisjoint(_collect_keys(payload))


def test_run_create_release_uses_selection_and_fallback_model_paths(tmp_path: Path) -> None:
    models_root = tmp_path / "artifacts" / "models"
    _write_source_artifacts(models_root, family="baseline_logreg", variant="none")
    selection_path = tmp_path / "artifacts" / "model_selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "winner": {
                    "model_family": "baseline_logreg",
                    "variant": "none",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = run_create_release(
        selection_path=selection_path,
        models_root=models_root,
        out_root=models_root / "releases",
    )
    assert manifest["identity"]["model_family"] == "baseline_logreg"
    assert manifest["identity"]["variant"] == "none"
    release_dir = models_root / "releases" / manifest["model_version"]
    assert (release_dir / "model.joblib").exists()
    assert (release_dir / "metadata.json").exists()
    assert (release_dir / "release.json").exists()
