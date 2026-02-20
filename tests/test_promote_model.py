from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.promote_model import main as promote_main
from src.promote_model import run_model_promotion


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _build_basic_fixture(root: Path) -> tuple[Path, Path, Path]:
    selection_path = root / "artifacts" / "model_selection.json"
    models_root = root / "artifacts" / "models"
    variant_dir = models_root / "baseline_logreg" / "none"
    variant_dir.mkdir(parents=True, exist_ok=True)

    (variant_dir / "model.joblib").write_bytes(b"MODEL_V1")
    _write_json(variant_dir / "metadata.json", {"model_kind": "LogisticRegression", "variant": "none"})
    _write_json(
        selection_path,
        {
            "status": "PASS",
            "winner": {
                "model_family": "baseline_logreg",
                "variant": "none",
            },
        },
    )
    out_dir = root / "app" / "model"
    return selection_path, models_root, out_dir


def test_promote_model_happy_path(tmp_path: Path) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)
    promoted = run_model_promotion(
        selection_path=selection_path,
        models_root=models_root,
        out_dir=out_dir,
        force=False,
        backup=True,
    )

    dest_model = out_dir / "model.joblib"
    dest_meta = out_dir / "metadata.json"
    promoted_json = out_dir / "promoted_model.json"
    assert dest_model.exists()
    assert dest_meta.exists()
    assert promoted_json.exists()

    promoted_payload = json.loads(promoted_json.read_text(encoding="utf-8"))
    assert promoted_payload["winner"] == {"model_family": "baseline_logreg", "variant": "none"}
    assert promoted_payload["sha256"]["model"] == _sha256_bytes(b"MODEL_V1")
    assert isinstance(promoted_payload["sha256"]["metadata"], str)
    assert len(promoted_payload["sha256"]["metadata"]) == 64
    assert promoted["dest_paths"]["model"] == str(dest_model)


def test_destination_exists_without_force_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.joblib").write_bytes(b"OLD_MODEL")

    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--selection-path",
            str(selection_path),
            "--models-root",
            str(models_root),
            "--out-dir",
            str(out_dir),
            "--force",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        promote_main()
    assert exc.value.code == 1


def test_backup_enabled_creates_backup_snapshot(tmp_path: Path) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.joblib").write_bytes(b"OLD_MODEL")
    _write_json(out_dir / "metadata.json", {"old": True})

    run_model_promotion(
        selection_path=selection_path,
        models_root=models_root,
        out_dir=out_dir,
        force=True,
        backup=True,
    )

    backups_root = out_dir / "backups"
    backup_dirs = [path for path in backups_root.iterdir() if path.is_dir()]
    assert len(backup_dirs) == 1
    backup_dir = backup_dirs[0]
    assert (backup_dir / "model.joblib").read_bytes() == b"OLD_MODEL"
    assert json.loads((backup_dir / "metadata.json").read_text(encoding="utf-8")) == {"old": True}
    assert (out_dir / "model.joblib").read_bytes() == b"MODEL_V1"


def test_invalid_selection_or_missing_source_fail_with_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path, models_root, out_dir = _build_basic_fixture(tmp_path)

    # Case 1: selection without winner.
    _write_json(selection_path, {"status": "PASS"})
    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--selection-path",
            str(selection_path),
            "--models-root",
            str(models_root),
            "--out-dir",
            str(out_dir),
        ],
    )
    with pytest.raises(SystemExit) as exc1:
        promote_main()
    assert exc1.value.code == 1

    # Case 2: source model.joblib missing.
    _write_json(
        selection_path,
        {
            "status": "PASS",
            "winner": {
                "model_family": "baseline_logreg",
                "variant": "none",
            },
        },
    )
    (models_root / "baseline_logreg" / "none" / "model.joblib").unlink()
    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--selection-path",
            str(selection_path),
            "--models-root",
            str(models_root),
            "--out-dir",
            str(out_dir),
        ],
    )
    with pytest.raises(SystemExit) as exc2:
        promote_main()
    assert exc2.value.code == 1
