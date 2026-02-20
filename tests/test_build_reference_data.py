from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.build_reference_data as build_reference_data


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_valid_serving_metadata(
    *,
    expected_raw_cols: list[str],
    expected_model_cols: list[str],
    variant: str = "default",
) -> dict[str, object]:
    eval_block = {
        "threshold": 0.3,
        "metrics": {
            "recall": 0.7,
            "precision": 0.6,
            "f1": 0.64,
            "roc_auc": 0.69,
            "pr_auc": 0.66,
            "positive_rate": 0.52,
        },
        "confusion_matrix": {"tn": 8, "fp": 2, "fn": 3, "tp": 7},
    }
    return {
        "model_family": "nonlinear_hgb",
        "variant": variant,
        "model_version": "2026-02-20T12-00-00Z",
        "trained_at": "2026-02-20T12:00:00+00:00",
        "promoted_at": "2026-02-20T13:00:00+00:00",
        "random_state": 42,
        "train_pair": {"year_t": 2022, "year_t1": 2023, "n": 2000, "n_pos": 700, "prevalence": 0.35},
        "holdout_pair": {"year_t": 2023, "year_t1": 2024, "n": 1200, "n_pos": 450, "prevalence": 0.375},
        "dataset": {"path_hint": "dataset/fake.xlsx", "basename": "fake.xlsx", "sha256": None},
        "expected_raw_cols": expected_raw_cols,
        "expected_model_cols": expected_model_cols,
        "excluded_cols": ["Nome_Anon", "Avaliador1"],
        "feature_engineering": {"enabled": True, "enable_age_bucket": False, "engineered_cols": ["score_sum"]},
        "feature_pruning": {"plan_hash": "abc123", "kept_model_cols_count": len(expected_model_cols), "dropped_summary": {}},
        "threshold_policy": {
            "operational_fixed_threshold": 0.3,
            "recall_target_for_calibration": 0.9,
            "calibrated_threshold": 0.27,
            "topk_fallback_fraction": 0.2,
            "notes": ["top-k is batch only"],
        },
        "evaluation_train_at_0.5": eval_block,
        "evaluation_train_at_0.30": eval_block,
        "evaluation_holdout_at_0.5": eval_block,
        "evaluation_holdout_at_0.30": eval_block,
        "evaluation_holdout_at_calibrated_threshold": eval_block,
        "versions": {
            "python": "3.11.10",
            "pandas": "2.2.2",
            "numpy": "1.26.4",
            "scikit_learn": None,
            "joblib": None,
        },
        "artifact_hashes": {"model_joblib_sha256": "0" * 64, "metadata_sha256": None},
    }


def test_build_reference_data_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"MODEL_BYTES")

    expected_raw_cols = ["Mat", "Por", "Ing"]
    expected_model_cols = ["Mat", "Por", "Ing", "score_sum"]
    _write_json(
        model_dir / "metadata.json",
        _build_valid_serving_metadata(
            expected_raw_cols=expected_raw_cols,
            expected_model_cols=expected_model_cols,
        ),
    )

    fake_dataset = tmp_path / "dataset.xlsx"
    fake_dataset.write_bytes(b"fake")

    y_train = pd.Series(([1] * 700) + ([0] * 1300), dtype="Int64")
    ids = pd.Series([f"S{i:04d}" for i in range(len(y_train))], dtype="string")
    x_raw = pd.DataFrame(
        {
            "Mat": np.linspace(0, 1, len(y_train)),
            "Por": np.linspace(1, 2, len(y_train)),
            "Ing": np.linspace(2, 3, len(y_train)),
        }
    )
    x_model = x_raw.copy()
    x_model["score_sum"] = x_model["Mat"] + x_model["Por"] + x_model["Ing"]

    monkeypatch.setattr(
        build_reference_data,
        "load_pede_workbook_with_metadata",
        lambda file_path: ({2022: pd.DataFrame(), 2023: pd.DataFrame()}, {}, {}),
    )
    monkeypatch.setattr(
        build_reference_data,
        "make_temporal_pairs",
        lambda *args, **kwargs: (pd.DataFrame(), y_train, ids),
    )
    monkeypatch.setattr(
        build_reference_data,
        "build_raw_from_ids",
        lambda **kwargs: x_raw,
    )
    monkeypatch.setattr(
        build_reference_data,
        "transform_raw_to_model_frame",
        lambda **kwargs: x_model,
    )

    out_dir = tmp_path / "app" / "model" / "reference"
    report = build_reference_data.run_build_reference_data(
        file_path=fake_dataset,
        model_dir=model_dir,
        out_dir=out_dir,
        max_rows=1000,
        backup=True,
        force=False,
        include_raw_diagnostic=False,
    )

    csv_path = out_dir / "reference_model_frame.csv"
    profile_path = out_dir / "reference_profile.json"
    meta_path = out_dir / "reference_meta.json"

    assert csv_path.exists()
    assert profile_path.exists()
    assert meta_path.exists()
    assert report["used_rows"] == 1000

    saved_df = pd.read_csv(csv_path)
    assert len(saved_df) == 1000
    assert list(saved_df.columns) == expected_model_cols

    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile_payload["overview"]["n_rows_reference"] == 1000
    assert profile_payload["overview"]["n_features"] == len(expected_model_cols)
    assert len(profile_payload["features"]) == len(expected_model_cols)

    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_payload["sampling"]["used_rows"] == 1000
    assert meta_payload["sampling"]["strategy"] == "stratified_deterministic"
    assert isinstance(meta_payload["sha256"]["reference_csv"], str)
    assert len(meta_payload["sha256"]["reference_csv"]) == 64


def test_build_reference_data_privacy_blocks_forbidden_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"MODEL_BYTES")

    expected_raw_cols = ["RA", "Mat"]
    expected_model_cols = ["RA", "Mat"]
    _write_json(
        model_dir / "metadata.json",
        _build_valid_serving_metadata(
            expected_raw_cols=expected_raw_cols,
            expected_model_cols=expected_model_cols,
            variant="privacy_break",
        ),
    )

    fake_dataset = tmp_path / "dataset.xlsx"
    fake_dataset.write_bytes(b"fake")

    y_train = pd.Series([1, 0, 1, 0], dtype="Int64")
    ids = pd.Series(["A", "B", "C", "D"], dtype="string")
    x_raw = pd.DataFrame({"RA": ["A", "B", "C", "D"], "Mat": [1.0, 2.0, 3.0, 4.0]})

    monkeypatch.setattr(
        build_reference_data,
        "load_pede_workbook_with_metadata",
        lambda file_path: ({2022: pd.DataFrame(), 2023: pd.DataFrame()}, {}, {}),
    )
    monkeypatch.setattr(
        build_reference_data,
        "make_temporal_pairs",
        lambda *args, **kwargs: (pd.DataFrame(), y_train, ids),
    )
    monkeypatch.setattr(build_reference_data, "build_raw_from_ids", lambda **kwargs: x_raw)
    monkeypatch.setattr(build_reference_data, "transform_raw_to_model_frame", lambda **kwargs: x_raw)

    with pytest.raises(ValueError, match="forbidden columns"):
        build_reference_data.run_build_reference_data(
            file_path=fake_dataset,
            model_dir=model_dir,
            out_dir=tmp_path / "app" / "model" / "reference",
            max_rows=1000,
            backup=True,
            force=False,
            include_raw_diagnostic=False,
        )


def test_build_reference_data_backup_enabled_creates_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"MODEL_BYTES")

    expected_raw_cols = ["Mat", "Por"]
    expected_model_cols = ["Mat", "Por"]
    _write_json(
        model_dir / "metadata.json",
        _build_valid_serving_metadata(
            expected_raw_cols=expected_raw_cols,
            expected_model_cols=expected_model_cols,
            variant="backup_case",
        ),
    )

    fake_dataset = tmp_path / "dataset.xlsx"
    fake_dataset.write_bytes(b"fake")

    y_train = pd.Series([1, 0, 1, 0, 1, 0], dtype="Int64")
    ids = pd.Series(["A", "B", "C", "D", "E", "F"], dtype="string")
    x_raw = pd.DataFrame({"Mat": [1, 2, 3, 4, 5, 6], "Por": [2, 3, 4, 5, 6, 7]})

    monkeypatch.setattr(
        build_reference_data,
        "load_pede_workbook_with_metadata",
        lambda file_path: ({2022: pd.DataFrame(), 2023: pd.DataFrame()}, {}, {}),
    )
    monkeypatch.setattr(
        build_reference_data,
        "make_temporal_pairs",
        lambda *args, **kwargs: (pd.DataFrame(), y_train, ids),
    )
    monkeypatch.setattr(build_reference_data, "build_raw_from_ids", lambda **kwargs: x_raw)
    monkeypatch.setattr(build_reference_data, "transform_raw_to_model_frame", lambda **kwargs: x_raw)

    out_dir = tmp_path / "app" / "model" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_model_frame.csv").write_text("old", encoding="utf-8")
    _write_json(out_dir / "reference_profile.json", {"old": True})
    _write_json(out_dir / "reference_meta.json", {"old": True})

    build_reference_data.run_build_reference_data(
        file_path=fake_dataset,
        model_dir=model_dir,
        out_dir=out_dir,
        max_rows=4,
        backup=True,
        force=True,
        include_raw_diagnostic=False,
    )

    backups_root = out_dir / "backups"
    backup_dirs = [path for path in backups_root.iterdir() if path.is_dir()]
    assert len(backup_dirs) == 1
    backup_dir = backup_dirs[0]
    assert (backup_dir / "reference_model_frame.csv").read_text(encoding="utf-8") == "old"
    assert json.loads((backup_dir / "reference_profile.json").read_text(encoding="utf-8")) == {"old": True}
    assert json.loads((backup_dir / "reference_meta.json").read_text(encoding="utf-8")) == {"old": True}


def test_build_reference_data_fails_when_promoted_model_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Promoted model not found"):
        build_reference_data.run_build_reference_data(
            file_path=tmp_path / "missing_dataset.xlsx",
            model_dir=tmp_path / "app" / "model",
            out_dir=tmp_path / "app" / "model" / "reference",
        )


def test_build_reference_data_existing_destination_without_force_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"MODEL_BYTES")
    _write_json(
        model_dir / "metadata.json",
        _build_valid_serving_metadata(
            expected_raw_cols=["Mat"],
            expected_model_cols=["Mat"],
            variant="exists_case",
        ),
    )

    fake_dataset = tmp_path / "dataset.xlsx"
    fake_dataset.write_bytes(b"fake")
    y_train = pd.Series([1, 0, 1, 0], dtype="Int64")
    ids = pd.Series(["A", "B", "C", "D"], dtype="string")
    x_raw = pd.DataFrame({"Mat": [1, 2, 3, 4]})

    monkeypatch.setattr(
        build_reference_data,
        "load_pede_workbook_with_metadata",
        lambda file_path: ({2022: pd.DataFrame(), 2023: pd.DataFrame()}, {}, {}),
    )
    monkeypatch.setattr(
        build_reference_data,
        "make_temporal_pairs",
        lambda *args, **kwargs: (pd.DataFrame(), y_train, ids),
    )
    monkeypatch.setattr(build_reference_data, "build_raw_from_ids", lambda **kwargs: x_raw)
    monkeypatch.setattr(build_reference_data, "transform_raw_to_model_frame", lambda **kwargs: x_raw)

    out_dir = tmp_path / "app" / "model" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_model_frame.csv").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="Reference destination exists"):
        build_reference_data.run_build_reference_data(
            file_path=fake_dataset,
            model_dir=model_dir,
            out_dir=out_dir,
            force=False,
            backup=False,
        )


def test_build_reference_data_include_raw_diagnostic_outputs_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"MODEL_BYTES")
    _write_json(
        model_dir / "metadata.json",
        _build_valid_serving_metadata(
            expected_raw_cols=["Mat", "Por"],
            expected_model_cols=["Mat", "Por"],
            variant="raw_diag",
        ),
    )

    fake_dataset = tmp_path / "dataset.xlsx"
    fake_dataset.write_bytes(b"fake")
    y_train = pd.Series([1, 0, 1, 0, 1, 0], dtype="Int64")
    ids = pd.Series(["A", "B", "C", "D", "E", "F"], dtype="string")
    x_raw = pd.DataFrame({"Mat": [1, 2, 3, 4, 5, 6], "Por": [6, 5, 4, 3, 2, 1]})

    monkeypatch.setattr(
        build_reference_data,
        "load_pede_workbook_with_metadata",
        lambda file_path: ({2022: pd.DataFrame(), 2023: pd.DataFrame()}, {}, {}),
    )
    monkeypatch.setattr(
        build_reference_data,
        "make_temporal_pairs",
        lambda *args, **kwargs: (pd.DataFrame(), y_train, ids),
    )
    monkeypatch.setattr(build_reference_data, "build_raw_from_ids", lambda **kwargs: x_raw)
    monkeypatch.setattr(build_reference_data, "transform_raw_to_model_frame", lambda **kwargs: x_raw)

    out_dir = tmp_path / "app" / "model" / "reference"
    report = build_reference_data.run_build_reference_data(
        file_path=fake_dataset,
        model_dir=model_dir,
        out_dir=out_dir,
        force=False,
        backup=False,
        include_raw_diagnostic=True,
        max_rows=4,
    )
    assert report["status"] == "PASS"
    assert (out_dir / "reference_raw_diagnostic.csv").exists()
    meta_payload = json.loads((out_dir / "reference_meta.json").read_text(encoding="utf-8"))
    assert meta_payload["paths"]["raw_diagnostic_csv"] is not None


def test_internal_helpers_cover_branches(tmp_path: Path) -> None:
    rng = np.random.RandomState(42)
    empty_sample = build_reference_data._sample_without_replacement(np.array([1, 2, 3]), 0, rng)
    assert empty_sample.size == 0
    full_sample = build_reference_data._sample_without_replacement(np.array([1, 2]), 5, rng)
    assert full_sample.tolist() == [1, 2]

    with pytest.raises(ValueError, match="max_rows must be > 0"):
        build_reference_data._stratified_indices(pd.Series([1, 0, 1]), 0)

    single_class_idx = build_reference_data._stratified_indices(pd.Series([1] * 20), 5, random_state=42)
    assert len(single_class_idx) == 5

    numeric_empty = build_reference_data._numeric_summary(pd.Series([np.nan, np.nan]))
    assert numeric_empty["mean"] is None

    cat_top = build_reference_data._categorical_top_values(pd.Series(list("abcdefghijk")), limit=3)
    assert cat_top[-1]["value"] == "_OTHER_"

    assert build_reference_data._infer_profile_kind(pd.Series([0, 1, 0])) == "binary"
    assert build_reference_data._infer_profile_kind(pd.Series([1.2, 3.4])) == "numeric"
    assert build_reference_data._infer_profile_kind(pd.Series(["x", "y", "z"])) == "categorical"

    with pytest.raises(ValueError, match="forbidden columns"):
        build_reference_data._assert_no_forbidden_columns(["Avaliador1", "Mat"])

    data_path = tmp_path / "metadata.json"
    data_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON payload"):
        build_reference_data._safe_read_json(data_path)


def test_build_reference_data_main_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_reference_data, "setup_logging", lambda: None)

    # Success path.
    monkeypatch.setattr(
        build_reference_data,
        "run_build_reference_data",
        lambda **kwargs: {
            "status": "PASS",
            "used_rows": 10,
            "n_features": 3,
            "reference_csv": "x.csv",
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["python", "--backup", "1", "--force", "0", "--include-raw-diagnostic", "0"],
    )
    build_reference_data.main()

    # Error path.
    def _raise_error(**kwargs: object) -> dict[str, object]:
        raise ValueError("boom")

    monkeypatch.setattr(build_reference_data, "run_build_reference_data", _raise_error)
    monkeypatch.setattr("sys.argv", ["python"])
    with pytest.raises(SystemExit) as exc:
        build_reference_data.main()
    assert exc.value.code == 1
