from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.preprocessing as prep
import src.train_baseline as train_baseline
from src.evaluate_holdout import run_holdout_evaluation
from src.training_utils import build_raw_from_ids


class _FakePipeline:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FakePipeline":
        self._y_mean = float(y.mean()) if len(y) else 0.0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = min(max(self._y_mean, 0.01), 0.99)
        probs = np.full(len(X), score, dtype=float)
        return np.column_stack([1.0 - probs, probs])


class _FakeJoblib:
    @staticmethod
    def dump(obj: object, path: Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(obj, handle)

    @staticmethod
    def load(path: Path) -> object:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def _build_fake_yearly_frames_with_holdout(n_rows: int = 24) -> dict[int, pd.DataFrame]:
    expected_raw_cols = prep.get_expected_raw_feature_columns()
    numeric_cols = set(prep.NUMERIC_COLS)
    ids = [f"Z{i:03d}" for i in range(n_rows)]

    def _build_year(offset: int) -> pd.DataFrame:
        data: dict[str, object] = {"RA": pd.Series(ids, dtype="string")}
        for idx, col in enumerate(expected_raw_cols):
            if col in numeric_cols:
                values = np.linspace(1, n_rows, n_rows, dtype=float) + idx + offset
                data[col] = pd.Series(values, dtype="Float64")
            else:
                values = (["A", "B", "C"] * ((n_rows // 3) + 1))[:n_rows]
                data[col] = pd.Series(values, dtype="string")
        return pd.DataFrame(data)

    df_2022 = _build_year(0)
    df_2023 = _build_year(2)
    # Ensure target column with variability exists in both transition targets.
    defas_2023 = ([-1.0, 0.0, 1.0, -2.0, 2.0, 0.5] * ((n_rows // 6) + 1))[:n_rows]
    df_2023["Defasagem"] = pd.Series(defas_2023, dtype="Float64")
    defas_2024 = ([0.0, -1.0, 1.0, -2.0, 2.0, 0.5] * ((n_rows // 6) + 1))[:n_rows]
    df_2024 = pd.DataFrame(
        {
            "RA": pd.Series(ids, dtype="string"),
            "Defasagem": pd.Series(defas_2024, dtype="Float64"),
        }
    )
    return {2022: df_2022, 2023: df_2023, 2024: df_2024}


def _fake_require_dependencies() -> dict[str, object]:
    class _FakeLogReg:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    return {
        "joblib": _FakeJoblib,
        "sklearn_version": "fake-1.0",
        "LogisticRegression": _FakeLogReg,
    }


def test_build_raw_from_ids_keeps_order_and_adds_missing_columns() -> None:
    df = pd.DataFrame(
        {
            "RA": pd.Series(["A", "B", "C"], dtype="string"),
            "Mat": pd.Series([1.0, 2.0, 3.0], dtype="Float64"),
            "Turma": pd.Series(["X", "Y", "Z"], dtype="string"),
        }
    )
    ids = pd.Series(["C", "A", "D"], dtype="string")
    expected_cols = ["Mat", "Turma", "Ing", "Data_Nasc"]
    built = build_raw_from_ids(df, ids, expected_cols)
    assert list(built.columns) == expected_cols
    assert len(built) == 3
    assert built.iloc[0]["Mat"] == 3.0
    assert pd.isna(built.iloc[2]["Mat"])
    assert built["Ing"].isna().all()
    assert built["Data_Nasc"].isna().all()


def test_train_baseline_metadata_contains_evaluation_holdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_frames = _build_fake_yearly_frames_with_holdout()
    fake_dataset = tmp_path / "fake_dataset.xlsx"
    fake_dataset.write_bytes(b"fake")

    monkeypatch.setattr(
        train_baseline,
        "_require_training_dependencies",
        _fake_require_dependencies,
    )
    monkeypatch.setattr(
        train_baseline,
        "load_pede_workbook_with_metadata",
        lambda file_path: (fake_frames, {}, {}),
    )
    monkeypatch.setattr(
        train_baseline,
        "build_model_pipeline",
        lambda **kwargs: _FakePipeline(),
    )

    report = train_baseline.run_baseline_training(
        dataset_path=fake_dataset,
        year_t=2022,
        year_t1=2023,
        out_dir=tmp_path / "models",
        scaler_strategy="standard",
        variants="none",
        enable_feature_engineering=False,
        enable_age_bucket=False,
        eval_holdout=True,
        strict=True,
    )
    metadata_path = Path(report["variants"]["none"]["metadata_path"])
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    holdout = payload.get("evaluation_holdout")
    assert isinstance(holdout, dict)
    assert set(holdout.keys()) >= {"pair", "n", "n_pos", "prevalence", "metrics"}
    assert set(holdout["confusion_matrix_at_0.5"].keys()) == {"tn", "fp", "fn", "tp"}
    assert payload.get("metrics_holdout_at_0.5") is not None


def test_evaluate_holdout_empty_discovery_returns_fail(tmp_path: Path) -> None:
    out_json = tmp_path / "holdout_evaluation.json"
    report = run_holdout_evaluation(
        models_root=tmp_path / "artifacts" / "models",
        output_json=out_json,
        write_markdown=False,
    )
    assert report["status"] == "FAIL"
    assert report["errors"]
    assert out_json.exists()
