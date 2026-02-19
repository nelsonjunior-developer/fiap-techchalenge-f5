from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.preprocessing as prep
import src.train_hgb as train_hgb


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


class _FakeHGB:
    def __init__(
        self,
        random_state: int | None = None,
        max_iter: int | None = None,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        min_samples_leaf: int | None = None,
    ) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf


def _build_fake_yearly_frames(n_rows: int = 24) -> dict[int, pd.DataFrame]:
    expected_raw_cols = prep.get_expected_raw_feature_columns()
    numeric_cols = set(prep.NUMERIC_COLS)
    ids = [f"H{i:03d}" for i in range(n_rows)]

    data_2022: dict[str, object] = {"RA": pd.Series(ids, dtype="string")}
    for idx, col in enumerate(expected_raw_cols):
        if col in numeric_cols:
            values = np.linspace(1, n_rows, n_rows, dtype=float) + idx
            if col == "Ing":
                values[::5] = np.nan
            data_2022[col] = pd.Series(values, dtype="Float64")
        else:
            values = (["A", "B", "C", "A", "B", "C"] * ((n_rows // 6) + 1))[:n_rows]
            data_2022[col] = pd.Series(values, dtype="string")

    df_2022 = pd.DataFrame(data_2022)
    defas_values = ([-1.0, 0.0, 1.0, -2.0, 2.0, 0.5] * ((n_rows // 6) + 1))[:n_rows]
    df_2023 = pd.DataFrame(
        {
            "RA": pd.Series(ids, dtype="string"),
            "Defasagem": pd.Series(defas_values, dtype="Float64"),
        }
    )
    df_2024 = df_2023.copy()
    return {2022: df_2022, 2023: df_2023, 2024: df_2024}


def _fake_require_dependencies() -> dict[str, object]:
    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    def _recall(y_true: np.ndarray, y_pred: np.ndarray, zero_division: int = 0) -> float:
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        return _safe_div(tp, tp + fn)

    def _precision(y_true: np.ndarray, y_pred: np.ndarray, zero_division: int = 0) -> float:
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        return _safe_div(tp, tp + fp)

    def _f1(y_true: np.ndarray, y_pred: np.ndarray, zero_division: int = 0) -> float:
        precision = _precision(y_true, y_pred, zero_division=zero_division)
        recall = _recall(y_true, y_pred, zero_division=zero_division)
        return _safe_div(2 * precision * recall, precision + recall)

    def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
        return float(np.mean(scores))

    def _pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
        return float(np.mean(scores))

    return {
        "joblib": _FakeJoblib,
        "sklearn_version": "fake-1.0",
        "HistGradientBoostingClassifier": _FakeHGB,
        "average_precision_score": _pr_auc,
        "f1_score": _f1,
        "precision_score": _precision,
        "recall_score": _recall,
        "roc_auc_score": _roc_auc,
    }


def test_train_hgb_saves_artifacts_and_metadata_without_pii(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_frames = _build_fake_yearly_frames()
    fake_dataset = tmp_path / "fake_dataset.xlsx"
    fake_dataset.write_bytes(b"fake")

    monkeypatch.setattr(
        train_hgb,
        "_require_training_dependencies",
        _fake_require_dependencies,
    )
    monkeypatch.setattr(
        train_hgb,
        "load_pede_workbook_with_metadata",
        lambda file_path: (fake_frames, {}, {}),
    )
    monkeypatch.setattr(
        train_hgb,
        "build_preprocessing_bundle",
        lambda **kwargs: {
            "expected_raw_cols": prep.get_expected_raw_feature_columns()
        },
    )
    monkeypatch.setattr(
        train_hgb,
        "build_model_pipeline",
        lambda **kwargs: _FakePipeline(),
    )

    report = train_hgb.run_hgb_training(
        file_path=fake_dataset,
        year_t=2022,
        year_t1=2023,
        out_dir=tmp_path / "models",
        variants="default",
        enable_feature_engineering=False,
        enable_age_bucket=False,
        eval_holdout=False,
        strict=True,
    )
    variant_report = report["variants"]["default"]
    model_path = Path(variant_report["model_path"])
    metadata_path = Path(variant_report["metadata_path"])
    assert model_path.exists()
    assert metadata_path.exists()

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    forbidden_keys = {"ids", "ra_list", "students", "rows", "RA_list"}
    assert forbidden_keys.isdisjoint(payload.keys())
    assert "class_imbalance_strategy" in payload
    assert "prediction_policy" in payload
    assert isinstance(payload.get("evaluation_train"), dict)
    assert payload.get("evaluation_holdout") is None

    loaded = _FakeJoblib.load(model_path)
    sample_raw = fake_frames[2022].loc[:, prep.get_expected_raw_feature_columns()].head(3)
    probs = loaded.predict_proba(sample_raw)
    assert probs.shape == (3, 2)


def test_train_hgb_roundtrip_with_real_sklearn_if_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("sklearn")
    real_joblib = pytest.importorskip("joblib")
    fake_frames = _build_fake_yearly_frames()
    fake_dataset = tmp_path / "fake_dataset.xlsx"
    fake_dataset.write_bytes(b"fake")

    monkeypatch.setattr(
        train_hgb,
        "load_pede_workbook_with_metadata",
        lambda file_path: (fake_frames, {}, {}),
    )

    report = train_hgb.run_hgb_training(
        file_path=fake_dataset,
        year_t=2022,
        year_t1=2023,
        out_dir=tmp_path / "models_real",
        variants="default",
        enable_feature_engineering=False,
        enable_age_bucket=False,
        eval_holdout=False,
        strict=True,
    )
    model_path = Path(report["variants"]["default"]["model_path"])
    loaded = real_joblib.load(model_path)
    sample_raw = fake_frames[2022].loc[:, prep.get_expected_raw_feature_columns()].head(3)
    probs = loaded.predict_proba(sample_raw)
    assert probs.shape == (3, 2)


def test_train_hgb_holdout_requires_extra_flag_before_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _deps_should_not_run() -> dict[str, object]:
        raise RuntimeError("deps should not be called")

    monkeypatch.setattr(train_hgb, "_require_training_dependencies", _deps_should_not_run)
    with pytest.raises(ValueError, match="holdout pair"):
        train_hgb.run_hgb_training(
            year_t=2023,
            year_t1=2024,
            allow_nontrain_pair=True,
            allow_holdout_training=False,
        )


def test_train_hgb_main_exits_nonzero_for_disallowed_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_hgb, "setup_logging", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "--year-t",
            "2023",
            "--year-t1",
            "2024",
            "--allow-nontrain-pair",
            "1",
            "--allow-holdout-training",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        train_hgb.main()
    assert exc.value.code == 1


def test_train_hgb_cv_only_official_pair_before_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _deps_should_not_run() -> dict[str, object]:
        raise RuntimeError("deps should not be called")

    monkeypatch.setattr(train_hgb, "_require_training_dependencies", _deps_should_not_run)
    with pytest.raises(ValueError, match="restricted to official training pair"):
        train_hgb.run_hgb_training(
            year_t=2022,
            year_t1=2024,
            allow_nontrain_pair=True,
            enable_cv=True,
        )
