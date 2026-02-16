from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.preprocessing import build_preprocessing_bundle
from src.train_pipeline import build_model_pipeline


def _build_raw_frame(expected_raw_cols: list[str], n_rows: int = 8) -> pd.DataFrame:
    data: dict[str, object] = {}
    numeric_cols = {
        "Ano ingresso",
        "Cf",
        "Cg",
        "Ct",
        "Defasagem",
        "IAA",
        "IAN",
        "IDA",
        "IEG",
        "INDE",
        "INDE 22",
        "IPS",
        "IPV",
        "Idade",
        "Ing",
        "Mat",
        "Nº Av",
        "Por",
    }
    for idx, col in enumerate(expected_raw_cols):
        if col in numeric_cols:
            values = np.linspace(1, n_rows, n_rows, dtype=float) + idx
            if n_rows > 2:
                values[2] = np.nan
            data[col] = pd.Series(values, dtype="Float64")
        else:
            values = ["A", "B", "A", "C", "B", "A", "C", "B"][:n_rows]
            series = pd.Series(values, dtype="string")
            if n_rows > 3:
                series.iloc[3] = pd.NA
            data[col] = series
    return pd.DataFrame(data).loc[:, expected_raw_cols]


def _build_target(n_rows: int) -> pd.Series:
    values = np.array([0, 1] * (n_rows // 2 + 1))[:n_rows]
    return pd.Series(values, dtype="Int64")


def test_build_pipeline_has_expected_steps() -> None:
    model = LogisticRegression(max_iter=200)
    pipeline = build_model_pipeline(
        model=model,
        year_t=2022,
        scaler_strategy="standard",
        enable_feature_engineering=False,
        strict_raw=True,
    )
    assert list(pipeline.named_steps.keys()) == [
        "raw_to_model",
        "preprocessor",
        "model",
    ]


def test_pipeline_fit_predict_proba_smoke_logreg() -> None:
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=False,
    )
    expected_raw_cols = list(bundle["expected_raw_cols"])
    X_raw = _build_raw_frame(expected_raw_cols, n_rows=10)
    y = _build_target(len(X_raw))

    pipeline = build_model_pipeline(
        model=LogisticRegression(max_iter=200),
        year_t=2022,
        scaler_strategy="standard",
        enable_feature_engineering=False,
        strict_raw=True,
    )
    pipeline.fit(X_raw, y)
    probs = pipeline.predict_proba(X_raw)
    assert probs.shape == (len(X_raw), 2)


def test_pipeline_allows_extra_non_suspicious_cols_raw() -> None:
    bundle = build_preprocessing_bundle(
        numeric_scaler="none",
        enable_feature_engineering=False,
    )
    expected_raw_cols = list(bundle["expected_raw_cols"])
    X_raw = _build_raw_frame(expected_raw_cols, n_rows=8)
    X_raw["extra_ok"] = pd.Series(["ok"] * len(X_raw), dtype="string")
    y = _build_target(len(X_raw))

    pipeline = build_model_pipeline(
        model=HistGradientBoostingClassifier(random_state=42),
        year_t=2023,
        scaler_strategy="none",
        enable_feature_engineering=False,
        strict_raw=True,
    )
    pipeline.fit(X_raw, y)
    preds = pipeline.predict(X_raw)
    assert preds.shape == (len(X_raw),)


def test_pipeline_blocks_suspicious_extra_cols_raw_deterministic() -> None:
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=False,
    )
    expected_raw_cols = list(bundle["expected_raw_cols"])
    X_raw = _build_raw_frame(expected_raw_cols, n_rows=8)
    X_raw["Defasagem_t1"] = pd.Series(np.linspace(-1, 1, len(X_raw)), dtype="Float64")
    y = _build_target(len(X_raw))

    pipeline = build_model_pipeline(
        model=LogisticRegression(max_iter=200),
        year_t=2022,
        scaler_strategy="standard",
        enable_feature_engineering=False,
        strict_raw=True,
    )
    with pytest.raises(ValueError, match="(?i)leakage-like"):
        pipeline.fit(X_raw, y)


def test_pipeline_joblib_roundtrip(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    bundle = build_preprocessing_bundle(
        numeric_scaler="standard",
        enable_feature_engineering=False,
    )
    expected_raw_cols = list(bundle["expected_raw_cols"])
    X_raw = _build_raw_frame(expected_raw_cols, n_rows=10)
    y = _build_target(len(X_raw))

    pipeline = build_model_pipeline(
        model=LogisticRegression(max_iter=200),
        year_t=2022,
        scaler_strategy="standard",
        enable_feature_engineering=False,
        strict_raw=True,
    )
    pipeline.fit(X_raw, y)

    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)
    probs = loaded.predict_proba(X_raw)
    assert probs.shape == (len(X_raw), 2)
