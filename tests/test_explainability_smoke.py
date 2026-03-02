from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.explainability as explainability
from src.privacy import find_forbidden_json_keys


class DummyRawToModel:
    def __init__(self) -> None:
        self.expected_raw_cols_ = ["Mat", "Por", "Idade", "Gênero", "Defasagem"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        avg = (
            pd.to_numeric(X["Mat"], errors="coerce").fillna(0.0)
            + pd.to_numeric(X["Por"], errors="coerce").fillna(0.0)
        ) / 2.0
        return pd.DataFrame(
            {
                "avg_grades": avg.astype(float),
                "feat_num": pd.to_numeric(X["Idade"], errors="coerce").fillna(0.0).astype(float),
                "feat_cat": X["Gênero"].astype("string").fillna("MISSING"),
            }
        )


class DummyPreprocessor:
    def get_feature_names_out(self):
        return np.array(["avg_grades", "feat_num", "feat_cat__Feminino"], dtype=object)

    def transform(self, X_model: pd.DataFrame) -> np.ndarray:
        cat = (X_model["feat_cat"].astype(str) == "Feminino").astype(float).to_numpy()
        return np.column_stack(
            [
                pd.to_numeric(X_model["avg_grades"], errors="coerce").fillna(0.0).to_numpy(),
                pd.to_numeric(X_model["feat_num"], errors="coerce").fillna(0.0).to_numpy(),
                cat,
            ]
        )


class DummyEstimator:
    feature_importances_ = np.array([0.65, 0.25, 0.10], dtype=float)


class DummyPipeline:
    def __init__(self) -> None:
        self.named_steps = {
            "raw_to_model": DummyRawToModel(),
            "preprocessor": DummyPreprocessor(),
            "model": DummyEstimator(),
        }

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        scores = np.clip(
            pd.to_numeric(X_raw["Idade"], errors="coerce").fillna(10.0).to_numpy(dtype=float) / 20.0,
            0.0,
            1.0,
        )
        return np.column_stack([1.0 - scores, scores])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _keys(payload):
    keys = set()
    if isinstance(payload, dict):
        for k, v in payload.items():
            keys.add(str(k).lower())
            keys |= _keys(v)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _keys(item)
    return keys


def _common_monkeypatches(monkeypatch, X_raw: pd.DataFrame, y_true: pd.Series, ids: pd.Series) -> None:
    monkeypatch.setattr(
        explainability,
        "_require_explain_dependencies",
        lambda: {"joblib": type("JoblibStub", (), {"load": staticmethod(lambda _: DummyPipeline())})()},
    )
    monkeypatch.setattr(
        explainability,
        "load_pede_workbook_with_metadata",
        lambda file_path: ({2023: pd.DataFrame(), 2024: pd.DataFrame()}, {}, {}),
    )
    monkeypatch.setattr(
        explainability,
        "make_temporal_pairs",
        lambda *args, **kwargs: (pd.DataFrame(), y_true, ids),
    )
    monkeypatch.setattr(
        explainability,
        "build_raw_from_ids",
        lambda **kwargs: X_raw,
    )


def test_explainability_smoke_generates_json_md_and_is_privacy_safe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"fake-joblib")
    _write_json(
        model_dir / "metadata.json",
        {
            "model_version": "2026-02-23T00-00-00Z",
            "model_family": "nonlinear_hgb",
            "variant": "default",
            "expected_raw_cols": ["Mat", "Por", "Idade", "Gênero", "Defasagem"],
            "threshold_policy": {"operational_fixed_threshold": 0.30},
        },
    )
    fake_dataset = tmp_path / "dataset.xlsx"
    fake_dataset.write_bytes(b"fake-xlsx")

    X_raw = pd.DataFrame(
        {
            "Mat": [6.0, 5.5, 7.0, 8.0],
            "Por": [5.5, 6.0, 7.5, 7.0],
            "Idade": [11, 13, 16, 17],
            "Gênero": ["Feminino", "Masculino", "Feminino", "Masculino"],
            "Defasagem": [-1, 0, -3, -2],
            "Instituição de ensino": ["Pública", "Pública", "Privada", "Pública"],
            "Fase": ["Fase Alfa", "Fase 1", "Fase 2", "Fase 2"],
            "Fase_Ideal": ["Fase 1", "Fase 1", "Fase 2", "Fase 2"],
            "Pedra_Ano": ["A", "A", "B", "B"],
            "Turma": ["T1", "T1", "T2", "T3"],
        }
    )
    y_true = pd.Series([1, 0, 1, 0], dtype="Int64")
    ids = pd.Series(["S1", "S2", "S3", "S4"], dtype="string")
    _common_monkeypatches(monkeypatch, X_raw=X_raw, y_true=y_true, ids=ids)

    out_json = tmp_path / "artifacts" / "explainability_report.json"
    out_md = tmp_path / "artifacts" / "explainability_report.md"
    out_csv = tmp_path / "artifacts" / "feature_importance.csv"
    report = explainability.run_explainability(
        model_dir=model_dir,
        dataset_path=fake_dataset,
        year_t=2023,
        year_t1=2024,
        out_json=out_json,
        out_md=out_md,
        out_csv=out_csv,
        top_k=2,
        max_rows=100,
        seed=42,
        write_markdown=True,
    )

    assert report["status"] == "PASS"
    assert report["global_importance"]["method"] == "feature_importances_"
    assert len(report["global_importance"]["top_k"]) <= 2
    assert out_json.exists()
    assert out_md.exists()
    assert out_csv.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert find_forbidden_json_keys(payload) == []
    keys = _keys(payload)
    assert "ra" not in keys
    assert "ids" not in keys
    assert "by_slice" in payload["error_analysis"]


def test_explainability_fallback_expected_raw_cols_from_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"fake-joblib")
    _write_json(
        model_dir / "metadata.json",
        {
            "model_version": "2026-02-23T00-00-00Z",
            "model_family": "nonlinear_hgb",
            "variant": "default",
            "expected_raw_cols": [],
            "threshold_policy": {"operational_fixed_threshold": 0.30},
        },
    )
    fake_dataset = tmp_path / "dataset.xlsx"
    fake_dataset.write_bytes(b"fake-xlsx")

    X_raw = pd.DataFrame(
        {
            "Mat": [6.0, 7.0, 8.0],
            "Por": [6.0, 7.0, 8.0],
            "Idade": [10, 12, 15],
            "Gênero": ["Feminino", "Masculino", "Feminino"],
            "Defasagem": [0, -1, -2],
        }
    )
    y_true = pd.Series([0, 1, 1], dtype="Int64")
    ids = pd.Series(["S1", "S2", "S3"], dtype="string")
    _common_monkeypatches(monkeypatch, X_raw=X_raw, y_true=y_true, ids=ids)

    out_json = tmp_path / "artifacts" / "explainability_report.json"
    report = explainability.run_explainability(
        model_dir=model_dir,
        dataset_path=fake_dataset,
        out_json=out_json,
        out_md=tmp_path / "artifacts" / "explainability_report.md",
        write_markdown=False,
    )

    assert report["status"] == "PASS"
    assert "expected_raw_cols_fallback_from_model" in report["notes"]
    assert int(report["contract"]["expected_raw_cols_count"]) > 0
