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


class DummyCoefEstimator:
    coef_ = np.array([[1.0, -2.0, 0.5]], dtype=float)


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


def test_sample_holdout_branches_and_rank_helpers() -> None:
    X = pd.DataFrame({"a": range(8), "b": range(8)})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype="Int64")

    sampled_X, sampled_y, info, notes = explainability._sample_holdout(
        X, y, max_rows=4, seed=42
    )
    assert info["sampled"] is True
    assert info["stratified"] is True
    assert len(sampled_X) == 4
    assert len(sampled_y) == 4
    assert "sampling_stratified_by_target" in notes

    y_single = pd.Series([1, 1, 1, 1, 1, 1, 1, 1], dtype="Int64")
    sampled_X2, sampled_y2, info2, notes2 = explainability._sample_holdout(
        X, y_single, max_rows=3, seed=7
    )
    assert info2["sampled"] is True
    assert info2["stratified"] is False
    assert len(sampled_X2) == 3
    assert len(sampled_y2) == 3
    assert "sampling_random_single_class" in notes2

    top_rows, top_notes = explainability._rank_top_features(
        ["f0", "f1", "f2"],
        np.array([0.2, np.nan, 0.5], dtype=float),
        top_k=2,
    )
    assert len(top_rows) == 2
    assert top_rows[0]["feature"] == "f2"
    assert top_notes == []

    top_rows_mismatch, top_notes_mismatch = explainability._rank_top_features(
        ["f0", "f1"],
        np.array([0.9, 0.8, 0.7], dtype=float),
        top_k=3,
    )
    assert len(top_rows_mismatch) == 2
    assert "importance_length_mismatch_truncated" in top_notes_mismatch


def test_importance_resolution_and_feature_name_fallbacks() -> None:
    X_pre = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 1.0], [3.0, 4.0, 0.0]])
    y_true = np.array([0, 1, 1], dtype=int)

    values1, method1, notes1 = explainability._resolve_importances(
        DummyEstimator(),
        X_pre,
        y_true,
        seed=42,
    )
    assert method1 == "feature_importances_"
    assert len(values1) == 3
    assert notes1 == []

    values2, method2, notes2 = explainability._resolve_importances(
        DummyCoefEstimator(),
        X_pre,
        y_true,
        seed=42,
    )
    assert method2 == "coef_abs"
    assert np.allclose(values2, np.array([1.0, 2.0, 0.5], dtype=float))
    assert notes2 == []

    class NoFeatureNames:
        pass

    names, name_notes = explainability._get_feature_names_from_preprocessor(
        NoFeatureNames(),
        n_features=3,
    )
    assert names == ["f0", "f1", "f2"]
    assert "feature_names_fallback" in name_notes


def test_error_slice_and_decile_helpers_cover_edge_cases() -> None:
    X_raw = pd.DataFrame(
        {
            "Gênero": ["Feminino", "Masculino", "Feminino", "Masculino"],
            "Instituição de ensino": ["Pública", "Privada", "Pública", "Privada"],
            "Fase": ["Fase 1", "Fase 1", "Fase 2", "Fase 2"],
            "Fase_Ideal": ["Fase 1", "Fase 1", "Fase 2", "Fase 2"],
            "Pedra_Ano": ["A", "A", "B", "B"],
            "Turma": ["T1", "T2", "T1", "T2"],
            "Idade": [10, 12, 16, np.nan],
            "Defasagem": [-3, -1, 0, np.nan],
        }
    )
    X_model = pd.DataFrame({"avg_grades": [6.0, 6.5, 7.0, 7.5]})
    y_true = np.array([1, 0, 1, 0], dtype=int)
    y_pred = np.array([1, 1, 0, 0], dtype=int)
    by_slice, notes = explainability._build_error_slices(
        X_raw=X_raw,
        X_model=X_model,
        y_true=y_true,
        y_pred=y_pred,
    )
    assert "Gênero" in by_slice
    assert "Idade_bin" in by_slice
    assert "Defasagem_bin" in by_slice
    assert "avg_grades_quartile" in by_slice
    assert notes == []

    score_rows = explainability._score_decile_analysis(
        scores=np.array([0.5, 0.5, 0.5, 0.5], dtype=float),
        y_true=y_true,
        y_pred=y_pred,
    )
    assert len(score_rows) == 1
    assert score_rows[0]["decile"] == "D00"


def test_parse_args_and_main_paths(monkeypatch, tmp_path: Path) -> None:
    args = explainability._parse_args.__wrapped__ if hasattr(explainability._parse_args, "__wrapped__") else None
    _ = args  # appease lint in case decorator is absent

    monkeypatch.setattr(
        "sys.argv",
        [
            "explainability.py",
            "--model-dir",
            "app/model",
            "--dataset-path",
            "dataset.xlsx",
            "--year-t",
            "2023",
            "--year-t1",
            "2024",
            "--out-json",
            "artifacts/x.json",
            "--out-md",
            "artifacts/x.md",
            "--top-k",
            "10",
            "--max-rows",
            "123",
            "--seed",
            "7",
            "--no-markdown",
        ],
    )
    parsed = explainability._parse_args()
    assert parsed.model_dir == "app/model"
    assert parsed.dataset_path == "dataset.xlsx"
    assert parsed.top_k == 10
    assert parsed.max_rows == 123
    assert parsed.seed == 7
    assert parsed.no_markdown is True

    calls: dict[str, object] = {}

    def fake_setup_logging() -> None:
        calls["setup"] = True

    def fake_run_explainability(**kwargs):
        calls["kwargs"] = kwargs
        return {"status": "PASS"}

    monkeypatch.setattr(explainability, "setup_logging", fake_setup_logging)
    monkeypatch.setattr(explainability, "run_explainability", fake_run_explainability)
    monkeypatch.setattr(
        explainability,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "model_dir": "app/model",
                "dataset_path": "dataset.xlsx",
                "year_t": 2023,
                "year_t1": 2024,
                "out_json": str(tmp_path / "out.json"),
                "out_md": str(tmp_path / "out.md"),
                "out_csv": None,
                "top_k": 10,
                "max_rows": 100,
                "seed": 42,
                "no_markdown": False,
            },
        )(),
    )
    assert explainability.main() == 0
    assert calls.get("setup") is True
    assert "kwargs" in calls

    monkeypatch.setattr(explainability, "run_explainability", lambda **kwargs: {"status": "FAIL"})
    assert explainability.main() == 1

    monkeypatch.setattr(explainability, "run_explainability", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert explainability.main() == 1
