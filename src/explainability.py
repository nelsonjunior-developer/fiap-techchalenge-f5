"""Local explainability report for champion serving model (offline, privacy-safe)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data import get_default_dataset_path, load_pede_workbook_with_metadata, make_temporal_pairs
from src.metrics import compute_classification_metrics_at_threshold, summarize_proba
from src.privacy import find_forbidden_json_keys
from src.preprocessing import get_expected_raw_feature_columns
from src.serving_context import extract_model_identity, extract_operational_threshold, load_serving_metadata
from src.training_utils import build_raw_from_ids
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_PROMOTED_MISSING_MESSAGE = "Promoted serving model not found. Run src.promote_model first."
_DEFAULT_MAX_ROWS = 1000
_DEFAULT_TOP_K = 20
_MAX_ROWS_PERMUTATION = 500


def _require_explain_dependencies() -> dict[str, Any]:
    try:
        import joblib
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "joblib is required for explainability. Install requirements-dev.txt"
        ) from exc
    return {"joblib": joblib}


def _resolve_dataset_path(dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        return get_default_dataset_path()
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return path


def _resolve_model_artifacts(model_dir: str | Path) -> dict[str, Path]:
    base = Path(model_dir)
    return {
        "model_dir": base,
        "model_path": base / "model.joblib",
        "metadata_path": base / "metadata.json",
    }


def _extract_expected_raw_cols_from_model(model: Any) -> list[str]:
    try:
        raw_step = model.named_steps["raw_to_model"]
        raw_cols = list(
            getattr(raw_step, "expected_raw_cols_", getattr(raw_step, "expected_raw_cols", []))
        )
    except Exception:
        raw_cols = []
    cleaned = [str(col).strip() for col in raw_cols if str(col).strip()]
    if cleaned:
        return cleaned
    return get_expected_raw_feature_columns()


def _resolve_expected_raw_cols(metadata: Mapping[str, Any] | None, model: Any) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    raw_cols = (metadata or {}).get("expected_raw_cols") if isinstance(metadata, Mapping) else None
    if isinstance(raw_cols, list):
        cleaned = [str(col).strip() for col in raw_cols if str(col).strip()]
        if cleaned:
            return cleaned, notes + ["expected_raw_cols_from_metadata"]
    notes.append("expected_raw_cols_fallback_from_model")
    return _extract_expected_raw_cols_from_model(model), notes


def _sample_holdout(
    X_raw: pd.DataFrame,
    y_true: pd.Series,
    *,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], list[str]]:
    if max_rows <= 0:
        raise ValueError("max_rows must be > 0.")

    n_total = int(len(X_raw))
    y = pd.Series(y_true).astype("Int64")
    notes: list[str] = []
    if n_total <= max_rows:
        return (
            X_raw,
            y,
            {"sampled": False, "n_before": n_total, "n_after": n_total, "stratified": False},
            notes,
        )

    if int(y.nunique(dropna=True)) <= 1:
        sampled_idx = X_raw.sample(n=max_rows, random_state=seed).index
        notes.append("sampling_random_single_class")
        return (
            X_raw.loc[sampled_idx].reset_index(drop=True),
            y.loc[sampled_idx].reset_index(drop=True),
            {"sampled": True, "n_before": n_total, "n_after": int(max_rows), "stratified": False},
            notes,
        )

    class_counts = y.value_counts().sort_index()
    quotas: dict[int, int] = {}
    remainders: dict[int, float] = {}
    for cls, count in class_counts.items():
        proportion = float(count / n_total)
        raw_quota = proportion * max_rows
        quota = int(np.floor(raw_quota))
        if count > 0:
            quota = max(1, quota)
        quota = min(quota, int(count))
        quotas[int(cls)] = quota
        remainders[int(cls)] = raw_quota - np.floor(raw_quota)

    allocated = int(sum(quotas.values()))
    while allocated < max_rows:
        candidates = []
        for cls, count in class_counts.items():
            cls_int = int(cls)
            if quotas[cls_int] < int(count):
                candidates.append((remainders[cls_int], cls_int))
        if not candidates:
            break
        candidates.sort(reverse=True)
        cls_to_add = candidates[0][1]
        quotas[cls_to_add] += 1
        allocated += 1

    while allocated > max_rows:
        removable = [(q, cls) for cls, q in quotas.items() if q > 1]
        if not removable:
            break
        removable.sort(reverse=True)
        cls_to_remove = removable[0][1]
        quotas[cls_to_remove] -= 1
        allocated -= 1

    sampled_parts: list[pd.Index] = []
    for cls in sorted(quotas):
        class_idx = y[y == cls].index
        n_take = int(min(quotas[cls], len(class_idx)))
        sampled_cls = pd.Series(class_idx).sample(n=n_take, random_state=seed + cls).to_list()
        sampled_parts.append(pd.Index(sampled_cls))

    sampled_idx = pd.Index([])
    for idx in sampled_parts:
        sampled_idx = sampled_idx.append(idx)
    sampled_idx = pd.Index(sampled_idx).drop_duplicates()
    if len(sampled_idx) > max_rows:
        sampled_idx = sampled_idx[:max_rows]
    if len(sampled_idx) < max_rows:
        remaining_idx = X_raw.index.difference(sampled_idx)
        n_fill = min(max_rows - len(sampled_idx), len(remaining_idx))
        if n_fill > 0:
            fill_idx = pd.Series(remaining_idx).sample(n=n_fill, random_state=seed + 99).to_list()
            sampled_idx = sampled_idx.append(pd.Index(fill_idx))
            notes.append("sampling_fill_from_remaining")

    sampled_idx = pd.Index(sampled_idx).drop_duplicates()
    notes.append("sampling_stratified_by_target")
    X_out = X_raw.loc[sampled_idx].reset_index(drop=True)
    y_out = y.loc[sampled_idx].reset_index(drop=True)
    return (
        X_out,
        y_out,
        {"sampled": True, "n_before": n_total, "n_after": int(len(X_out)), "stratified": True},
        notes,
    )


def _as_dense_2d(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix after preprocessing, got shape={arr.shape}")
    return arr


def _get_feature_names_from_preprocessor(preprocessor: Any, n_features: int) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = [str(name).strip() for name in preprocessor.get_feature_names_out()]
            cleaned = [name for name in names if name]
            if cleaned and len(cleaned) == n_features:
                return cleaned, notes
            notes.append("feature_names_mismatch_using_fallback")
        except Exception:
            notes.append("feature_names_extraction_failed_using_fallback")
    return [f"f{i}" for i in range(n_features)], notes + ["feature_names_fallback"]


def _resolve_pipeline_components(model_pipeline: Any) -> tuple[Any, Any, Any]:
    named_steps = getattr(model_pipeline, "named_steps", {})
    if not isinstance(named_steps, Mapping):
        raise ValueError("Loaded model artifact must be a sklearn-like pipeline with named_steps.")
    if "model" not in named_steps:
        raise ValueError("Pipeline missing 'model' step.")
    return named_steps.get("raw_to_model"), named_steps.get("preprocessor"), named_steps["model"]


def _prepare_transformed_matrix(
    model_pipeline: Any,
    X_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    notes: list[str] = []
    raw_to_model, preprocessor, _ = _resolve_pipeline_components(model_pipeline)

    if raw_to_model is not None:
        X_model = raw_to_model.transform(X_raw)
        if not isinstance(X_model, pd.DataFrame):
            X_model = pd.DataFrame(X_model)
    else:
        X_model = X_raw.copy()
        notes.append("raw_to_model_step_missing_using_raw_frame")

    if preprocessor is None:
        notes.append("preprocessor_step_missing_using_model_frame_direct")
        X_pre = _as_dense_2d(X_model.to_numpy())
        feature_names = [str(col).strip() for col in X_model.columns]
        if len(feature_names) != X_pre.shape[1]:
            feature_names = [f"f{i}" for i in range(X_pre.shape[1])]
            notes.append("feature_names_fallback")
        return X_model, X_pre, feature_names, notes

    transformed = preprocessor.transform(X_model)
    X_pre = _as_dense_2d(transformed)
    feature_names, feature_notes = _get_feature_names_from_preprocessor(preprocessor, X_pre.shape[1])
    notes.extend(feature_notes)
    return X_model, X_pre, feature_names, notes


def _resolve_importances(
    estimator: Any,
    X_pre: np.ndarray,
    y_true: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, str, list[str]]:
    notes: list[str] = []
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(getattr(estimator, "feature_importances_"), dtype=float).ravel()
        return values, "feature_importances_", notes

    if hasattr(estimator, "coef_"):
        coef = np.asarray(getattr(estimator, "coef_"), dtype=float)
        if coef.ndim == 1:
            values = np.abs(coef).ravel()
        else:
            values = np.mean(np.abs(coef), axis=0).ravel()
        return values, "coef_abs", notes

    try:
        from sklearn.inspection import permutation_importance
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "Permutation importance fallback requires scikit-learn."
        ) from exc

    X_perm = X_pre
    y_perm = np.asarray(y_true, dtype=int)
    if len(X_perm) > _MAX_ROWS_PERMUTATION:
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(len(X_perm), size=_MAX_ROWS_PERMUTATION, replace=False)
        X_perm = X_perm[sampled_idx]
        y_perm = y_perm[sampled_idx]
        notes.append("permutation_importance_sampled")

    result = permutation_importance(
        estimator,
        X_perm,
        y_perm,
        scoring="average_precision",
        n_repeats=3,
        random_state=seed,
    )
    values = np.asarray(result.importances_mean, dtype=float).ravel()
    notes.append("used_permutation_importance_fallback")
    return values, "permutation_importance", notes


def _rank_top_features(
    feature_names: Sequence[str],
    importances: np.ndarray,
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")

    values = np.asarray(importances, dtype=float).ravel()
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    names = [str(name).strip() or f"f{i}" for i, name in enumerate(feature_names)]
    if len(values) != len(names):
        n = min(len(values), len(names))
        values = values[:n]
        names = names[:n]
        notes.append("importance_length_mismatch_truncated")

    if len(values) == 0:
        return [], notes + ["empty_importance_vector"]

    ranked_idx = np.argsort(-values)
    top_idx = ranked_idx[: min(top_k, len(ranked_idx))]
    top_features: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_idx, start=1):
        top_features.append(
            {
                "rank": int(rank),
                "feature": str(names[int(idx)]),
                "importance": float(values[int(idx)]),
            }
        )
    return top_features, notes


def _normalize_group_series(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").fillna("MISSING").str.strip()
    normalized = normalized.replace({"": "MISSING"})
    return normalized.fillna("MISSING")


def _bin_age(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    binned = pd.cut(
        numeric,
        bins=[-np.inf, 10, 14, 18, np.inf],
        labels=["<=10", "11-14", "15-18", "19+"],
        include_lowest=True,
    )
    return pd.Series(binned.astype("string")).fillna("MISSING")


def _bin_defasagem(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")

    def _label(v: Any) -> str:
        if pd.isna(v):
            return "MISSING"
        value = float(v)
        if value <= -3:
            return "<=-3"
        if -2 <= value <= -1:
            return "-2..-1"
        if value >= 0:
            return ">=0"
        return "OUTRO"

    return numeric.map(_label).astype("string")


def _bin_quartiles(values: pd.Series, *, label_prefix: str) -> pd.Series | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if int(numeric.nunique(dropna=True)) < 2:
        return None
    try:
        quartiles = pd.qcut(
            numeric,
            q=4,
            labels=[f"{label_prefix}_Q1", f"{label_prefix}_Q2", f"{label_prefix}_Q3", f"{label_prefix}_Q4"],
            duplicates="drop",
        )
    except ValueError:
        return None
    return pd.Series(quartiles.astype("string")).fillna("MISSING")


def _safe_ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _aggregate_slice(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "group": _normalize_group_series(groups),
            "y_true": np.asarray(y_true, dtype=int),
            "y_pred": np.asarray(y_pred, dtype=int),
        }
    )
    rows: list[dict[str, Any]] = []
    for group_name, group_df in frame.groupby("group", dropna=False):
        y_t = group_df["y_true"].to_numpy(dtype=int)
        y_p = group_df["y_pred"].to_numpy(dtype=int)
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        n_total = int(len(group_df))
        rows.append(
            {
                "group": str(group_name),
                "n_total": n_total,
                "prevalence": float(np.mean(y_t)) if n_total else 0.0,
                "predicted_positive_rate": float(np.mean(y_p)) if n_total else 0.0,
                "recall_in_slice": _safe_ratio(tp, tp + fn),
                "precision_in_slice": _safe_ratio(tp, tp + fp),
                "fn_rate_in_slice": _safe_ratio(fn, fn + tp),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    rows.sort(key=lambda item: int(item["n_total"]), reverse=True)
    return rows[:top_n]


def _score_decile_analysis(
    *,
    scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    if len(scores) == 0:
        return []
    frame = pd.DataFrame(
        {
            "score": np.asarray(scores, dtype=float),
            "y_true": np.asarray(y_true, dtype=int),
            "y_pred": np.asarray(y_pred, dtype=int),
        }
    )
    unique_scores = int(frame["score"].nunique(dropna=True))
    if unique_scores <= 1:
        frame["decile"] = "D00"
    else:
        n_bins = min(10, unique_scores)
        q = pd.qcut(frame["score"], q=n_bins, labels=False, duplicates="drop")
        frame["decile"] = q.map(lambda v: f"D{int(v):02d}" if pd.notna(v) else "D00")

    rows: list[dict[str, Any]] = []
    for decile, group_df in frame.groupby("decile", dropna=False):
        y_t = group_df["y_true"].to_numpy(dtype=int)
        y_p = group_df["y_pred"].to_numpy(dtype=int)
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        n_total = int(len(group_df))
        rows.append(
            {
                "decile": str(decile),
                "n_total": n_total,
                "score_min": float(group_df["score"].min()),
                "score_max": float(group_df["score"].max()),
                "prevalence": float(np.mean(y_t)) if n_total else 0.0,
                "predicted_positive_rate": float(np.mean(y_p)) if n_total else 0.0,
                "recall_in_decile": _safe_ratio(tp, tp + fn),
                "precision_in_decile": _safe_ratio(tp, tp + fp),
            }
        )
    rows.sort(key=lambda item: item["decile"])
    return rows


def _build_error_slices(
    *,
    X_raw: pd.DataFrame,
    X_model: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    notes: list[str] = []
    by_slice: dict[str, list[dict[str, Any]]] = {}

    candidate_cols = [
        "Gênero",
        "Instituição de ensino",
        "Fase",
        "Fase_Ideal",
        "Pedra_Ano",
        "Turma",
    ]
    for col in candidate_cols:
        if col in X_raw.columns:
            by_slice[col] = _aggregate_slice(
                y_true=y_true,
                y_pred=y_pred,
                groups=X_raw[col],
                top_n=10,
            )

    if "Idade" in X_raw.columns:
        by_slice["Idade_bin"] = _aggregate_slice(
            y_true=y_true,
            y_pred=y_pred,
            groups=_bin_age(X_raw["Idade"]),
            top_n=10,
        )

    if "Defasagem" in X_raw.columns:
        by_slice["Defasagem_bin"] = _aggregate_slice(
            y_true=y_true,
            y_pred=y_pred,
            groups=_bin_defasagem(X_raw["Defasagem"]),
            top_n=10,
        )

    if "avg_grades" in X_model.columns:
        quartiles = _bin_quartiles(X_model["avg_grades"], label_prefix="avg_grades")
        if quartiles is not None:
            by_slice["avg_grades_quartile"] = _aggregate_slice(
                y_true=y_true,
                y_pred=y_pred,
                groups=quartiles,
                top_n=10,
            )
        else:
            notes.append("avg_grades_quartile_unavailable")

    return by_slice, notes


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def run_explainability(
    *,
    model_dir: str | Path = "app/model",
    dataset_path: str | Path | None = None,
    year_t: int = 2023,
    year_t1: int = 2024,
    out_json: str | Path = "artifacts/explainability_report.json",
    out_md: str | Path = "artifacts/explainability_report.md",
    out_csv: str | Path | None = None,
    top_k: int = _DEFAULT_TOP_K,
    max_rows: int = _DEFAULT_MAX_ROWS,
    seed: int = 42,
    write_markdown: bool = True,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")

    deps = _require_explain_dependencies()
    artifacts = _resolve_model_artifacts(model_dir)
    model_path = artifacts["model_path"]
    metadata_path = artifacts["metadata_path"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"{_PROMOTED_MISSING_MESSAGE} model={model_path.exists()} metadata={metadata_path.exists()}"
        )

    model_pipeline = deps["joblib"].load(model_path)
    metadata = load_serving_metadata(metadata_path)
    threshold, threshold_notes = extract_operational_threshold(metadata)
    identity, identity_notes = extract_model_identity(metadata)
    expected_raw_cols, expected_notes = _resolve_expected_raw_cols(metadata, model_pipeline)

    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    yearly_frames, _, _ = load_pede_workbook_with_metadata(file_path=resolved_dataset_path)
    if int(year_t) not in yearly_frames or int(year_t1) not in yearly_frames:
        raise ValueError(
            f"Years not available in dataset for explainability: {year_t}->{year_t1}"
        )

    _, y_holdout, ids = make_temporal_pairs(
        yearly_frames[int(year_t)],
        yearly_frames[int(year_t1)],
        int(year_t),
        int(year_t1),
    )
    X_raw_full = build_raw_from_ids(
        df_year_t=yearly_frames[int(year_t)],
        ids=ids,
        expected_raw_cols=expected_raw_cols,
    )
    X_raw, y_true, sampling_info, sampling_notes = _sample_holdout(
        X_raw_full,
        y_holdout,
        max_rows=max_rows,
        seed=seed,
    )

    if not hasattr(model_pipeline, "predict_proba"):
        raise ValueError("Serving model does not expose predict_proba.")
    proba_matrix = np.asarray(model_pipeline.predict_proba(X_raw), dtype=float)
    if proba_matrix.ndim != 2 or proba_matrix.shape[1] < 2:
        raise ValueError("model predict_proba output must have shape (n, 2)")
    scores = np.asarray(proba_matrix[:, 1], dtype=float)
    y_true_np = pd.Series(y_true).astype(int).to_numpy()
    y_pred = (scores >= float(threshold)).astype(int)

    metrics_payload = compute_classification_metrics_at_threshold(
        y_true=y_true_np,
        y_proba=scores,
        threshold=float(threshold),
    )
    score_summary = summarize_proba(scores)

    X_model, X_pre, feature_names, transform_notes = _prepare_transformed_matrix(model_pipeline, X_raw)
    _, _, estimator = _resolve_pipeline_components(model_pipeline)
    importances, method, importance_notes = _resolve_importances(
        estimator,
        X_pre,
        y_true_np,
        seed=seed,
    )
    top_features, top_notes = _rank_top_features(feature_names, importances, top_k=top_k)

    by_slice, slice_notes = _build_error_slices(
        X_raw=X_raw,
        X_model=X_model,
        y_true=y_true_np,
        y_pred=y_pred,
    )
    by_score_decile = _score_decile_analysis(scores=scores, y_true=y_true_np, y_pred=y_pred)

    notes: list[str] = [
        "explainability_global_importance_is_not_causal",
        "error_analysis_is_aggregated_no_row_level_records",
        "holdout_read_only_analysis",
    ]
    notes.extend(threshold_notes)
    notes.extend(identity_notes)
    notes.extend(expected_notes)
    notes.extend(sampling_notes)
    notes.extend(transform_notes)
    notes.extend(importance_notes)
    notes.extend(top_notes)
    notes.extend(slice_notes)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "pair": {"year_t": int(year_t), "year_t1": int(year_t1)},
        "model": {
            "dir": str(artifacts["model_dir"]),
            "model_path_basename": model_path.name,
            "metadata_path_basename": metadata_path.name,
            "model_version": str(identity["model_version"]),
            "model_family": str(identity["model_family"]),
            "variant": str(identity["variant"]),
        },
        "n_evaluated": int(len(X_raw)),
        "sampling": sampling_info,
        "contract": {"expected_raw_cols_count": int(len(expected_raw_cols))},
        "threshold_operational": float(threshold),
        "pred_proba_summary": score_summary,
        "metrics_at_operational_threshold": metrics_payload,
        "confusion_matrix": metrics_payload.get("confusion_matrix", {}),
        "global_importance": {
            "method": method,
            "n_features_total": int(min(len(feature_names), len(np.asarray(importances).ravel()))),
            "top_k": top_features,
        },
        "error_analysis": {
            "by_slice": by_slice,
            "by_score_decile": by_score_decile,
        },
        "notes": list(dict.fromkeys(str(note) for note in notes if str(note).strip())),
    }

    forbidden_present = find_forbidden_json_keys(report)
    if forbidden_present:
        report["status"] = "FAIL"
        report.setdefault("errors", [])
        report["errors"].append(
            f"Privacy check failed: forbidden keys found: {forbidden_present}"
        )

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if out_csv:
        out_csv_path = Path(out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report["global_importance"]["top_k"]).to_csv(out_csv_path, index=False)

    if write_markdown:
        lines = [
            "# Explainability Report (Local)",
            "",
            f"Status: **{report['status']}**",
            "",
            f"Pair: `{int(year_t)}->{int(year_t1)}`",
            f"Model: `{report['model']['model_family']}/{report['model']['variant']}`",
            f"Model version: `{report['model']['model_version']}`",
            f"Threshold operacional: `{float(report['threshold_operational']):.4f}`",
            f"N avaliado: `{int(report['n_evaluated'])}`",
            "",
            "## Métricas (holdout)",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Recall | {_fmt_metric(metrics_payload.get('recall'))} |",
            f"| PR-AUC | {_fmt_metric(metrics_payload.get('pr_auc'))} |",
            f"| Precision | {_fmt_metric(metrics_payload.get('precision'))} |",
            f"| F1 | {_fmt_metric(metrics_payload.get('f1'))} |",
            f"| ROC-AUC | {_fmt_metric(metrics_payload.get('roc_auc'))} |",
            "",
            "## Top Features (global importance)",
            "",
            f"Método: `{method}`",
            "",
            "| Rank | Feature | Importance |",
            "|---:|---|---:|",
        ]
        for row in report["global_importance"]["top_k"]:
            lines.append(
                f"| {int(row['rank'])} | `{row['feature']}` | {float(row['importance']):.6f} |"
            )
        lines.extend(
            [
                "",
                "## Error Analysis (agregado)",
                "",
                "Resumo por score decile:",
                "",
                "| Decile | n | Score min | Score max | Recall | Precision |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["error_analysis"]["by_score_decile"]:
            lines.append(
                f"| {row['decile']} | {int(row['n_total'])} | "
                f"{float(row['score_min']):.4f} | {float(row['score_max']):.4f} | "
                f"{_fmt_metric(row.get('recall_in_decile'))} | {_fmt_metric(row.get('precision_in_decile'))} |"
            )
        lines.extend(
            [
                "",
                "## Notas",
                "- Importância global não implica causalidade.",
                "- Relatório agregado e privacy-safe (sem IDs/RA, sem registros individuais).",
            ]
        )
        lines.extend([f"- {note}" for note in report.get("notes", [])])

        out_md_path = Path(out_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate local explainability report for champion serving model "
            "(global importances + aggregated error analysis)."
        )
    )
    parser.add_argument("--model-dir", type=str, default="app/model", help="Serving model directory.")
    parser.add_argument("--dataset-path", type=str, default=None, help="PEDE XLSX dataset path.")
    parser.add_argument("--year-t", type=int, default=2023, help="Feature year (t).")
    parser.add_argument("--year-t1", type=int, default=2024, help="Label year (t+1).")
    parser.add_argument(
        "--out-json",
        type=str,
        default="artifacts/explainability_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="artifacts/explainability_report.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional CSV path for top feature importances.",
    )
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K, help="Top-k feature importances.")
    parser.add_argument("--max-rows", type=int, default=_DEFAULT_MAX_ROWS, help="Max rows for holdout analysis.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable Markdown report output.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = _parse_args()
    try:
        report = run_explainability(
            model_dir=args.model_dir,
            dataset_path=args.dataset_path,
            year_t=int(args.year_t),
            year_t1=int(args.year_t1),
            out_json=args.out_json,
            out_md=args.out_md,
            out_csv=args.out_csv,
            top_k=int(args.top_k),
            max_rows=int(args.max_rows),
            seed=int(args.seed),
            write_markdown=not bool(args.no_markdown),
        )
        _logger.info(
            "Explainability report generated | status=%s pair=%s->%s out_json=%s",
            report.get("status"),
            int(args.year_t),
            int(args.year_t1),
            args.out_json,
        )
        if report.get("status") == "FAIL":
            return 1
        return 0
    except Exception as exc:
        _logger.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
