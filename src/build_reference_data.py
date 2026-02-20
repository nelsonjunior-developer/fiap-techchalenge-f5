"""Build and persist drift reference data from promoted model metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.data import get_default_dataset_path, load_pede_workbook_with_metadata, make_temporal_pairs
from src.metadata_schema import validate_metadata
from src.preprocessing import transform_raw_to_model_frame
from src.training_utils import build_raw_from_ids
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)
_FORBIDDEN_KEYS = {"ra", "ra_list", "ids", "student_ids", "students", "records"}
_FORBIDDEN_COLS_EXACT = {"ra", "nome_anon"}
_FORBIDDEN_COLS_PREFIX = ("avaliador",)
_TOP_VALUES_LIMIT = 10
_DEFAULT_REFERENCE_FILE = "reference_model_frame.csv"
_DEFAULT_PROFILE_FILE = "reference_profile.json"
_DEFAULT_META_FILE = "reference_meta.json"
_OPTIONAL_RAW_DIAGNOSTIC_FILE = "reference_raw_diagnostic.csv"
_PROMOTED_MISSING_MESSAGE = "Promoted model not found. Run src.promote_model first."


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload (expected object): {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys |= _collect_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_keys(item)
    return keys


def _parse_bool_flag(value: int) -> bool:
    return bool(int(value))


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_dataset_path(file_path: str | Path | None) -> Path:
    if file_path is None:
        return get_default_dataset_path()
    resolved = Path(file_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path not found: {resolved}")
    return resolved


def _assert_no_forbidden_columns(columns: list[str]) -> None:
    lower_cols = [str(column).strip().lower() for column in columns]
    forbidden_found = []
    for column in lower_cols:
        if column in _FORBIDDEN_COLS_EXACT:
            forbidden_found.append(column)
        elif any(column.startswith(prefix) for prefix in _FORBIDDEN_COLS_PREFIX):
            forbidden_found.append(column)
    if forbidden_found:
        raise ValueError(
            "MODEL frame contains forbidden columns (PII/IDs): "
            f"{sorted(set(forbidden_found))}"
        )


def _sample_without_replacement(
    indices: np.ndarray,
    sample_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    target = int(sample_size)
    if target <= 0:
        return np.array([], dtype=int)
    if target >= len(indices):
        return np.array(indices, dtype=int)
    return np.asarray(rng.choice(indices, size=target, replace=False), dtype=int)


def _stratified_indices(
    y_true: pd.Series | np.ndarray,
    max_rows: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    y = pd.Series(y_true).astype(int).to_numpy()
    n_rows = int(len(y))
    target_rows = int(max_rows)
    if target_rows <= 0:
        raise ValueError("max_rows must be > 0.")
    if n_rows <= target_rows:
        return np.arange(n_rows, dtype=int)

    rng = np.random.RandomState(int(random_state))
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Single-class fallback: deterministic random sample without stratification.
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        sampled = _sample_without_replacement(np.arange(n_rows, dtype=int), target_rows, rng)
        return np.sort(sampled)

    prevalence = float(len(pos_idx) / n_rows)
    n_pos_target = int(round(target_rows * prevalence))
    n_pos_target = max(1, min(n_pos_target, target_rows - 1))
    n_neg_target = target_rows - n_pos_target

    # Respect class availability and re-distribute remainder deterministically.
    n_pos_target = min(n_pos_target, len(pos_idx))
    n_neg_target = min(n_neg_target, len(neg_idx))
    remainder = target_rows - (n_pos_target + n_neg_target)
    if remainder > 0:
        pos_room = len(pos_idx) - n_pos_target
        add_pos = min(remainder, max(pos_room, 0))
        n_pos_target += add_pos
        remainder -= add_pos
    if remainder > 0:
        neg_room = len(neg_idx) - n_neg_target
        add_neg = min(remainder, max(neg_room, 0))
        n_neg_target += add_neg

    sampled_pos = _sample_without_replacement(pos_idx, n_pos_target, rng)
    sampled_neg = _sample_without_replacement(neg_idx, n_neg_target, rng)
    sampled = np.concatenate([sampled_pos, sampled_neg]).astype(int)
    if len(sampled) > target_rows:
        sampled = sampled[:target_rows]
    return np.sort(sampled)


def _numeric_summary(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    non_null = values[~np.isnan(values)]
    if non_null.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(non_null)),
        "std": float(np.std(non_null)),
        "min": float(np.min(non_null)),
        "p05": float(np.quantile(non_null, 0.05)),
        "p50": float(np.quantile(non_null, 0.50)),
        "p95": float(np.quantile(non_null, 0.95)),
        "max": float(np.max(non_null)),
    }


def _categorical_top_values(series: pd.Series, limit: int = _TOP_VALUES_LIMIT) -> list[dict[str, Any]]:
    normalized = pd.Series(series, copy=False).astype("string")
    normalized = normalized.fillna("_MISSING_").str.strip().replace("", "_MISSING_")
    counts = normalized.value_counts(dropna=False, normalize=True)
    top = counts.head(int(limit))
    payload = [
        {"value": str(value), "freq": float(freq)}
        for value, freq in top.items()
    ]
    other_freq = float(counts.iloc[int(limit) :].sum()) if len(counts) > int(limit) else 0.0
    if other_freq > 0.0:
        payload.append({"value": "_OTHER_", "freq": other_freq})
    return payload


def _infer_profile_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return "numeric"
        unique = {float(value) for value in numeric.unique().tolist()}
        if unique.issubset({0.0, 1.0}) and len(unique) <= 2:
            return "binary"
        return "numeric"
    normalized = pd.Series(series, copy=False).astype("string").dropna()
    if normalized.nunique(dropna=False) <= 2:
        return "binary"
    return "categorical"


def _build_reference_profile(
    X_model_sample: pd.DataFrame,
    *,
    n_train: int,
    n_pos: int,
    prevalence: float,
) -> dict[str, Any]:
    feature_profiles: list[dict[str, Any]] = []
    for feature in X_model_sample.columns:
        series = X_model_sample[feature]
        kind = _infer_profile_kind(series)
        entry: dict[str, Any] = {
            "feature": str(feature),
            "dtype_kind": kind,
            "missing_rate": float(series.isna().mean()) if len(series) else 0.0,
        }
        if kind == "numeric":
            entry["numeric_summary"] = _numeric_summary(series)
        else:
            entry["top_values"] = _categorical_top_values(series, limit=_TOP_VALUES_LIMIT)
        feature_profiles.append(entry)

    return {
        "overview": {
            "n_rows_reference": int(len(X_model_sample)),
            "n_features": int(X_model_sample.shape[1]),
            "train_prevalence": {
                "n_train": int(n_train),
                "n_pos": int(n_pos),
                "prevalence": float(prevalence),
            },
        },
        "features": feature_profiles,
    }


def _backup_existing_reference(
    *,
    out_dir: Path,
    target_paths: list[Path],
) -> Path | None:
    existing = [path for path in target_paths if path.exists()]
    if not existing:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = out_dir / "backups" / timestamp
    suffix = 1
    while backup_dir.exists():
        backup_dir = out_dir / "backups" / f"{timestamp}_{suffix:02d}"
        suffix += 1
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil_target = backup_dir / path.name
        shutil_target.write_bytes(path.read_bytes())
    return backup_dir


def _ensure_promoted_artifacts(model_dir: Path) -> tuple[Path, Path]:
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(_PROMOTED_MISSING_MESSAGE)
    return model_path, metadata_path


def run_build_reference_data(
    *,
    file_path: str | Path | None = None,
    model_dir: str | Path = "app/model",
    out_dir: str | Path = "app/model/reference",
    max_rows: int = 1000,
    backup: bool = True,
    force: bool = False,
    include_raw_diagnostic: bool = False,
) -> dict[str, Any]:
    model_dir_path = Path(model_dir)
    model_path, metadata_path = _ensure_promoted_artifacts(model_dir_path)
    metadata = _safe_read_json(metadata_path)
    ok, errors = validate_metadata(metadata)
    if not ok:
        error_excerpt = "; ".join(errors[:10])
        raise ValueError(
            f"Promoted metadata invalid. Run src.promote_model again. {error_excerpt}"
        )

    expected_raw_cols = metadata.get("expected_raw_cols")
    expected_model_cols = metadata.get("expected_model_cols")
    if not isinstance(expected_raw_cols, list) or not expected_raw_cols:
        raise ValueError("metadata.expected_raw_cols must be a non-empty list.")
    if not isinstance(expected_model_cols, list) or not expected_model_cols:
        raise ValueError("metadata.expected_model_cols must be a non-empty list.")

    feature_engineering = metadata.get("feature_engineering", {})
    if not isinstance(feature_engineering, dict):
        raise ValueError("metadata.feature_engineering must be an object.")
    enable_feature_engineering = bool(feature_engineering.get("enabled", False))
    enable_age_bucket = bool(feature_engineering.get("enable_age_bucket", False))

    train_pair = metadata.get("train_pair")
    if not isinstance(train_pair, dict):
        raise ValueError("metadata.train_pair must be an object.")
    year_t = _to_int(train_pair.get("year_t"), 2022)
    year_t1 = _to_int(train_pair.get("year_t1"), 2023)

    resolved_dataset_path = _resolve_dataset_path(file_path)
    dfs, _, _ = load_pede_workbook_with_metadata(file_path=resolved_dataset_path)
    if year_t not in dfs or year_t1 not in dfs:
        raise ValueError(
            f"Train pair {year_t}->{year_t1} unavailable in dataset. Years loaded: {sorted(dfs.keys())}"
        )

    _, y_train, ids = make_temporal_pairs(
        dfs[year_t],
        dfs[year_t1],
        year_t,
        year_t1,
    )
    X_raw_train = build_raw_from_ids(
        df_year_t=dfs[year_t],
        ids=ids,
        expected_raw_cols=expected_raw_cols,
    )
    X_model_train = transform_raw_to_model_frame(
        X_raw=X_raw_train,
        year_t=year_t,
        expected_raw_cols=list(expected_raw_cols),
        expected_model_cols=list(expected_model_cols),
        enable_feature_engineering=enable_feature_engineering,
        enable_age_bucket=enable_age_bucket,
        feature_pruning_plan=None,
        strict_raw=True,
        include_report=False,
        context=f"reference_{year_t}_{year_t1}",
    )
    if isinstance(X_model_train, tuple):
        X_model_train = X_model_train[0]
    X_model_train = X_model_train.loc[:, list(expected_model_cols)].copy()
    _assert_no_forbidden_columns(list(X_model_train.columns))

    y_array = pd.Series(y_train).astype(int).to_numpy()
    sample_indices = _stratified_indices(
        y_array,
        max_rows=int(max_rows),
        random_state=int(RANDOM_STATE),
    )
    X_model_sample = X_model_train.iloc[sample_indices].reset_index(drop=True)
    X_raw_sample = X_raw_train.iloc[sample_indices].reset_index(drop=True)
    n_train = int(len(y_array))
    n_pos = int(np.sum(y_array == 1))
    prevalence = float(n_pos / n_train) if n_train else 0.0

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    reference_csv_path = out_dir_path / _DEFAULT_REFERENCE_FILE
    profile_json_path = out_dir_path / _DEFAULT_PROFILE_FILE
    meta_json_path = out_dir_path / _DEFAULT_META_FILE
    raw_diag_path = out_dir_path / _OPTIONAL_RAW_DIAGNOSTIC_FILE

    target_paths = [reference_csv_path, profile_json_path, meta_json_path]
    if include_raw_diagnostic:
        target_paths.append(raw_diag_path)
    existing_targets = [path for path in target_paths if path.exists()]
    if existing_targets and not bool(force):
        raise ValueError(
            "Reference destination exists. Use --force 1 to overwrite (backup enabled by default)."
        )

    backup_dir: Path | None = None
    if bool(backup):
        backup_dir = _backup_existing_reference(
            out_dir=out_dir_path,
            target_paths=target_paths,
        )

    X_model_sample.to_csv(reference_csv_path, index=False)
    if include_raw_diagnostic:
        _assert_no_forbidden_columns(list(X_raw_sample.columns))
        X_raw_sample.to_csv(raw_diag_path, index=False)

    profile_payload = _build_reference_profile(
        X_model_sample,
        n_train=n_train,
        n_pos=n_pos,
        prevalence=prevalence,
    )
    profile_json_path.write_text(
        json.dumps(profile_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata.get("model_version")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
        "winner": {
            "model_family": metadata.get("model_family"),
            "variant": metadata.get("variant"),
        },
        "train_pair": {
            "year_t": int(year_t),
            "year_t1": int(year_t1),
            "n": int(n_train),
            "n_pos": int(n_pos),
            "prevalence": float(prevalence),
        },
        "sampling": {
            "max_rows": int(max_rows),
            "used_rows": int(len(X_model_sample)),
            "strategy": "stratified_deterministic",
            "random_state": int(RANDOM_STATE),
        },
        "paths": {
            "reference_csv": str(reference_csv_path),
            "profile_json": str(profile_json_path),
            "meta_json": str(meta_json_path),
            "raw_diagnostic_csv": str(raw_diag_path) if include_raw_diagnostic else None,
        },
        "sha256": {
            "reference_csv": _sha256(reference_csv_path),
            "profile_json": _sha256(profile_json_path),
            "meta_json": None,
            "model_joblib": _sha256(model_path),
            "metadata_json": _sha256(metadata_path),
        },
        "backup": {
            "enabled": bool(backup),
            "path": str(backup_dir) if backup_dir is not None else None,
        },
        "notes": [
            "Reference dataset is MODEL frame post feature engineering and pruning.",
            "No IDs/PII stored.",
        ],
    }
    meta_json_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_payload["sha256"]["meta_json"] = _sha256(meta_json_path)
    meta_json_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for payload in (profile_payload, meta_payload):
        forbidden = sorted(_FORBIDDEN_KEYS & _collect_keys(payload))
        if forbidden:
            raise ValueError(f"Privacy check failed: forbidden keys found: {forbidden}")

    return {
        "status": "PASS",
        "reference_csv": str(reference_csv_path),
        "profile_json": str(profile_json_path),
        "meta_json": str(meta_json_path),
        "used_rows": int(len(X_model_sample)),
        "n_features": int(X_model_sample.shape[1]),
        "backup_path": None if backup_dir is None else str(backup_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reference MODEL-frame dataset for drift monitoring."
    )
    parser.add_argument(
        "--file-path",
        "--dataset-path",
        dest="file_path",
        type=str,
        default=None,
        help="Path to XLSX dataset. Defaults to DATASET_PATH env / project default.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="app/model",
        help="Directory with promoted model.joblib and metadata.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="app/model/reference",
        help="Output directory for reference artifacts.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Maximum rows to persist in reference_model_frame.csv.",
    )
    parser.add_argument(
        "--backup",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, backup existing reference files before overwrite.",
    )
    parser.add_argument(
        "--force",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, allow overwrite in destination directory.",
    )
    parser.add_argument(
        "--include-raw-diagnostic",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, persist optional raw diagnostic sample (not used as official drift baseline).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    try:
        report = run_build_reference_data(
            file_path=args.file_path,
            model_dir=args.model_dir,
            out_dir=args.out_dir,
            max_rows=int(args.max_rows),
            backup=_parse_bool_flag(args.backup),
            force=_parse_bool_flag(args.force),
            include_raw_diagnostic=_parse_bool_flag(args.include_raw_diagnostic),
        )
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("%s", exc)
        raise SystemExit(1) from exc

    _logger.info(
        "Reference data generated | rows=%d features=%d csv=%s",
        report["used_rows"],
        report["n_features"],
        report["reference_csv"],
    )


if __name__ == "__main__":
    main()
