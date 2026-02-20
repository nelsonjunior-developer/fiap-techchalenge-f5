"""Schema validation for serving metadata.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_KEYS: dict[str, Any] = {
    "top_level": [
        "model_family",
        "variant",
        "model_version",
        "trained_at",
        "promoted_at",
        "random_state",
        "train_pair",
        "holdout_pair",
        "dataset",
        "expected_raw_cols",
        "expected_model_cols",
        "excluded_cols",
        "feature_engineering",
        "feature_pruning",
        "threshold_policy",
        "evaluation_train_at_0.5",
        "evaluation_train_at_0.30",
        "evaluation_holdout_at_0.5",
        "evaluation_holdout_at_0.30",
        "evaluation_holdout_at_calibrated_threshold",
        "versions",
        "artifact_hashes",
    ],
    "versions": ["python", "pandas", "numpy", "scikit_learn", "joblib"],
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_str_or_none(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _validate_required_str(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    if key not in payload:
        errors.append(f"missing required key: {key}")
        return
    if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
        errors.append(f"invalid type/value for key '{key}' (expected non-empty str)")


def _validate_required_list_of_str(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    if key not in payload:
        errors.append(f"missing required key: {key}")
        return
    raw = payload.get(key)
    if not isinstance(raw, list):
        errors.append(f"invalid type for key '{key}' (expected list[str])")
        return
    if any(not isinstance(item, str) for item in raw):
        errors.append(f"invalid type for key '{key}' (expected list[str])")


def _validate_pair_block(block: dict[str, Any], key: str, errors: list[str], allow_null: bool = False) -> None:
    required = ["year_t", "year_t1", "n", "n_pos", "prevalence"]
    raw = block.get(key)
    if raw is None and allow_null:
        return
    if not isinstance(raw, dict):
        errors.append(f"invalid type for key '{key}' (expected object)")
        return
    for field in required:
        if field not in raw:
            errors.append(f"missing required key: {key}.{field}")
    if "year_t" in raw and not isinstance(raw["year_t"], int):
        errors.append(f"invalid type for key '{key}.year_t' (expected int)")
    if "year_t1" in raw and not isinstance(raw["year_t1"], int):
        errors.append(f"invalid type for key '{key}.year_t1' (expected int)")
    if "n" in raw and not isinstance(raw["n"], int):
        errors.append(f"invalid type for key '{key}.n' (expected int)")
    if "n_pos" in raw and not isinstance(raw["n_pos"], int):
        errors.append(f"invalid type for key '{key}.n_pos' (expected int)")
    if "prevalence" in raw and not _is_number(raw["prevalence"]):
        errors.append(f"invalid type for key '{key}.prevalence' (expected float)")


def _validate_dataset_block(payload: dict[str, Any], errors: list[str]) -> None:
    if "dataset" not in payload:
        errors.append("missing required key: dataset")
        return
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("invalid type for key 'dataset' (expected object)")
        return
    for key in ("path_hint", "basename", "sha256"):
        if key not in dataset:
            errors.append(f"missing required key: dataset.{key}")
            continue
        if not _is_str_or_none(dataset[key]):
            errors.append(f"invalid type for key 'dataset.{key}' (expected str|null)")
    if "bytes" in dataset and dataset["bytes"] is not None and not isinstance(dataset["bytes"], int):
        errors.append("invalid type for key 'dataset.bytes' (expected int|null)")
    if "mtime_utc" in dataset and not _is_str_or_none(dataset["mtime_utc"]):
        errors.append("invalid type for key 'dataset.mtime_utc' (expected str|null)")


def _validate_feature_engineering(payload: dict[str, Any], errors: list[str]) -> None:
    key = "feature_engineering"
    raw = payload.get(key)
    if not isinstance(raw, dict):
        errors.append(f"invalid type for key '{key}' (expected object)")
        return
    for field in ("enabled", "enable_age_bucket", "engineered_cols"):
        if field not in raw:
            errors.append(f"missing required key: {key}.{field}")
    if "enabled" in raw and not isinstance(raw["enabled"], bool):
        errors.append(f"invalid type for key '{key}.enabled' (expected bool)")
    if "enable_age_bucket" in raw and not isinstance(raw["enable_age_bucket"], bool):
        errors.append(f"invalid type for key '{key}.enable_age_bucket' (expected bool)")
    if "engineered_cols" in raw:
        if not isinstance(raw["engineered_cols"], list) or any(
            not isinstance(item, str) for item in raw["engineered_cols"]
        ):
            errors.append(f"invalid type for key '{key}.engineered_cols' (expected list[str])")


def _validate_feature_pruning(payload: dict[str, Any], errors: list[str]) -> None:
    key = "feature_pruning"
    raw = payload.get(key)
    if not isinstance(raw, dict):
        errors.append(f"invalid type for key '{key}' (expected object)")
        return
    for field in ("plan_hash", "kept_model_cols_count", "dropped_summary"):
        if field not in raw:
            errors.append(f"missing required key: {key}.{field}")
    if "plan_hash" in raw and not _is_str_or_none(raw["plan_hash"]):
        errors.append(f"invalid type for key '{key}.plan_hash' (expected str|null)")
    if "kept_model_cols_count" in raw and not isinstance(raw["kept_model_cols_count"], int):
        errors.append(f"invalid type for key '{key}.kept_model_cols_count' (expected int)")
    if "dropped_summary" in raw and raw["dropped_summary"] is not None and not isinstance(
        raw["dropped_summary"], dict
    ):
        errors.append(f"invalid type for key '{key}.dropped_summary' (expected dict|null)")


def _validate_threshold_policy(payload: dict[str, Any], errors: list[str]) -> None:
    key = "threshold_policy"
    raw = payload.get(key)
    if not isinstance(raw, dict):
        errors.append(f"invalid type for key '{key}' (expected object)")
        return
    for field in (
        "operational_fixed_threshold",
        "recall_target_for_calibration",
        "calibrated_threshold",
        "topk_fallback_fraction",
        "notes",
    ):
        if field not in raw:
            errors.append(f"missing required key: {key}.{field}")
    if "operational_fixed_threshold" in raw and not _is_number(raw["operational_fixed_threshold"]):
        errors.append(f"invalid type for key '{key}.operational_fixed_threshold' (expected float)")
    if "recall_target_for_calibration" in raw and not _is_number(raw["recall_target_for_calibration"]):
        errors.append(f"invalid type for key '{key}.recall_target_for_calibration' (expected float)")
    if "calibrated_threshold" in raw and not _is_number(raw["calibrated_threshold"]) and raw[
        "calibrated_threshold"
    ] is not None:
        errors.append(f"invalid type for key '{key}.calibrated_threshold' (expected float|null)")
    if "topk_fallback_fraction" in raw and not _is_number(raw["topk_fallback_fraction"]):
        errors.append(f"invalid type for key '{key}.topk_fallback_fraction' (expected float)")
    if "notes" in raw and (
        not isinstance(raw["notes"], list) or any(not isinstance(item, str) for item in raw["notes"])
    ):
        errors.append(f"invalid type for key '{key}.notes' (expected list[str])")


def _validate_eval_block(payload: dict[str, Any], key: str, errors: list[str], allow_null: bool) -> None:
    raw = payload.get(key)
    if raw is None and allow_null:
        return
    if not isinstance(raw, dict):
        errors.append(f"invalid type for key '{key}' (expected object)")
        return
    if "threshold" not in raw:
        errors.append(f"missing required key: {key}.threshold")
    if "metrics" not in raw:
        errors.append(f"missing required key: {key}.metrics")
    if "confusion_matrix" not in raw:
        errors.append(f"missing required key: {key}.confusion_matrix")

    if "threshold" in raw and not _is_number(raw["threshold"]):
        errors.append(f"invalid type for key '{key}.threshold' (expected float)")

    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        for metric_key in ("recall", "precision", "f1", "roc_auc", "pr_auc", "positive_rate"):
            if metric_key not in metrics:
                errors.append(f"missing required key: {key}.metrics.{metric_key}")
            elif metrics[metric_key] is not None and not _is_number(metrics[metric_key]):
                errors.append(
                    f"invalid type for key '{key}.metrics.{metric_key}' (expected float|null)"
                )
    elif "metrics" in raw:
        errors.append(f"invalid type for key '{key}.metrics' (expected object)")

    cm = raw.get("confusion_matrix")
    if isinstance(cm, dict):
        for cm_key in ("tn", "fp", "fn", "tp"):
            if cm_key not in cm:
                errors.append(f"missing required key: {key}.confusion_matrix.{cm_key}")
            elif not isinstance(cm[cm_key], int):
                errors.append(f"invalid type for key '{key}.confusion_matrix.{cm_key}' (expected int)")
    elif "confusion_matrix" in raw:
        errors.append(f"invalid type for key '{key}.confusion_matrix' (expected object)")


def _validate_versions(payload: dict[str, Any], errors: list[str]) -> None:
    key = "versions"
    raw = payload.get(key)
    if not isinstance(raw, dict):
        errors.append(f"invalid type for key '{key}' (expected object)")
        return
    for field in ("python", "pandas", "numpy", "scikit_learn", "joblib"):
        if field not in raw:
            errors.append(f"missing required key: {key}.{field}")
            continue
        value = raw[field]
        if field in {"scikit_learn", "joblib"}:
            if not _is_str_or_none(value):
                errors.append(f"invalid type for key '{key}.{field}' (expected str|null)")
        elif not isinstance(value, str):
            errors.append(f"invalid type for key '{key}.{field}' (expected str)")


def _validate_artifact_hashes(payload: dict[str, Any], errors: list[str]) -> None:
    key = "artifact_hashes"
    raw = payload.get(key)
    if not isinstance(raw, dict):
        errors.append(f"invalid type for key '{key}' (expected object)")
        return
    if "model_joblib_sha256" not in raw:
        errors.append("missing required key: artifact_hashes.model_joblib_sha256")
    elif not isinstance(raw["model_joblib_sha256"], str):
        errors.append("invalid type for key 'artifact_hashes.model_joblib_sha256' (expected str)")
    if "metadata_sha256" in raw and raw["metadata_sha256"] is not None and not isinstance(
        raw["metadata_sha256"], str
    ):
        errors.append("invalid type for key 'artifact_hashes.metadata_sha256' (expected str|null)")


def validate_metadata(metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate serving metadata structure and required keys."""
    errors: list[str] = []
    if not isinstance(metadata, dict):
        return False, ["metadata must be a JSON object"]

    _validate_required_str(metadata, "model_family", errors)
    if isinstance(metadata.get("model_family"), str):
        model_family = str(metadata.get("model_family")).strip()
        if model_family not in {"baseline_logreg", "nonlinear_hgb"}:
            errors.append(
                "invalid value for key 'model_family' (expected baseline_logreg|nonlinear_hgb)"
            )
    _validate_required_str(metadata, "variant", errors)
    _validate_required_str(metadata, "model_version", errors)
    _validate_required_str(metadata, "trained_at", errors)

    if "promoted_at" not in metadata:
        errors.append("missing required key: promoted_at")
    elif not _is_str_or_none(metadata["promoted_at"]):
        errors.append("invalid type for key 'promoted_at' (expected str|null)")

    if "random_state" not in metadata:
        errors.append("missing required key: random_state")
    elif not isinstance(metadata["random_state"], int):
        errors.append("invalid type for key 'random_state' (expected int)")

    _validate_pair_block(metadata, "train_pair", errors, allow_null=False)
    _validate_pair_block(metadata, "holdout_pair", errors, allow_null=True)
    _validate_dataset_block(metadata, errors)
    _validate_required_list_of_str(metadata, "expected_raw_cols", errors)
    _validate_required_list_of_str(metadata, "expected_model_cols", errors)
    _validate_required_list_of_str(metadata, "excluded_cols", errors)
    _validate_feature_engineering(metadata, errors)
    _validate_feature_pruning(metadata, errors)
    _validate_threshold_policy(metadata, errors)
    _validate_eval_block(metadata, "evaluation_train_at_0.5", errors, allow_null=False)
    _validate_eval_block(metadata, "evaluation_train_at_0.30", errors, allow_null=False)
    _validate_eval_block(metadata, "evaluation_holdout_at_0.5", errors, allow_null=True)
    _validate_eval_block(metadata, "evaluation_holdout_at_0.30", errors, allow_null=True)
    _validate_eval_block(
        metadata,
        "evaluation_holdout_at_calibrated_threshold",
        errors,
        allow_null=True,
    )
    _validate_versions(metadata, errors)
    _validate_artifact_hashes(metadata, errors)

    return len(errors) == 0, errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate serving metadata schema.")
    parser.add_argument(
        "--path",
        type=str,
        default="app/model/metadata.json",
        help="Path to metadata.json to validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata_path = Path(args.path)
    if not metadata_path.exists():
        print(f"metadata path not found: {metadata_path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"invalid metadata json: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    ok, errors = validate_metadata(payload if isinstance(payload, dict) else {})
    if not ok:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print("metadata schema valid")


if __name__ == "__main__":
    main()
