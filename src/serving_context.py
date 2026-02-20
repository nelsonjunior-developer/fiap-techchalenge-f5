"""Helpers to load serving metadata and resolve model identity/threshold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.utils import get_logger

_logger = get_logger(__name__)

DEFAULT_OPERATIONAL_THRESHOLD = 0.30
DEFAULT_MODEL_VERSION = "unknown"
DEFAULT_MODEL_FAMILY = "unknown"
DEFAULT_VARIANT = "unknown"
DEFAULT_METADATA_PATH = Path("app/model/metadata.json")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_serving_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    """Load serving metadata object; returns empty dict on missing/invalid payload."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        _logger.info(
            "serving metadata unavailable | basename=%s",
            metadata_path.name,
        )
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _logger.warning(
            "serving metadata invalid json | basename=%s",
            metadata_path.name,
        )
        return {}
    if not isinstance(payload, dict):
        _logger.warning(
            "serving metadata invalid payload type | basename=%s",
            metadata_path.name,
        )
        return {}
    return payload


def extract_operational_threshold(
    metadata: Mapping[str, Any] | None,
    *,
    default_threshold: float = DEFAULT_OPERATIONAL_THRESHOLD,
) -> tuple[float, list[str]]:
    """Extract operational threshold from metadata with legacy-compatible fallback."""
    notes: list[str] = []
    meta = dict(metadata) if isinstance(metadata, Mapping) else {}
    threshold_policy = meta.get("threshold_policy")
    threshold_value: float | None = None
    if isinstance(threshold_policy, Mapping):
        threshold_value = _safe_float(threshold_policy.get("operational_fixed_threshold"))
        if threshold_value is None:
            operational = threshold_policy.get("operational")
            if isinstance(operational, Mapping):
                threshold_value = _safe_float(operational.get("threshold"))
                if threshold_value is not None:
                    notes.append("threshold_from_legacy_policy_operational")
    if threshold_value is None:
        threshold_value = float(default_threshold)
        notes.append("fallback_default_threshold")
    elif threshold_value < 0.0 or threshold_value > 1.0:
        threshold_value = float(default_threshold)
        notes.append("fallback_default_threshold_invalid_metadata")
    return float(threshold_value), notes


def extract_model_identity(metadata: Mapping[str, Any] | None) -> tuple[dict[str, str], list[str]]:
    """Extract model identity from metadata with unknown defaults."""
    notes: list[str] = []
    meta = dict(metadata) if isinstance(metadata, Mapping) else {}

    model_version = str(meta.get("model_version") or "").strip()
    if not model_version:
        model_version = DEFAULT_MODEL_VERSION
        notes.append("fallback_unknown_model_version")

    model_family = str(meta.get("model_family") or "").strip()
    if not model_family:
        model_family = DEFAULT_MODEL_FAMILY
        notes.append("fallback_unknown_model_family")

    variant = str(meta.get("variant") or "").strip()
    if not variant:
        variant = DEFAULT_VARIANT
        notes.append("fallback_unknown_variant")

    return {
        "model_version": model_version,
        "model_family": model_family,
        "variant": variant,
    }, notes


__all__ = [
    "DEFAULT_METADATA_PATH",
    "DEFAULT_MODEL_FAMILY",
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_OPERATIONAL_THRESHOLD",
    "DEFAULT_VARIANT",
    "extract_model_identity",
    "extract_operational_threshold",
    "load_serving_metadata",
]

