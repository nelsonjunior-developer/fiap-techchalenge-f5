"""Operational privacy guardrails for logs, monitoring payloads, and artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.contracts import PII_COLUMNS

SENSITIVE_FIELD_NAMES: set[str] = {str(name).strip() for name in set(PII_COLUMNS) | {"RA"} if str(name).strip()}
SENSITIVE_KEY_PATTERNS: tuple[str, ...] = (
    "ra",
    "student",
    "students",
    "ids",
    "ra_list",
    "identifiers",
    "records",
    "payload",
    "probas",
    "risk_probas",
    "scores",
)
MAX_LIST_LEN_LOGGABLE = 20

_SENSITIVE_FIELD_NAMES_NORM = {str(name).strip().lower() for name in SENSITIVE_FIELD_NAMES}
_SUSPICIOUS_EXACT_KEYS = {
    "ra",
    "ra_list",
    "ids",
    "student_ids",
    "students",
    "student",
    "identifiers",
    "records",
    "record",
    "payload",
    "payload_raw",
    "probas",
    "risk_probas",
    "scores",
    "score_list",
}
_LIST_LEN_EXEMPT_KEYS = {"bin_edges", "bin_counts"}  # histogram aggregates
FORBIDDEN_ARTIFACT_KEYS: frozenset[str] = frozenset(
    {"ra", "ra_list", "ids", "student_ids", "students", "records"}
)


def _normalize_key(key: Any) -> str:
    return str(key).strip()


def _json_safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_scalar(item())
        except Exception:
            pass
    return None


def _to_json_compatible(value: Any) -> Any:
    scalar = _json_safe_scalar(value)
    if scalar is not None or value is None:
        return scalar
    if isinstance(value, Mapping):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, BaseException):
        return {"type": value.__class__.__name__, "message": str(value)}
    return str(value)


def _tokenize_key(name: str) -> set[str]:
    normalized = name.lower()
    tokens = []
    current = []
    for ch in normalized:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return set(tokens)


def _is_sensitive_field_name(name: str) -> bool:
    norm = name.strip().lower()
    if not norm:
        return False
    if norm in _SENSITIVE_FIELD_NAMES_NORM:
        return True
    if norm.startswith("avaliador"):
        return True
    return False


def _is_suspicious_key(name: str) -> bool:
    norm = name.strip().lower()
    if not norm:
        return False
    if norm in _SUSPICIOUS_EXACT_KEYS:
        return True
    # Avoid broad substring false positives like "generated_at"; use tokens.
    tokens = _tokenize_key(norm)
    if "student" in tokens or "students" in tokens or "identifiers" in tokens:
        return True
    if "ids" in tokens and "invalid" not in tokens:
        return True
    if tokens == {"ra"}:
        return True
    return False


def collect_json_keys(payload: Any) -> set[str]:
    """Recursively collect lowercase JSON object keys from dict/list structures."""
    keys: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            keys.add(_normalize_key(key).lower())
            keys |= collect_json_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= collect_json_keys(item)
    return keys


def find_forbidden_json_keys(
    payload: Any,
    *,
    forbidden_keys: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Return sorted forbidden keys found in an artifact-style JSON payload."""
    forbidden = FORBIDDEN_ARTIFACT_KEYS if forbidden_keys is None else frozenset(forbidden_keys)
    return sorted(forbidden & collect_json_keys(payload))


def redact_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive keys and risky list payloads from a dict (for logging)."""
    redacted_keys: list[str] = []

    def _redact(value: Any, *, parent_key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for raw_key, raw_value in value.items():
                key = _normalize_key(raw_key)
                if _is_sensitive_field_name(key):
                    if key and key not in redacted_keys:
                        redacted_keys.append(key)
                    continue
                if _is_suspicious_key(key):
                    if key and key not in redacted_keys:
                        redacted_keys.append(key)
                    if isinstance(raw_value, (list, tuple, set)):
                        seq = list(raw_value)
                        if any(isinstance(item, Mapping) for item in seq):
                            out[key] = "[REDACTED_LIST]"
                        elif len(seq) > MAX_LIST_LEN_LOGGABLE:
                            out[key] = f"[REDACTED_LIST_LEN:{len(seq)}]"
                        else:
                            out[key] = "[REDACTED_LIST]"
                    elif isinstance(raw_value, Mapping):
                        out[key] = "[REDACTED_OBJECT]"
                    else:
                        out[key] = "[REDACTED_VALUE]"
                    continue
                out[key] = _redact(raw_value, parent_key=key)
            return out

        if isinstance(value, (list, tuple, set)):
            seq = list(value)
            lowered_parent = (parent_key or "").strip().lower()
            if any(isinstance(item, Mapping) for item in seq):
                if parent_key and parent_key not in redacted_keys:
                    redacted_keys.append(parent_key)
                return "[REDACTED_LIST]"
            if lowered_parent and lowered_parent not in _LIST_LEN_EXEMPT_KEYS and len(seq) > MAX_LIST_LEN_LOGGABLE:
                if parent_key and parent_key not in redacted_keys:
                    redacted_keys.append(parent_key)
                return f"[REDACTED_LIST_LEN:{len(seq)}]"
            return [_redact(item, parent_key=parent_key) for item in seq]

        return _to_json_compatible(value)

    sanitized = _redact(dict(obj))
    if not isinstance(sanitized, dict):
        sanitized = {"value": _to_json_compatible(sanitized)}
    if redacted_keys:
        sanitized["redacted_keys"] = sorted(dict.fromkeys(redacted_keys))
    return sanitized


def safe_log_extra(extra: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a redacted + JSON-serializable dict for logging `context/extra`."""
    if extra is None:
        return None
    try:
        sanitized = redact_dict(dict(extra))
    except Exception:
        sanitized = {"value": str(extra)}
    try:
        json.dumps(sanitized, ensure_ascii=False)
    except Exception:
        return {"value": str(sanitized)}
    return sanitized


def is_safe_json_payload(obj: Any) -> bool:
    """Heuristic privacy validator for monitoring/artifact JSON payloads.

    This checks for:
    - explicit sensitive field names (`RA`, `Nome_Anon`, `Avaliador*`, etc.)
    - suspicious payload-like keys carrying nested objects or lists of dicts
    - suspicious ID-like keys carrying large lists

    Aggregate metrics like `ra_null`, `ra_duplicates`, histograms and small scalar lists
    are allowed.
    """

    def _walk(value: Any, *, parent_key: str | None = None) -> bool:
        if isinstance(value, Mapping):
            for raw_key, raw_value in value.items():
                key = _normalize_key(raw_key)
                if _is_sensitive_field_name(key):
                    return False

                suspicious = _is_suspicious_key(key)
                if suspicious:
                    if isinstance(raw_value, Mapping):
                        return False
                    if isinstance(raw_value, (list, tuple, set)):
                        seq = list(raw_value)
                        if len(seq) == 0:
                            continue
                        if any(isinstance(item, Mapping) for item in seq):
                            return False
                        if len(seq) > MAX_LIST_LEN_LOGGABLE and key.lower() not in _LIST_LEN_EXEMPT_KEYS:
                            return False

                if not _walk(raw_value, parent_key=key):
                    return False
            return True

        if isinstance(value, (list, tuple, set)):
            seq = list(value)
            lowered_parent = (parent_key or "").strip().lower()
            if lowered_parent in _SUSPICIOUS_EXACT_KEYS:
                if any(isinstance(item, Mapping) for item in seq):
                    return False
                if len(seq) > MAX_LIST_LEN_LOGGABLE and lowered_parent not in _LIST_LEN_EXEMPT_KEYS:
                    return False
            for item in seq:
                if not _walk(item, parent_key=parent_key):
                    return False
            return True

        return True

    return _walk(obj)


__all__ = [
    "FORBIDDEN_ARTIFACT_KEYS",
    "MAX_LIST_LEN_LOGGABLE",
    "SENSITIVE_FIELD_NAMES",
    "SENSITIVE_KEY_PATTERNS",
    "collect_json_keys",
    "find_forbidden_json_keys",
    "is_safe_json_payload",
    "redact_dict",
    "safe_log_extra",
]
