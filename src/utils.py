"""Shared logging utilities for consistent project observability.

Highlights:
- JSON logs by default (`LOG_FORMAT=json`) for Docker/stdout compatibility
- Optional plain formatter (`LOG_FORMAT=plain`)
- Idempotent root-handler setup (tagged handlers only)
- Structured event helper (`log_event`) with basic anti-PII context redaction
- Request-scoped `request_id` support via `contextvars` (API middleware)
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import date, datetime, time, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Final, Mapping

from src.privacy import safe_log_extra

_DEFAULT_LOG_LEVEL: Final[str] = "INFO"
_DEFAULT_LOG_FORMAT_MODE: Final[str] = "json"
_DEFAULT_FILE_LOG_PATH: Final[str] = "logs/app.log"
_PLAIN_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_STDOUT_HANDLER_TAG: Final[str] = "project_stdout"
_FILE_HANDLER_TAG: Final[str] = "project_file"
_PROJECT_HANDLER_TAGS: Final[set[str]] = {_STDOUT_HANDLER_TAG, _FILE_HANDLER_TAG}
_REQUEST_ID_VAR: ContextVar[str | None] = ContextVar("project_request_id", default=None)


def _resolve_log_level(level: str | int | None) -> tuple[int, str | None]:
    raw_level: str | int
    if level is None:
        raw_level = os.getenv("LOG_LEVEL", _DEFAULT_LOG_LEVEL)
    else:
        raw_level = level

    if isinstance(raw_level, int):
        return raw_level, None

    normalized = str(raw_level).upper().strip()
    resolved = getattr(logging, normalized, None)
    if isinstance(resolved, int):
        return resolved, None

    return (
        logging.INFO,
        f"Invalid LOG_LEVEL='{raw_level}'. Falling back to INFO.",
    )


def _resolve_log_format(log_format: str | None) -> tuple[str, str | None]:
    raw = os.getenv("LOG_FORMAT", _DEFAULT_LOG_FORMAT_MODE) if log_format is None else str(log_format)
    normalized = str(raw).strip().lower()
    if normalized in {"json", "plain"}:
        return normalized, None
    return ("json", f"Invalid LOG_FORMAT='{raw}'. Falling back to json.")


def _resolve_log_file_path(log_file_path: str | Path | None) -> Path:
    if log_file_path is not None:
        return Path(log_file_path)
    raw = os.getenv("LOG_FILE_PATH", _DEFAULT_FILE_LOG_PATH).strip()
    return Path(raw or _DEFAULT_FILE_LOG_PATH)


def _should_enable_file_handler(log_to_file: bool) -> bool:
    if log_to_file:
        return True
    raw = os.getenv("LOG_TO_FILE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _json_safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return value.isoformat()
    return None


def _sanitize_for_json(value: Any) -> Any:
    scalar = _json_safe_scalar(value)
    if scalar is not None or value is None:
        return scalar

    if isinstance(value, Mapping):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, BaseException):
        return {"type": value.__class__.__name__, "message": str(value)}

    # numpy/pandas scalars often expose `.item()`
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _sanitize_for_json(item())
        except Exception:
            pass

    return str(value)


def sanitize_log_context(context: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    if context is None:
        return None, []
    sanitized = safe_log_extra(dict(context))
    if sanitized is None:
        return None, []
    if not isinstance(sanitized, dict):
        sanitized = {"value": _sanitize_for_json(sanitized)}
    redacted = sanitized.get("redacted_keys", [])
    if not isinstance(redacted, list):
        redacted = []
    return sanitized, [str(item) for item in redacted]


def set_request_id_context(request_id: str | None) -> object:
    return _REQUEST_ID_VAR.set(None if request_id is None else str(request_id))


def reset_request_id_context(token: object) -> None:
    _REQUEST_ID_VAR.reset(token)  # type: ignore[arg-type]


def get_request_id_context() -> str | None:
    value = _REQUEST_ID_VAR.get()
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for stdout/file handlers."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": str(record.levelname),
            "logger": str(record.name),
            "msg": record.getMessage(),
        }

        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = _sanitize_for_json(event)

        context = getattr(record, "context", None)
        if context is not None:
            if isinstance(context, Mapping):
                sanitized_context, _ = sanitize_log_context(dict(context))
                payload["context"] = sanitized_context
            else:
                payload["context"] = _sanitize_for_json(context)

        request_id = getattr(record, "request_id", None)
        if request_id is None:
            request_id = get_request_id_context()
        if request_id is not None:
            payload["request_id"] = str(request_id)

        model_version = getattr(record, "model_version", None)
        if model_version is not None:
            payload["model_version"] = str(model_version)

        if record.exc_info:
            exc_type = record.exc_info[0]
            exc_value = record.exc_info[1]
            payload["exc_info"] = {
                "type": None if exc_type is None else getattr(exc_type, "__name__", str(exc_type)),
                "message": None if exc_value is None else str(exc_value),
            }
            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                payload["trace"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _build_formatter(log_format_mode: str) -> logging.Formatter:
    if str(log_format_mode).lower() == "plain":
        return logging.Formatter(_PLAIN_LOG_FORMAT)
    return JsonFormatter()


def _remove_tagged_handler(root_logger: logging.Logger, tag: str) -> None:
    for handler in list(root_logger.handlers):
        if getattr(handler, "_project_handler_tag", None) != tag:
            continue
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def _ensure_stdout_handler(root_logger: logging.Logger, formatter: logging.Formatter) -> None:
    tagged_handler = next(
        (h for h in root_logger.handlers if getattr(h, "_project_handler_tag", None) == _STDOUT_HANDLER_TAG),
        None,
    )

    if tagged_handler is None:
        tagged_handler = logging.StreamHandler(sys.stdout)
        tagged_handler._project_handler_tag = _STDOUT_HANDLER_TAG  # type: ignore[attr-defined]
        root_logger.addHandler(tagged_handler)

    tagged_handler.setFormatter(formatter)


def _ensure_file_handler(
    root_logger: logging.Logger,
    formatter: logging.Formatter,
    log_file: Path,
) -> None:
    desired_path = str(log_file)
    tagged_handler = next(
        (h for h in root_logger.handlers if getattr(h, "_project_handler_tag", None) == _FILE_HANDLER_TAG),
        None,
    )
    if tagged_handler is not None:
        current_path = getattr(tagged_handler, "_project_log_file_path", None)
        if current_path != desired_path:
            _remove_tagged_handler(root_logger, _FILE_HANDLER_TAG)
            tagged_handler = None

    if tagged_handler is not None:
        tagged_handler.setFormatter(formatter)
        return

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
    except OSError:
        # If local filesystem cannot persist logs, keep stdout logging only.
        return

    file_handler._project_handler_tag = _FILE_HANDLER_TAG  # type: ignore[attr-defined]
    file_handler._project_log_file_path = desired_path  # type: ignore[attr-defined]
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def setup_logging(
    level: str | int | None = None,
    log_to_file: bool = False,
    *,
    log_format: str | None = None,
    log_file_path: str | Path | None = None,
) -> logging.Logger:
    """Configure root logging idempotently for the whole project.

    Strategy:
    - handlers are attached only to the root logger (tagged handlers only)
    - child loggers returned by `get_logger` keep `propagate=True`
    - untagged handlers (e.g. pytest caplog) are never removed
    """
    resolved_level, level_warning = _resolve_log_level(level)
    resolved_format_mode, format_warning = _resolve_log_format(log_format)
    formatter = _build_formatter(resolved_format_mode)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    _ensure_stdout_handler(root_logger, formatter)
    if _should_enable_file_handler(log_to_file):
        _ensure_file_handler(root_logger, formatter, _resolve_log_file_path(log_file_path))
    else:
        _remove_tagged_handler(root_logger, _FILE_HANDLER_TAG)

    for handler in root_logger.handlers:
        if getattr(handler, "_project_handler_tag", None) in _PROJECT_HANDLER_TAGS:
            handler.setLevel(resolved_level)

    if level_warning:
        root_logger.warning(level_warning)
    if format_warning:
        root_logger.warning(format_warning)
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger using project-wide defaults."""
    setup_logging()
    logger = logging.getLogger(name)
    # Root handlers are the source of truth in this project.
    logger.propagate = True
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    message: str | None = None,
    request_id: str | None = None,
    model_version: str | None = None,
    context: Mapping[str, Any] | None = None,
    **context_fields: Any,
) -> None:
    """Emit a structured log event with serialized + redacted context.

    `context_fields` are merged on top of `context`.
    """
    merged_context: dict[str, Any] = {}
    if isinstance(context, Mapping):
        merged_context.update(dict(context))
    merged_context.update(context_fields)
    sanitized_context, _ = sanitize_log_context(merged_context or None)

    resolved_request_id = request_id or get_request_id_context()
    resolved_model_version = model_version
    if resolved_model_version is None and isinstance(sanitized_context, dict):
        model_version_candidate = sanitized_context.get("model_version")
        if model_version_candidate is not None:
            resolved_model_version = str(model_version_candidate)

    extra: dict[str, Any] = {
        "event": str(event),
        "context": sanitized_context,
    }
    if resolved_request_id is not None:
        extra["request_id"] = str(resolved_request_id)
    if resolved_model_version is not None:
        extra["model_version"] = str(resolved_model_version)

    logger.log(int(level), str(message or event), extra=extra)


__all__ = [
    "JsonFormatter",
    "get_logger",
    "get_request_id_context",
    "log_event",
    "reset_request_id_context",
    "sanitize_log_context",
    "set_request_id_context",
    "setup_logging",
]
