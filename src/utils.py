"""Shared logging utilities for consistent project observability."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Final

_DEFAULT_LOG_LEVEL: Final[str] = "INFO"
_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_STDOUT_HANDLER_TAG: Final[str] = "project_stdout"
_FILE_HANDLER_TAG: Final[str] = "project_file"


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


def _should_enable_file_handler(log_to_file: bool) -> bool:
    if log_to_file:
        return True

    raw = os.getenv("LOG_TO_FILE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


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
    tagged_handler = next(
        (h for h in root_logger.handlers if getattr(h, "_project_handler_tag", None) == _FILE_HANDLER_TAG),
        None,
    )
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
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def setup_logging(level: str | int | None = None, log_to_file: bool = False) -> None:
    """Configure root logging idempotently for the whole project.

    Strategy: handlers are attached only to the root logger. Child loggers returned by
    `get_logger` use `propagate=True` and do not attach their own handlers.
    """
    resolved_level, level_warning = _resolve_log_level(level)
    formatter = logging.Formatter(_LOG_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    _ensure_stdout_handler(root_logger, formatter)
    if _should_enable_file_handler(log_to_file):
        _ensure_file_handler(root_logger, formatter, Path("logs/app.log"))

    for handler in root_logger.handlers:
        if getattr(handler, "_project_handler_tag", None) in {_STDOUT_HANDLER_TAG, _FILE_HANDLER_TAG}:
            handler.setLevel(resolved_level)

    if level_warning:
        root_logger.warning(level_warning)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger using project-wide defaults."""
    setup_logging()
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger
