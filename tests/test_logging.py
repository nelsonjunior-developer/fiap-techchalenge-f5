import logging
import pytest

from src.utils import get_logger, setup_logging


_PROJECT_HANDLER_TAGS = {"project_stdout", "project_file"}


def _count_project_handlers(tag: str) -> int:
    root_logger = logging.getLogger()
    return sum(
        1 for h in root_logger.handlers if getattr(h, "_project_handler_tag", None) == tag
    )


def _remove_project_handlers() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if getattr(handler, "_project_handler_tag", None) in _PROJECT_HANDLER_TAGS:
            root_logger.removeHandler(handler)
            handler.close()


@pytest.fixture(autouse=True)
def reset_logging_state(monkeypatch: pytest.MonkeyPatch):
    _remove_project_handlers()
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_TO_FILE", raising=False)
    yield
    _remove_project_handlers()


def test_setup_logging_runs_without_errors() -> None:
    setup_logging(level="INFO", log_to_file=False)
    assert _count_project_handlers("project_stdout") == 1


def test_get_logger_returns_logger_with_project_handler() -> None:
    logger = get_logger("tests.logging")
    assert isinstance(logger, logging.Logger)
    assert logger.propagate is True
    assert len(logger.handlers) == 0
    assert _count_project_handlers("project_stdout") == 1


def test_setup_logging_is_idempotent_for_stdout_handler() -> None:
    setup_logging(level="INFO", log_to_file=False)
    before = _count_project_handlers("project_stdout")
    setup_logging(level="DEBUG", log_to_file=False)
    after = _count_project_handlers("project_stdout")
    assert before == after
    logger = get_logger("tests.logging.idempotent")
    assert len(logger.handlers) == 0


def test_log_message_without_ra_does_not_emit_ra_token(caplog) -> None:
    logger = get_logger("tests.logging")
    with caplog.at_level(logging.INFO):
        logger.info("Pipeline start | total=%d", 10)
    assert "RA=" not in caplog.text


def test_invalid_log_level_falls_back_to_info(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setenv("LOG_LEVEL", "not_a_valid_level")
    with caplog.at_level(logging.WARNING):
        setup_logging(level=None, log_to_file=False)

    assert logging.getLogger().level == logging.INFO
    assert "Invalid LOG_LEVEL='not_a_valid_level'" in caplog.text


def test_child_logger_message_is_not_duplicated(caplog) -> None:
    setup_logging(level="INFO", log_to_file=False)
    child_logger = get_logger("tests.logging.child")

    with caplog.at_level(logging.INFO):
        child_logger.info("single-message-check")

    matches = [record for record in caplog.records if record.getMessage() == "single-message-check"]
    assert len(matches) == 1


def test_file_handler_failure_keeps_stdout_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_file_handler(*args, **kwargs):
        raise OSError("simulated file system failure")

    monkeypatch.setattr(logging, "FileHandler", failing_file_handler)
    setup_logging(level="INFO", log_to_file=True)

    assert _count_project_handlers("project_stdout") == 1
    assert _count_project_handlers("project_file") == 0
