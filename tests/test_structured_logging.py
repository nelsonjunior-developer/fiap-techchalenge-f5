from __future__ import annotations

import json
import logging

import pytest

from src.utils import get_logger, log_event, setup_logging

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - environment dependent
    TestClient = None  # type: ignore[assignment]

from app.main import app


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
def _reset_logging_env(monkeypatch: pytest.MonkeyPatch):
    _remove_project_handlers()
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.delenv("LOG_TO_FILE", raising=False)
    monkeypatch.delenv("LOG_FILE_PATH", raising=False)
    yield
    _remove_project_handlers()


def test_setup_logging_json_outputs_valid_json_with_event_and_context(capsys) -> None:
    setup_logging(level="INFO", log_to_file=False, log_format="json")
    logger = get_logger("tests.structured.json")

    log_event(logger, "unit_test_event", context={"count": 2, "nested": {"ok": True}})

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines, "expected at least one stdout log line"
    payload = json.loads(lines[-1])
    assert payload["level"] == "INFO"
    assert payload["logger"] == "tests.structured.json"
    assert payload["msg"] == "unit_test_event"
    assert payload["event"] == "unit_test_event"
    assert payload["context"]["count"] == 2
    assert payload["context"]["nested"]["ok"] is True
    assert "ts" in payload


def test_setup_logging_is_idempotent_for_tagged_handlers_in_json_mode() -> None:
    setup_logging(level="INFO", log_to_file=False, log_format="json")
    before_stdout = _count_project_handlers("project_stdout")
    before_file = _count_project_handlers("project_file")

    setup_logging(level="DEBUG", log_to_file=False, log_format="json")
    after_stdout = _count_project_handlers("project_stdout")
    after_file = _count_project_handlers("project_file")

    assert before_stdout == after_stdout == 1
    assert before_file == after_file == 0


def test_log_event_redacts_sensitive_keys_from_context(capsys) -> None:
    setup_logging(level="INFO", log_to_file=False, log_format="json")
    logger = get_logger("tests.structured.redact")

    log_event(
        logger,
        "privacy_check",
        RA="123456",
        Nome_Anon="ALUNO X",
        Avaliador1="Pessoa Y",
        safe_count=3,
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    context = payload.get("context") or {}
    assert "RA" not in context
    assert "Nome_Anon" not in context
    assert "Avaliador1" not in context
    assert context["safe_count"] == 3
    assert sorted(context.get("redacted_keys", [])) == ["Avaliador1", "Nome_Anon", "RA"]


def test_request_id_middleware_adds_header_and_log_record_contains_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    if TestClient is None:
        pytest.skip("TestClient unavailable in this environment")

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    request_id = response.headers.get("X-Request-ID")
    assert request_id

    matching = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "health_check"
    ]
    assert matching, "expected health_check structured log record"
    assert getattr(matching[-1], "request_id", None) == request_id

