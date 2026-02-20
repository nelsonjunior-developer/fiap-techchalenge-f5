from __future__ import annotations

import pytest
from pydantic import ValidationError, parse_obj_as

from app.request_schemas import PredictRequest, RecordModel, RecordsEnvelope


def test_predict_request_accepts_single_dict_structure() -> None:
    parsed = parse_obj_as(PredictRequest, {"a": 1, "b": 2})
    assert isinstance(parsed, RecordModel)


def test_predict_request_accepts_list_of_dicts_structure() -> None:
    parsed = parse_obj_as(PredictRequest, [{"a": 1}, {"a": 2}])
    assert isinstance(parsed, list)
    assert all(isinstance(item, RecordModel) for item in parsed)


def test_predict_request_accepts_envelope_structure() -> None:
    parsed = parse_obj_as(PredictRequest, {"records": [{"a": 1}, {"a": 2}]})
    assert isinstance(parsed, RecordsEnvelope)
    assert len(parsed.records) == 2


def test_predict_request_rejects_invalid_envelope_records_type() -> None:
    with pytest.raises(ValidationError):
        parse_obj_as(PredictRequest, {"records": "x"})


def test_predict_request_rejects_numeric_payload() -> None:
    with pytest.raises(ValidationError):
        parse_obj_as(PredictRequest, 123)
