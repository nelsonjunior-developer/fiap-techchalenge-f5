"""Pydantic request schemas for /predict structural validation."""

from __future__ import annotations

from typing import Any, Dict, List, Union

from pydantic import BaseModel, root_validator


class RecordModel(BaseModel):
    """Single record payload model with dynamic keys."""

    __root__: Dict[str, Any]

    @root_validator(pre=True)
    def _reject_reserved_records_key(cls, values: Any) -> Any:
        raw = values
        if isinstance(values, dict) and "__root__" in values:
            raw = values.get("__root__")
        if isinstance(raw, dict) and "records" in raw:
            raise ValueError("reserved key 'records' must use envelope format")
        return values

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__root__)


class RecordsEnvelope(BaseModel):
    records: List[RecordModel]


PredictRequest = Union[RecordsEnvelope, List[RecordModel], RecordModel]


__all__ = ["PredictRequest", "RecordModel", "RecordsEnvelope"]

