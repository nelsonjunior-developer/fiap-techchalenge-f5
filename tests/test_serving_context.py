from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.serving_context import (
    extract_model_identity,
    extract_operational_threshold,
    load_serving_metadata,
)


def test_load_serving_metadata_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    payload = load_serving_metadata(tmp_path / "missing-metadata.json")
    assert payload == {}


def test_load_serving_metadata_invalid_json_returns_empty_dict(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{invalid", encoding="utf-8")
    payload = load_serving_metadata(metadata_path)
    assert payload == {}


def test_load_serving_metadata_valid_json_returns_dict(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    expected = {"model_version": "v1"}
    metadata_path.write_text(json.dumps(expected), encoding="utf-8")
    payload = load_serving_metadata(metadata_path)
    assert payload == expected


def test_extract_operational_threshold_supports_modern_and_legacy_paths() -> None:
    modern, modern_notes = extract_operational_threshold(
        {"threshold_policy": {"operational_fixed_threshold": 0.31}}
    )
    assert modern == pytest.approx(0.31)
    assert modern_notes == []

    legacy, legacy_notes = extract_operational_threshold(
        {"threshold_policy": {"operational": {"threshold": 0.47}}}
    )
    assert legacy == pytest.approx(0.47)
    assert "threshold_from_legacy_policy_operational" in legacy_notes


def test_extract_operational_threshold_invalid_or_missing_uses_default() -> None:
    missing, missing_notes = extract_operational_threshold({})
    assert missing == pytest.approx(0.30)
    assert "fallback_default_threshold" in missing_notes

    invalid, invalid_notes = extract_operational_threshold(
        {"threshold_policy": {"operational_fixed_threshold": 1.8}}
    )
    assert invalid == pytest.approx(0.30)
    assert "fallback_default_threshold_invalid_metadata" in invalid_notes


def test_extract_model_identity_with_fallbacks() -> None:
    identity, notes = extract_model_identity(
        {"model_version": "v2", "model_family": "baseline_logreg", "variant": "none"}
    )
    assert identity["model_version"] == "v2"
    assert identity["model_family"] == "baseline_logreg"
    assert identity["variant"] == "none"
    assert notes == []

    fallback_identity, fallback_notes = extract_model_identity({})
    assert fallback_identity["model_version"] == "unknown"
    assert fallback_identity["model_family"] == "unknown"
    assert fallback_identity["variant"] == "unknown"
    assert "fallback_unknown_model_version" in fallback_notes
    assert "fallback_unknown_model_family" in fallback_notes
    assert "fallback_unknown_variant" in fallback_notes
