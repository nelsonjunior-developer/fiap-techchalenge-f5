from __future__ import annotations

import importlib

import pytest


def test_streamlit_app_module_import_smoke() -> None:
    try:
        importlib.import_module("streamlit")
    except ModuleNotFoundError:
        pytest.skip("streamlit not installed in current test environment")
    except Exception as exc:  # pragma: no cover - env-dependent conflicts (e.g. protobuf)
        pytest.skip(f"streamlit import unavailable in current environment: {exc}")

    module = importlib.import_module("dashboards.streamlit_app")
    assert hasattr(module, "main")

