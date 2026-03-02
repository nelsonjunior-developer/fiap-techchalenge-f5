from __future__ import annotations

import importlib

import pytest


def test_ops_dashboard_module_import_smoke() -> None:
    try:
        importlib.import_module("streamlit")
    except ModuleNotFoundError:
        pytest.skip("streamlit not installed in current test environment")
    except Exception as exc:  # pragma: no cover - env-dependent conflicts
        pytest.skip(f"streamlit import unavailable in current environment: {exc}")

    module = importlib.import_module("dashboards.ops_dashboard")
    assert hasattr(module, "main")
