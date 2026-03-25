"""Shared test fixtures."""

from __future__ import annotations

import pytest

from peven.petri.executors import _REGISTRY


@pytest.fixture(autouse=True)
def _clean_executor_registry():
    """Snapshot and restore the executor registry between tests."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)
