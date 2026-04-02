"""Shared pytest configuration and fixtures."""

import pytest

from mechestim._budget import _reset_global_default


@pytest.fixture(autouse=True)
def reset_global_budget():
    """Ensure no global default BudgetContext leaks between tests."""
    _reset_global_default()
    yield
    _reset_global_default()
