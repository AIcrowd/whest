"""Shared pytest configuration and fixtures."""

import pytest

import flopscope._weights as weights_module
from flopscope._budget import _reset_global_default, budget_reset
from flopscope._weights import reset_weights


@pytest.fixture(autouse=True)
def reset_global_budget():
    """Ensure no global default BudgetContext leaks between tests."""
    _reset_global_default()
    budget_reset()
    reset_weights()
    weights_module._WARNED_MESSAGES.clear()
    yield
    _reset_global_default()
    budget_reset()
    reset_weights()
    weights_module._WARNED_MESSAGES.clear()
