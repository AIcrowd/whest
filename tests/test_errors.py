"""Tests for whest error classes."""

import pytest

from whest.errors import (
    BudgetExhaustedError,
    WhestError,
    WhestWarning,
    NoBudgetContextError,
    SymmetryError,
)


def test_budget_exhausted_error_is_whest_error():
    with pytest.raises(WhestError):
        raise BudgetExhaustedError("einsum", flop_cost=100, flops_remaining=50)


def test_budget_exhausted_error_attributes():
    err = BudgetExhaustedError("einsum", flop_cost=100, flops_remaining=50)
    assert err.op_name == "einsum"
    assert err.flop_cost == 100
    assert err.flops_remaining == 50
    assert "einsum" in str(err)
    assert "100" in str(err)
    assert "50" in str(err)


def test_no_budget_context_error_is_whest_error():
    with pytest.raises(WhestError):
        raise NoBudgetContextError()


def test_no_budget_context_error_message():
    err = NoBudgetContextError()
    assert "BudgetContext" in str(err)


def test_symmetry_error_attributes():
    err = SymmetryError(axes=(0, 1), max_deviation=0.5)
    assert err.axes == (0, 1)
    assert err.max_deviation == 0.5
    assert "0, 1" in str(err)


def test_whest_warning_is_warning():
    assert issubclass(WhestWarning, UserWarning)
