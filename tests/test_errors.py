"""Tests for whest error classes — message formatting invariants."""

from __future__ import annotations

import pytest

from whest.errors import (
    BudgetExhaustedError,
    NoBudgetContextError,
    SymmetryError,
    UnsupportedFunctionError,
    WhestError,
    WhestWarning,
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


def test_min_version_message_unchanged():
    err = UnsupportedFunctionError("bitwise_count", min_version="2.1")
    msg = str(err)
    assert "numpy.bitwise_count" in msg
    assert "requires numpy >= 2.1" in msg
    assert err.func_name == "bitwise_count"
    assert err.min_version == "2.1"


def test_max_version_with_replacement():
    err = UnsupportedFunctionError("in1d", max_version="2.4", replacement="isin")
    msg = str(err)
    assert "in1d" in msg
    assert "removed in numpy 2.4" in msg
    assert "isin" in msg
    assert err.max_version == "2.4"
    assert err.replacement == "isin"


def test_max_version_without_replacement():
    err = UnsupportedFunctionError("trapz", max_version="2.4")
    msg = str(err)
    assert "trapz" in msg
    assert "removed in numpy 2.4" in msg
    assert err.replacement is None


def test_neither_version_set():
    err = UnsupportedFunctionError("some_op")
    msg = str(err)
    assert "some_op" in msg
    assert "not supported by whest" in msg
