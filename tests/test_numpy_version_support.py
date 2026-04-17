"""Tests for multi-version numpy support (2.0, 2.1, 2.2)."""

import numpy as np
import pytest

import whest as we
from whest._budget import BudgetContext
from whest.errors import UnsupportedFunctionError

# ---------------------------------------------------------------------------
# UnsupportedFunctionError
# ---------------------------------------------------------------------------


def test_unsupported_function_error_attributes():
    err = UnsupportedFunctionError("matvec", min_version="2.2")
    assert err.func_name == "matvec"
    assert err.min_version == "2.2"
    assert "numpy.matvec" in str(err)
    assert "numpy >= 2.2" in str(err)


def test_unsupported_function_error_is_whest_error():
    assert issubclass(UnsupportedFunctionError, we.WhestError)


# ---------------------------------------------------------------------------
# Version-gated functions: test they either work or raise correctly
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(np, "vecdot"), reason="requires numpy >= 2.1")
def test_vecdot_cost():
    a = np.ones((10, 5))
    b = np.ones((10, 5))
    with BudgetContext(flop_budget=1_000_000) as ctx:
        result = we.vecdot(a, b)
    assert result.shape == (10,)
    # Cost = output_size * contracted_axis = 10 * 5 = 50
    assert ctx.flops_used == 50


@pytest.mark.skipif(hasattr(np, "vecdot"), reason="only on numpy < 2.1")
def test_vecdot_raises_on_old_numpy():
    with BudgetContext(flop_budget=1_000_000):
        with pytest.raises(UnsupportedFunctionError, match="numpy >= 2.1"):
            we.vecdot(np.ones(3), np.ones(3))


@pytest.mark.skipif(not hasattr(np, "matvec"), reason="requires numpy >= 2.2")
def test_matvec_cost():
    A = np.ones((3, 4))
    v = np.ones(4)
    with BudgetContext(flop_budget=1_000_000) as ctx:
        result = we.matvec(A, v)
    assert result.shape == (3,)
    # Cost = output_size * contracted_axis = 3 * 4 = 12
    assert ctx.flops_used == 12


@pytest.mark.skipif(hasattr(np, "matvec"), reason="only on numpy < 2.2")
def test_matvec_raises_on_old_numpy():
    with BudgetContext(flop_budget=1_000_000):
        with pytest.raises(UnsupportedFunctionError, match="numpy >= 2.2"):
            we.matvec(np.ones((3, 4)), np.ones(4))


@pytest.mark.skipif(not hasattr(np, "vecmat"), reason="requires numpy >= 2.2")
def test_vecmat_cost():
    v = np.ones(3)
    A = np.ones((3, 4))
    with BudgetContext(flop_budget=1_000_000) as ctx:
        result = we.vecmat(v, A)
    assert result.shape == (4,)
    # Cost = output_size * contracted_axis = 4 * 3 = 12
    assert ctx.flops_used == 12


@pytest.mark.skipif(hasattr(np, "vecmat"), reason="only on numpy < 2.2")
def test_vecmat_raises_on_old_numpy():
    with BudgetContext(flop_budget=1_000_000):
        with pytest.raises(UnsupportedFunctionError, match="numpy >= 2.2"):
            we.vecmat(np.ones(3), np.ones((3, 4)))


@pytest.mark.skipif(not hasattr(np, "matvec"), reason="requires numpy >= 2.2")
def test_matvec_batched():
    A = np.ones((2, 3, 4))
    v = np.ones((2, 4))
    with BudgetContext(flop_budget=1_000_000) as ctx:
        result = we.matvec(A, v)
    assert result.shape == (2, 3)
    # Cost = output_size * contracted_axis = 6 * 4 = 24
    assert ctx.flops_used == 24


@pytest.mark.skipif(not hasattr(np, "bitwise_count"), reason="requires numpy >= 2.1")
def test_bitwise_count_works():
    with BudgetContext(flop_budget=1_000_000) as ctx:
        result = we.bitwise_count(np.array([0, 1, 3, 7, 255], dtype=np.uint8))
    assert list(result) == [0, 1, 2, 3, 8]
    assert ctx.flops_used == 5


@pytest.mark.skipif(not hasattr(np, "cumulative_sum"), reason="requires numpy >= 2.1")
def test_cumulative_sum_works():
    with BudgetContext(flop_budget=1_000_000) as ctx:
        result = we.cumulative_sum(np.array([1, 2, 3, 4]))
    assert list(result) == [1, 3, 6, 10]


@pytest.mark.skipif(not hasattr(np, "unstack"), reason="requires numpy >= 2.1")
def test_unstack_works():
    with BudgetContext(flop_budget=1_000_000) as ctx:
        result = we.unstack(np.array([[1, 2], [3, 4]]))
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Import always succeeds (stubs are importable)
# ---------------------------------------------------------------------------


def test_all_version_gated_functions_importable():
    """All version-gated functions should be importable regardless of numpy version."""
    assert hasattr(we, "vecdot")
    assert hasattr(we, "matvec")
    assert hasattr(we, "vecmat")
    assert hasattr(we, "bitwise_count")
    assert hasattr(we, "cumulative_sum")
    assert hasattr(we, "cumulative_prod")
    assert hasattr(we, "unstack")


def test_in1d_removed_in_numpy_2_4(monkeypatch):
    """When numpy.in1d is unavailable, we.in1d raises UnsupportedFunctionError."""
    import importlib
    import sys

    import numpy as _np

    import whest._sorting_ops as _sorting_ops

    # Save the real module object so we can restore it after the test —
    # otherwise the reload below leaves sys.modules pointing at the stub
    # version, which would leak to any later code that does a fresh import.
    _original_sorting_ops = sys.modules["whest._sorting_ops"]

    try:
        monkeypatch.delattr(_np, "in1d", raising=False)
        importlib.reload(_sorting_ops)

        with pytest.raises(UnsupportedFunctionError) as exc_info:
            _sorting_ops.in1d([1, 2, 3], [2, 3, 4])
        err = exc_info.value
        assert err.max_version == "2.4"
        assert err.replacement == "isin"
        assert "removed in numpy 2.4" in str(err)
    finally:
        # Restore sys.modules so later tests / consumers see the real in1d.
        sys.modules["whest._sorting_ops"] = _original_sorting_ops


def test_trapz_removed_in_numpy_2_4(monkeypatch):
    """When numpy.trapz is unavailable, we.trapz raises UnsupportedFunctionError."""
    import importlib
    import sys

    import numpy as _np

    import whest._pointwise as _pointwise

    _original_pointwise = sys.modules["whest._pointwise"]

    try:
        monkeypatch.delattr(_np, "trapz", raising=False)
        importlib.reload(_pointwise)

        with pytest.raises(UnsupportedFunctionError) as exc_info:
            _pointwise.trapz([1.0, 2.0, 3.0])
        err = exc_info.value
        assert err.max_version == "2.4"
        assert err.replacement == "trapezoid"
        assert "removed in numpy 2.4" in str(err)
    finally:
        sys.modules["whest._pointwise"] = _original_pointwise
