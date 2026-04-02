"""Tests for counted polynomial operations."""

import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim._polynomial import (
    poly,
    polyadd,
    polyder,
    polydiv,
    polyfit,
    polyint,
    polymul,
    polysub,
    polyval,
    roots,
)
from mechestim.errors import NoBudgetContextError

# ---------------------------------------------------------------------------
# polyval
# ---------------------------------------------------------------------------


def test_polyval_result():
    p = numpy.array([1.0, -2.0, 3.0])  # x^2 - 2x + 3
    x = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = polyval(p, x)
        assert numpy.allclose(result, numpy.polyval(p, x))


def test_polyval_cost():
    # coeffs [1, -2, 3] -> deg=2, 5 points -> cost = 2*5*2 = 20
    p = numpy.array([1.0, -2.0, 3.0])
    x = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])
    with BudgetContext(flop_budget=10**6) as budget:
        polyval(p, x)
        assert budget.flops_used == 20


def test_polyval_no_budget():
    p = numpy.array([1.0, 0.0])
    x = numpy.array([1.0])
    with pytest.raises(NoBudgetContextError):
        polyval(p, x)


# ---------------------------------------------------------------------------
# polyadd
# ---------------------------------------------------------------------------


def test_polyadd_result():
    a1 = numpy.array([1.0, 2.0, 3.0])
    a2 = numpy.array([4.0, 5.0])
    with BudgetContext(flop_budget=10**6):
        result = polyadd(a1, a2)
        assert numpy.allclose(result, numpy.polyadd(a1, a2))


def test_polyadd_cost():
    # [1,2,3] + [4,5] -> max(3, 2) = 3
    a1 = numpy.array([1.0, 2.0, 3.0])
    a2 = numpy.array([4.0, 5.0])
    with BudgetContext(flop_budget=10**6) as budget:
        polyadd(a1, a2)
        assert budget.flops_used == 3


def test_polyadd_no_budget():
    with pytest.raises(NoBudgetContextError):
        polyadd([1.0], [2.0])


# ---------------------------------------------------------------------------
# polysub
# ---------------------------------------------------------------------------


def test_polysub_result():
    a1 = numpy.array([1.0, 2.0])
    a2 = numpy.array([3.0, 4.0, 5.0])
    with BudgetContext(flop_budget=10**6):
        result = polysub(a1, a2)
        assert numpy.allclose(result, numpy.polysub(a1, a2))


def test_polysub_cost():
    # [1,2] - [3,4,5] -> max(2, 3) = 3
    a1 = numpy.array([1.0, 2.0])
    a2 = numpy.array([3.0, 4.0, 5.0])
    with BudgetContext(flop_budget=10**6) as budget:
        polysub(a1, a2)
        assert budget.flops_used == 3


def test_polysub_no_budget():
    with pytest.raises(NoBudgetContextError):
        polysub([1.0], [2.0])


# ---------------------------------------------------------------------------
# polyder
# ---------------------------------------------------------------------------


def test_polyder_result():
    p = numpy.array([1.0, 2.0, 3.0, 4.0])  # x^3 + 2x^2 + 3x + 4
    with BudgetContext(flop_budget=10**6):
        result = polyder(p)
        assert numpy.allclose(result, numpy.polyder(p))


def test_polyder_cost():
    # 4 coeffs -> cost = 4
    p = numpy.array([1.0, 2.0, 3.0, 4.0])
    with BudgetContext(flop_budget=10**6) as budget:
        polyder(p)
        assert budget.flops_used == 4


def test_polyder_no_budget():
    with pytest.raises(NoBudgetContextError):
        polyder([1.0, 2.0])


# ---------------------------------------------------------------------------
# polyint
# ---------------------------------------------------------------------------


def test_polyint_result():
    p = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6):
        result = polyint(p)
        assert numpy.allclose(result, numpy.polyint(p))


def test_polyint_cost():
    # 3 coeffs -> cost = 3
    p = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6) as budget:
        polyint(p)
        assert budget.flops_used == 3


def test_polyint_no_budget():
    with pytest.raises(NoBudgetContextError):
        polyint([1.0, 2.0])


# ---------------------------------------------------------------------------
# polymul
# ---------------------------------------------------------------------------


def test_polymul_result():
    a1 = numpy.array([1.0, 2.0])
    a2 = numpy.array([3.0, 4.0, 5.0])
    with BudgetContext(flop_budget=10**6):
        result = polymul(a1, a2)
        assert numpy.allclose(result, numpy.polymul(a1, a2))


def test_polymul_cost():
    # len 2 * len 3 -> cost = 6
    a1 = numpy.array([1.0, 2.0])
    a2 = numpy.array([3.0, 4.0, 5.0])
    with BudgetContext(flop_budget=10**6) as budget:
        polymul(a1, a2)
        assert budget.flops_used == 6


def test_polymul_no_budget():
    with pytest.raises(NoBudgetContextError):
        polymul([1.0], [2.0])


# ---------------------------------------------------------------------------
# polydiv
# ---------------------------------------------------------------------------


def test_polydiv_result():
    u = numpy.array([1.0, 2.0, 3.0])
    v = numpy.array([1.0, 1.0])
    with BudgetContext(flop_budget=10**6):
        q, r = polydiv(u, v)
        eq, er = numpy.polydiv(u, v)
        assert numpy.allclose(q, eq)
        assert numpy.allclose(r, er)


def test_polydiv_cost():
    # len 3 * len 2 -> cost = 6
    u = numpy.array([1.0, 2.0, 3.0])
    v = numpy.array([1.0, 1.0])
    with BudgetContext(flop_budget=10**6) as budget:
        polydiv(u, v)
        assert budget.flops_used == 6


def test_polydiv_no_budget():
    with pytest.raises(NoBudgetContextError):
        polydiv([1.0, 0.0], [1.0])


# ---------------------------------------------------------------------------
# polyfit
# ---------------------------------------------------------------------------


def test_polyfit_result():
    x = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])
    with BudgetContext(flop_budget=10**6):
        result = polyfit(x, y, 2)
        assert numpy.allclose(result, numpy.polyfit(x, y, 2), atol=1e-10)


def test_polyfit_cost():
    # 5 points, deg 2 -> cost = 2 * 5 * (2+1)^2 = 2 * 5 * 9 = 90
    x = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])
    with BudgetContext(flop_budget=10**6) as budget:
        polyfit(x, y, 2)
        assert budget.flops_used == 90


def test_polyfit_no_budget():
    with pytest.raises(NoBudgetContextError):
        polyfit([0.0, 1.0], [0.0, 1.0], 1)


# ---------------------------------------------------------------------------
# poly
# ---------------------------------------------------------------------------


def test_poly_result():
    zeros = numpy.array([1.0, 2.0, 3.0, 4.0])
    with BudgetContext(flop_budget=10**6):
        result = poly(zeros)
        assert numpy.allclose(result, numpy.poly(zeros))


def test_poly_cost():
    # 4 roots -> cost = 4^2 = 16
    zeros = numpy.array([1.0, 2.0, 3.0, 4.0])
    with BudgetContext(flop_budget=10**6) as budget:
        poly(zeros)
        assert budget.flops_used == 16


def test_poly_no_budget():
    with pytest.raises(NoBudgetContextError):
        poly([1.0, 2.0])


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------


def test_roots_result():
    p = numpy.array([1.0, -6.0, 11.0, -6.0])  # (x-1)(x-2)(x-3)
    with BudgetContext(flop_budget=10**6):
        result = roots(p)
        expected = numpy.roots(p)
        assert numpy.allclose(sorted(result.real), sorted(expected.real), atol=1e-10)


def test_roots_cost():
    # 4 coeffs -> n = len(p)-1 = 3 -> cost = 10 * 3^3 = 10 * 27 = 270
    p = numpy.array([1.0, -6.0, 11.0, -6.0])
    with BudgetContext(flop_budget=10**6) as budget:
        roots(p)
        assert budget.flops_used == 270


def test_roots_no_budget():
    with pytest.raises(NoBudgetContextError):
        roots([1.0, -1.0])


# ---------------------------------------------------------------------------
# Integration with __init__.py
# ---------------------------------------------------------------------------


def test_import_from_mechestim():
    import mechestim as me

    assert hasattr(me, "polyval")
    assert hasattr(me, "polyadd")
    assert hasattr(me, "polysub")
    assert hasattr(me, "polyder")
    assert hasattr(me, "polyint")
    assert hasattr(me, "polymul")
    assert hasattr(me, "polydiv")
    assert hasattr(me, "polyfit")
    assert hasattr(me, "poly")
    assert hasattr(me, "roots")
