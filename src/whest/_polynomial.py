"""Counted polynomial operations for whest."""

from __future__ import annotations

import inspect as _inspect

import numpy as _np

from whest._docstrings import attach_docstring
from whest._validation import require_budget

# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------


def polyval_cost(deg: int, m: int) -> int:
    """Cost for polyval: Horner's method = m * deg FLOPs (FMA=1 op)."""
    return max(m * deg, 1)


def polyadd_cost(n1: int, n2: int) -> int:
    """Cost for polyadd: max(n1, n2) FLOPs."""
    return max(n1, n2, 1)


def polysub_cost(n1: int, n2: int) -> int:
    """Cost for polysub: max(n1, n2) FLOPs."""
    return max(n1, n2, 1)


def polyder_cost(n: int) -> int:
    """Cost for polyder: n FLOPs (n = len of coeffs)."""
    return max(n, 1)


def polyint_cost(n: int) -> int:
    """Cost for polyint: n FLOPs (n = len of coeffs)."""
    return max(n, 1)


def polymul_cost(n1: int, n2: int) -> int:
    """Cost for polymul: n1 * n2 FLOPs."""
    return max(n1 * n2, 1)


def polydiv_cost(n1: int, n2: int) -> int:
    """Cost for polydiv: n1 * n2 FLOPs."""
    return max(n1 * n2, 1)


def polyfit_cost(m: int, deg: int) -> int:
    """Cost for polyfit: 2 * m * (deg+1)^2 FLOPs."""
    return max(2 * m * (deg + 1) ** 2, 1)


def poly_cost(n: int) -> int:
    """Cost for poly: $n^2$ FLOPs."""
    return max(n * n, 1)


def roots_cost(n: int) -> int:
    """Cost for roots: $n^3$ FLOPs (companion matrix eigendecomposition, simplified)."""
    return max(n**3, 1)


# ---------------------------------------------------------------------------
# Wrapped operations
# ---------------------------------------------------------------------------


def polyval(p, x):
    """Evaluate a polynomial at given points. Wraps ``numpy.polyval``."""
    budget = require_budget()
    p = _np.asarray(p)
    x = _np.asarray(x)
    deg = len(p) - 1
    m = x.size
    cost = polyval_cost(deg, m)
    with budget.deduct(
        "polyval", flop_cost=cost, subscripts=None, shapes=(p.shape, x.shape)
    ):
        result = _np.polyval(p, x)
    return result


attach_docstring(
    polyval, _np.polyval, "counted_custom", "m * deg FLOPs (Horner's method, FMA=1)"
)


def polyadd(a1, a2):
    """Add two polynomials. Wraps ``numpy.polyadd``."""
    budget = require_budget()
    a1 = _np.asarray(a1)
    a2 = _np.asarray(a2)
    n1 = len(a1)
    n2 = len(a2)
    cost = polyadd_cost(n1, n2)
    with budget.deduct(
        "polyadd", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    ):
        result = _np.polyadd(a1, a2)
    return result


attach_docstring(polyadd, _np.polyadd, "counted_custom", "max(n1, n2) FLOPs")


def polysub(a1, a2):
    """Subtract two polynomials. Wraps ``numpy.polysub``."""
    budget = require_budget()
    a1 = _np.asarray(a1)
    a2 = _np.asarray(a2)
    n1 = len(a1)
    n2 = len(a2)
    cost = polysub_cost(n1, n2)
    with budget.deduct(
        "polysub", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    ):
        result = _np.polysub(a1, a2)
    return result


attach_docstring(polysub, _np.polysub, "counted_custom", "max(n1, n2) FLOPs")


def polyder(p, m=1):
    """Differentiate a polynomial. Wraps ``numpy.polyder``."""
    budget = require_budget()
    p = _np.asarray(p)
    n = len(p)
    cost = polyder_cost(n)
    with budget.deduct("polyder", flop_cost=cost, subscripts=None, shapes=(p.shape,)):
        result = _np.polyder(p, m=m)
    return result


attach_docstring(polyder, _np.polyder, "counted_custom", "n FLOPs (n = len(coeffs))")


def polyint(p, m=1, k=None):
    """Integrate a polynomial. Wraps ``numpy.polyint``."""
    budget = require_budget()
    p = _np.asarray(p)
    n = len(p)
    cost = polyint_cost(n)
    with budget.deduct("polyint", flop_cost=cost, subscripts=None, shapes=(p.shape,)):
        if k is None:
            result = _np.polyint(p, m=m)
        else:
            result = _np.polyint(p, m=m, k=k)
    return result


attach_docstring(polyint, _np.polyint, "counted_custom", "n FLOPs (n = len(coeffs))")


def polymul(a1, a2):
    """Multiply polynomials. Wraps ``numpy.polymul``."""
    budget = require_budget()
    a1 = _np.asarray(a1)
    a2 = _np.asarray(a2)
    n1 = len(a1)
    n2 = len(a2)
    cost = polymul_cost(n1, n2)
    with budget.deduct(
        "polymul", flop_cost=cost, subscripts=None, shapes=(a1.shape, a2.shape)
    ):
        result = _np.polymul(a1, a2)
    return result


attach_docstring(polymul, _np.polymul, "counted_custom", "n1 * n2 FLOPs")


def polydiv(u, v):
    """Divide one polynomial by another. Wraps ``numpy.polydiv``."""
    budget = require_budget()
    u = _np.atleast_1d(_np.asarray(u))
    v = _np.atleast_1d(_np.asarray(v))
    n1 = len(u)
    n2 = len(v)
    cost = polydiv_cost(n1, n2)
    with budget.deduct(
        "polydiv", flop_cost=cost, subscripts=None, shapes=(u.shape, v.shape)
    ):
        result = _np.polydiv(u, v)
    return result


attach_docstring(polydiv, _np.polydiv, "counted_custom", "n1 * n2 FLOPs")


def polyfit(x, y, deg, **kwargs):
    """Least-squares polynomial fit. Wraps ``numpy.polyfit``."""
    budget = require_budget()
    x = _np.asarray(x)
    m = len(x)
    cost = polyfit_cost(m, deg)
    with budget.deduct("polyfit", flop_cost=cost, subscripts=None, shapes=(x.shape,)):
        result = _np.polyfit(x, y, deg, **kwargs)
    return result


attach_docstring(polyfit, _np.polyfit, "counted_custom", "2 * m * (deg+1)^2 FLOPs")
polyfit.__signature__ = _inspect.signature(_np.polyfit)


def poly(seq_of_zeros):
    """Return polynomial coefficients from roots. Wraps ``numpy.poly``."""
    budget = require_budget()
    seq = _np.asarray(seq_of_zeros)
    # If 2D (square matrix), n = shape[0]; if 1D, n = len(seq)
    if seq.ndim == 2:
        n = seq.shape[0]
    else:
        n = len(seq)
    cost = poly_cost(n)
    with budget.deduct("poly", flop_cost=cost, subscripts=None, shapes=(seq.shape,)):
        result = _np.poly(seq_of_zeros)
    return result


attach_docstring(poly, _np.poly, "counted_custom", "n^2 FLOPs")


def roots(p):
    """Return the roots of a polynomial with given coefficients. Wraps ``numpy.roots``."""
    budget = require_budget()
    p = _np.asarray(p)
    n = len(p) - 1  # degree = number of roots
    cost = roots_cost(n)
    with budget.deduct("roots", flop_cost=cost, subscripts=None, shapes=(p.shape,)):
        result = _np.roots(p)
    return result


attach_docstring(
    roots, _np.roots, "counted_custom", "n^3 FLOPs (companion matrix eig, simplified)"
)

import sys as _sys  # noqa: E402

from whest._ndarray import wrap_module_returns as _wrap_module_returns  # noqa: E402

_wrap_module_returns(_sys.modules[__name__])
