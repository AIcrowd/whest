# src/whest/linalg/_solvers.py
"""Linear solver wrappers with FLOP counting."""

from __future__ import annotations

import numpy as _np

from whest._docstrings import attach_docstring
from whest._symmetric import SymmetricTensor, as_symmetric
from whest._validation import require_budget
from whest.errors import SymmetryError


def _batch_size(shape):
    """Number of matrices in a batched array."""
    if len(shape) <= 2:
        return 1
    result = 1
    for d in shape[:-2]:
        result *= d
    return result


def _has_zero_dim(shape):
    """Check if any matrix dimension is zero."""
    return len(shape) >= 2 and (shape[-2] == 0 or shape[-1] == 0)


def solve_cost(n: int, nrhs: int = 1, symmetric: bool = False) -> int:
    r"""FLOP cost of solving a linear system Ax = b.

    Parameters
    ----------
    n : int
        Matrix dimension.
    nrhs : int, optional
        Ignored (kept for API compatibility). Default is 1.
    symmetric : bool, optional
        Ignored (kept for API compatibility). Default is False.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$.

    Notes
    -----
    Simplified cubic cost model for linear solve.
    """
    return max(n**3, 1)


def solve(a, b):
    """Solve linear system with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    cost = solve_cost(n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.solve", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.solve(a, b)
    return result


attach_docstring(solve, _np.linalg.solve, "linalg", r"$n^3$ FLOPs")


def inv_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of matrix inverse.

    Parameters
    ----------
    n : int
        Matrix dimension.
    symmetric : bool, optional
        If True, assume symmetric input. Default is False.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    Uses $n^3/3 + n^3$ for symmetric input (Cholesky factorization + n
    triangular solves against identity), or $n^3$ for general input (LU-based).
    """
    if symmetric:
        return max(n**3 // 3 + n**3, 1)
    return max(n**3, 1)


def inv(a):
    """Matrix inverse with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = a.shape[-1]
    batch = _batch_size(a.shape)
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = (
        inv_cost(n, symmetric=is_symmetric) * batch if not _has_zero_dim(a.shape) else 0
    )
    with budget.deduct(
        "linalg.inv", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.inv(a)
    if is_symmetric:
        try:
            result = as_symmetric(result, symmetry=a.symmetry)
        except SymmetryError:
            pass
    return result


attach_docstring(
    inv,
    _np.linalg.inv,
    "linalg",
    r"$n^3$ FLOPs, or $n^3/3 + n^3$ for SymmetricTensor input. Returns SymmetricTensor if input is symmetric.",
)


def lstsq_cost(m: int, n: int) -> int:
    """FLOP cost of least-squares solution.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    int
        Estimated FLOP count: m * n * min(m, n).

    Notes
    -----
    NumPy uses LAPACK ``gelsd`` (SVD-based) by default.
    """
    return max(m * n * min(m, n), 1)


def lstsq(a, b, rcond=None):
    """Least-squares solution with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    m, n = a.shape[-2], a.shape[-1]
    batch = _batch_size(a.shape)
    cost = lstsq_cost(m, n) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.lstsq", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.lstsq(a, b, rcond=rcond)
    return result


attach_docstring(
    lstsq, _np.linalg.lstsq, "linalg", r"$m \cdot n \cdot \min(m,n)$ FLOPs (SVD-based)"
)


def pinv_cost(m: int, n: int) -> int:
    """FLOP cost of pseudoinverse.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    int
        Estimated FLOP count: m * n * min(m, n).

    Notes
    -----
    Computed via SVD.
    """
    return max(m * n * min(m, n), 1)


def pinv(a, rcond=None, hermitian=False, *, rtol=None):
    """Pseudoinverse with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    m, n = a.shape[-2], a.shape[-1]
    batch = _batch_size(a.shape)
    cost = pinv_cost(m, n) * batch if not _has_zero_dim(a.shape) else 0
    kwargs = {"hermitian": hermitian}
    if rcond is not None:
        kwargs["rcond"] = rcond
    if rtol is not None:
        kwargs["rtol"] = rtol
    with budget.deduct(
        "linalg.pinv", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.pinv(a, **kwargs)
    return result


attach_docstring(
    pinv, _np.linalg.pinv, "linalg", r"$m \cdot n \cdot \min(m,n)$ FLOPs (SVD-based)"
)


def tensorsolve_cost(a_shape: tuple, ind: int | None = None) -> int:
    """FLOP cost of tensor solve.

    Parameters
    ----------
    a_shape : tuple of int
        Shape of the coefficient tensor.
    ind : int or None, optional
        Number of leading indices for the solution. Default is 2.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$ where $n$ = product of trailing dims.

    Notes
    -----
    Reduces to a standard linear solve after reshaping.
    """
    if ind is None:
        ind = 2
    n = 1
    for d in a_shape[ind:]:
        n *= d
    return max(n**3, 1)


def tensorsolve(a, b, axes=None):
    """Tensor solve with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    cost = tensorsolve_cost(a.shape)
    with budget.deduct(
        "linalg.tensorsolve", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.tensorsolve(a, b, axes=axes)
    return result


attach_docstring(
    tensorsolve,
    _np.linalg.tensorsolve,
    "linalg",
    r"$n^3$ FLOPs where n = product of trailing dims",
)


def tensorinv_cost(a_shape: tuple, ind: int = 2) -> int:
    """FLOP cost of tensor inverse.

    Parameters
    ----------
    a_shape : tuple of int
        Shape of the input tensor.
    ind : int, optional
        Number of leading indices. Default is 2.

    Returns
    -------
    int
        Estimated FLOP count: $n^3$ where $n$ = product of leading dims.

    Notes
    -----
    Reduces to a standard matrix inverse after reshaping.
    """
    n = 1
    for d in a_shape[:ind]:
        n *= d
    return max(n**3, 1)


def tensorinv(a, ind=2):
    """Tensor inverse with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    cost = tensorinv_cost(a.shape, ind=ind)
    with budget.deduct(
        "linalg.tensorinv", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        result = _np.linalg.tensorinv(a, ind=ind)
    return result


attach_docstring(
    tensorinv,
    _np.linalg.tensorinv,
    "linalg",
    r"$n^3$ FLOPs where n = product of leading dims",
)
