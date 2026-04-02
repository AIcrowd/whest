# src/mechestim/linalg/_solvers.py
"""Linear solver wrappers with FLOP counting."""
from __future__ import annotations
import numpy as _np
from mechestim._symmetric import SymmetricTensor, as_symmetric
from mechestim._validation import require_budget, validate_ndarray


def solve_cost(n: int, nrhs: int = 1, symmetric: bool = False) -> int:
    """FLOP cost of solving Ax = b for (n, n) matrix A with nrhs right-hand sides.

    Formula: n^3/3 + n*nrhs (Cholesky) if symmetric, else 2*n^3/3 + n^2*nrhs (LU).
    Source: Golub & Van Loan, "Matrix Computations", 4th ed.
    """
    if symmetric:
        return max(n ** 3 // 3 + n * nrhs, 1)
    return max(2 * n ** 3 // 3 + n ** 2 * nrhs, 1)


def solve(a, b):
    """Solve linear system with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"First argument must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    nrhs = b.shape[1] if b.ndim == 2 else 1
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = solve_cost(n, nrhs=nrhs, symmetric=is_symmetric)
    budget.deduct("linalg.solve", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.solve(a, b)


def inv_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of matrix inverse. n^3/3 + n^3/2 if symmetric, else n^3."""
    if symmetric:
        return max(n ** 3 // 3 + n ** 3 // 2, 1)
    return max(n ** 3, 1)


def inv(a):
    """Matrix inverse with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = inv_cost(n, symmetric=is_symmetric)
    budget.deduct("linalg.inv", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    result = _np.linalg.inv(a)
    if is_symmetric:
        result = as_symmetric(result, dims=(0, 1))
    return result


def lstsq_cost(m: int, n: int) -> int:
    """FLOP cost of least-squares solution of an (m, n) system.
    Formula: m * n * min(m, n)
    Source: NumPy uses LAPACK gelsd (SVD-based) by default.
    """
    return max(m * n * min(m, n), 1)


def lstsq(a, b, rcond=None):
    """Least-squares solution with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2:
        raise ValueError(f"First argument must be 2D, got {a.ndim}D")
    m, n = a.shape
    cost = lstsq_cost(m, n)
    budget.deduct("linalg.lstsq", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.lstsq(a, b, rcond=rcond)


def pinv_cost(m: int, n: int) -> int:
    """FLOP cost of pseudoinverse of an (m, n) matrix.
    Formula: m * n * min(m, n)
    Source: Computed via SVD.
    """
    return max(m * n * min(m, n), 1)


def pinv(a, rcond=None, hermitian=False):
    """Pseudoinverse with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2:
        raise ValueError(f"Input must be 2D, got {a.ndim}D")
    m, n = a.shape
    cost = pinv_cost(m, n)
    budget.deduct("linalg.pinv", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    kwargs = {"hermitian": hermitian}
    if rcond is not None:
        kwargs["rcond"] = rcond
    return _np.linalg.pinv(a, **kwargs)


def tensorsolve_cost(a_shape: tuple, ind: int | None = None) -> int:
    """FLOP cost of tensor solve. Reduces to solve after reshaping.
    Formula: n^3 where n = product of dims from ind onward.
    Source: Delegates to solve internally.
    """
    if ind is None:
        ind = 2
    n = 1
    for d in a_shape[ind:]:
        n *= d
    return max(n ** 3, 1)


def tensorsolve(a, b, axes=None):
    """Tensor solve with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    cost = tensorsolve_cost(a.shape)
    budget.deduct("linalg.tensorsolve", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.tensorsolve(a, b, axes=axes)


def tensorinv_cost(a_shape: tuple, ind: int = 2) -> int:
    """FLOP cost of tensor inverse. Reduces to inv after reshaping.
    Formula: n^3 where n = product of first `ind` dims.
    Source: Delegates to inv internally.
    """
    n = 1
    for d in a_shape[:ind]:
        n *= d
    return max(n ** 3, 1)


def tensorinv(a, ind=2):
    """Tensor inverse with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    cost = tensorinv_cost(a.shape, ind=ind)
    budget.deduct("linalg.tensorinv", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.tensorinv(a, ind=ind)
