# src/mechestim/linalg/_properties.py
"""Matrix property wrappers with FLOP counting."""
from __future__ import annotations
import numpy as _np
from mechestim._validation import require_budget, validate_ndarray


def trace_cost(n: int) -> int:
    """FLOP cost of matrix trace. Formula: n. Source: Sum of n diagonal elements."""
    return max(n, 1)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """Matrix trace with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    n = min(a.shape[axis1], a.shape[axis2])
    if offset > 0:
        n = min(n, a.shape[axis2] - offset)
    elif offset < 0:
        n = min(n, a.shape[axis1] + offset)
    n = max(n, 0)
    cost = trace_cost(n)
    budget.deduct("linalg.trace", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)


def det_cost(n: int) -> int:
    """FLOP cost of determinant. Formula: n^3. Source: LU factorization."""
    return max(n ** 3, 1)


def det(a):
    """Determinant with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = det_cost(n)
    budget.deduct("linalg.det", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.det(a)


def slogdet_cost(n: int) -> int:
    """FLOP cost of sign and log-determinant. Formula: n^3. Source: Same as det."""
    return max(n ** 3, 1)


def slogdet(a):
    """Sign and log-determinant with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    cost = slogdet_cost(n)
    budget.deduct("linalg.slogdet", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.slogdet(a)


def norm_cost(shape: tuple, ord=None) -> int:
    """FLOP cost of matrix or vector norm. Dispatches on ord and dimensionality.
    Source: Direct analysis of norm definitions."""
    numel = 1
    for d in shape:
        numel *= d
    numel = max(numel, 1)
    if len(shape) == 1:
        if ord is None or ord == 2 or ord == -2:
            return numel
        elif ord in (1, -1, _np.inf, -_np.inf, 0):
            return numel
        else:
            return 2 * numel
    else:
        m, n = shape[-2], shape[-1]
        if ord is None or ord == "fro":
            return 2 * numel
        elif ord in (1, -1, _np.inf, -_np.inf):
            return numel
        elif ord == 2 or ord == -2:
            return m * n * min(m, n)
        elif ord == "nuc":
            return m * n * min(m, n)
        return numel


def norm(x, ord=None, axis=None, keepdims=False):
    """Matrix or vector norm with FLOP counting."""
    budget = require_budget()
    validate_ndarray(x)
    if axis is None:
        effective_shape = x.shape
    elif isinstance(axis, int):
        effective_shape = (x.shape[axis],)
    else:
        effective_shape = tuple(x.shape[ax] for ax in axis)
    cost = norm_cost(effective_shape, ord=ord)
    budget.deduct("linalg.norm", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def vector_norm_cost(shape: tuple, ord=None) -> int:
    """FLOP cost of vector norm. Source: Direct analysis."""
    numel = 1
    for d in shape:
        numel *= d
    numel = max(numel, 1)
    if ord is None or ord == 2 or ord == -2 or ord in (1, -1, _np.inf, -_np.inf, 0):
        return numel
    return 2 * numel


def vector_norm(x, ord=2, axis=None, keepdims=False):
    """Vector norm with FLOP counting."""
    budget = require_budget()
    validate_ndarray(x)
    if axis is not None:
        if isinstance(axis, int):
            effective_shape = (x.shape[axis],)
        else:
            effective_shape = tuple(x.shape[ax] for ax in axis)
    else:
        effective_shape = x.shape
    cost = vector_norm_cost(effective_shape, ord=ord)
    budget.deduct("linalg.vector_norm", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.vector_norm(x, ord=ord, axis=axis, keepdims=keepdims)


def matrix_norm_cost(shape: tuple, ord=None) -> int:
    """FLOP cost of matrix norm. Source: Direct analysis."""
    m, n = shape[-2], shape[-1]
    numel = m * n
    if ord is None or ord == "fro":
        return 2 * numel
    elif ord in (1, -1, _np.inf, -_np.inf):
        return numel
    elif ord == 2 or ord == -2:
        return m * n * min(m, n)
    elif ord == "nuc":
        return m * n * min(m, n)
    return numel


def matrix_norm(x, ord="fro", keepdims=False):
    """Matrix norm with FLOP counting."""
    budget = require_budget()
    validate_ndarray(x)
    cost = matrix_norm_cost(x.shape, ord=ord)
    budget.deduct("linalg.matrix_norm", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.matrix_norm(x, ord=ord, keepdims=keepdims)


def cond_cost(m: int, n: int) -> int:
    """FLOP cost of condition number. Formula: m * n * min(m, n). Source: SVD."""
    return max(m * n * min(m, n), 1)


def cond(x, p=None):
    """Condition number with FLOP counting."""
    budget = require_budget()
    validate_ndarray(x)
    if x.ndim != 2:
        raise ValueError(f"Input must be 2D, got {x.ndim}D")
    m, n = x.shape
    cost = cond_cost(m, n)
    budget.deduct("linalg.cond", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.cond(x, p=p)


def matrix_rank_cost(m: int, n: int) -> int:
    """FLOP cost of matrix rank. Formula: m * n * min(m, n). Source: SVD."""
    return max(m * n * min(m, n), 1)


def matrix_rank(A, tol=None, hermitian=False):
    """Matrix rank with FLOP counting."""
    budget = require_budget()
    validate_ndarray(A)
    if A.ndim != 2:
        raise ValueError(f"Input must be 2D, got {A.ndim}D")
    m, n = A.shape
    cost = matrix_rank_cost(m, n)
    budget.deduct("linalg.matrix_rank", flop_cost=cost, subscripts=None, shapes=(A.shape,))
    return _np.linalg.matrix_rank(A, tol=tol, hermitian=hermitian)
