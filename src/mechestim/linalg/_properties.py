# src/mechestim/linalg/_properties.py
"""Matrix property wrappers with FLOP counting."""

from __future__ import annotations

import numpy as _np

from mechestim._cost_model import FMA_COST
from mechestim._docstrings import attach_docstring
from mechestim._symmetric import SymmetricTensor
from mechestim._validation import require_budget


def trace_cost(n: int) -> int:
    """FLOP cost of matrix trace.

    Parameters
    ----------
    n : int
        Number of diagonal elements to sum.

    Returns
    -------
    int
        Estimated FLOP count: n.

    Notes
    -----
    Simply sums n diagonal elements.
    """
    return max(n, 1)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """Matrix trace with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    n = min(a.shape[axis1], a.shape[axis2])
    if offset > 0:
        n = min(n, a.shape[axis2] - offset)
    elif offset < 0:
        n = min(n, a.shape[axis1] + offset)
    n = max(n, 0)
    cost = trace_cost(n)
    budget.deduct("linalg.trace", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)


attach_docstring(trace, _np.trace, "linalg", r"$n$ FLOPs")


def det_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of determinant.

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
    Uses $n^3/3$ for symmetric input (Cholesky), or $n^3/3$ (FMA = 1 op) for general
    input (LU factorization).
    """
    if symmetric:
        return max(n**3 // 3, 1)
    return max(FMA_COST * n**3 // 3, 1)


def det(a):
    """Determinant with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = det_cost(n, symmetric=is_symmetric)
    budget.deduct("linalg.det", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.det(a)


attach_docstring(
    det,
    _np.linalg.det,
    "linalg",
    r"$n^3$ FLOPs (LU), or $n^3/3$ (Cholesky) for SymmetricTensor input",
)


def slogdet_cost(n: int, symmetric: bool = False) -> int:
    """FLOP cost of sign and log-determinant.

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
    Uses $n^3/3$ for symmetric input (Cholesky), or $n^3/3$ (FMA = 1 op) for general
    input (LU factorization).
    """
    if symmetric:
        return max(n**3 // 3, 1)
    return max(FMA_COST * n**3 // 3, 1)


def slogdet(a):
    """Sign and log-determinant with FLOP counting."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    n = a.shape[0]
    is_symmetric = isinstance(a, SymmetricTensor)
    cost = slogdet_cost(n, symmetric=is_symmetric)
    budget.deduct("linalg.slogdet", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.slogdet(a)


attach_docstring(
    slogdet,
    _np.linalg.slogdet,
    "linalg",
    r"$n^3$ FLOPs (LU), or $n^3/3$ (Cholesky) for SymmetricTensor input",
)


def norm_cost(shape: tuple, ord=None) -> int:
    """FLOP cost of matrix or vector norm.

    Parameters
    ----------
    shape : tuple of int
        Shape of the input array (or effective shape along norm axes).
    ord : {None, 'fro', 'nuc', inf, -inf, int}, optional
        Order of the norm.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    Cost depends on the ``ord`` parameter and input dimensionality.
    SVD-based norms (2-norm, nuclear norm) cost m * n * min(m, n).
    """
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
            return FMA_COST * numel
    else:
        m, n = shape[-2], shape[-1]
        if ord is None or ord == "fro":
            return FMA_COST * numel
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
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    if axis is None:
        effective_shape = x.shape
    elif isinstance(axis, int):
        effective_shape = (x.shape[axis],)
    else:
        effective_shape = tuple(x.shape[ax] for ax in axis)
    cost = norm_cost(effective_shape, ord=ord)
    budget.deduct("linalg.norm", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


attach_docstring(
    norm, _np.linalg.norm, "linalg", "depends on ord parameter -- see docstring"
)


def vector_norm_cost(shape: tuple, ord=None) -> int:
    """FLOP cost of vector norm.

    Parameters
    ----------
    shape : tuple of int
        Shape of the input array (or effective shape along norm axes).
    ord : {None, inf, -inf, int, float}, optional
        Order of the norm.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    Most norms cost n FLOPs (one pass over elements). General p-norms
    cost 2n due to exponentiation.
    """
    numel = 1
    for d in shape:
        numel *= d
    numel = max(numel, 1)
    if ord is None or ord == 2 or ord == -2 or ord in (1, -1, _np.inf, -_np.inf, 0):
        return numel
    return FMA_COST * numel


def vector_norm(x, ord=2, axis=None, keepdims=False):
    """Vector norm with FLOP counting."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    if axis is not None:
        if isinstance(axis, int):
            effective_shape = (x.shape[axis],)
        else:
            effective_shape = tuple(x.shape[ax] for ax in axis)
    else:
        effective_shape = x.shape
    cost = vector_norm_cost(effective_shape, ord=ord)
    budget.deduct(
        "linalg.vector_norm", flop_cost=cost, subscripts=None, shapes=(x.shape,)
    )
    return _np.linalg.vector_norm(x, ord=ord, axis=axis, keepdims=keepdims)


attach_docstring(
    vector_norm, _np.linalg.vector_norm, "linalg", "depends on ord parameter"
)


def matrix_norm_cost(shape: tuple, ord=None) -> int:
    """FLOP cost of matrix norm.

    Parameters
    ----------
    shape : tuple of int
        Shape of the input array (last two dims are the matrix).
    ord : {None, 'fro', 'nuc', inf, -inf, 1, -1, 2, -2}, optional
        Order of the norm.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    SVD-based norms (2-norm, nuclear norm) cost m * n * min(m, n).
    Frobenius norm costs 2mn. Entry-sum norms cost mn.
    """
    m, n = shape[-2], shape[-1]
    numel = m * n
    if ord is None or ord == "fro":
        return FMA_COST * numel
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
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    cost = matrix_norm_cost(x.shape, ord=ord)
    budget.deduct(
        "linalg.matrix_norm", flop_cost=cost, subscripts=None, shapes=(x.shape,)
    )
    return _np.linalg.matrix_norm(x, ord=ord, keepdims=keepdims)


attach_docstring(
    matrix_norm, _np.linalg.matrix_norm, "linalg", "depends on ord parameter"
)


def cond_cost(m: int, n: int, p=None) -> int:
    """FLOP cost of condition number.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    p : {None, 2, -2, 1, -1, inf, -inf}, optional
        Norm type. ``None`` and ``2``/``-2`` use SVD; ``1``/``-1``/``inf``/``-inf``
        use LU factorization, which is cheaper.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    For ``p=None``, ``p=2``, or ``p=-2``, computed via SVD (cost m*n*min(m,n)).
    For ``p=1``, ``p=-1``, ``p=inf``, or ``p=-inf``, computed via LU factorization
    (cost ~min(m,n)^3 + m*n for norm).
    """
    if p is None or p == 2 or p == -2:
        return max(m * n * min(m, n), 1)
    # LU-based: factorization cost + norm computation
    k = min(m, n)
    return max(k**3 + m * n, 1)


def cond(x, p=None):
    """Condition number with FLOP counting."""
    budget = require_budget()
    if not isinstance(x, _np.ndarray):
        x = _np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Input must be 2D, got {x.ndim}D")
    m, n = x.shape
    cost = cond_cost(m, n, p=p)
    budget.deduct("linalg.cond", flop_cost=cost, subscripts=None, shapes=(x.shape,))
    return _np.linalg.cond(x, p=p)


attach_docstring(
    cond,
    _np.linalg.cond,
    "linalg",
    r"$m \cdot n \cdot \min(m,n)$ FLOPs (SVD) or $\min(m,n)^3 + mn$ (LU) depending on p",
)


def matrix_rank_cost(m: int, n: int) -> int:
    """FLOP cost of matrix rank.

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


def matrix_rank(A, tol=None, hermitian=False):
    """Matrix rank with FLOP counting."""
    budget = require_budget()
    if not isinstance(A, _np.ndarray):
        A = _np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"Input must be 2D, got {A.ndim}D")
    m, n = A.shape
    cost = matrix_rank_cost(m, n)
    budget.deduct(
        "linalg.matrix_rank", flop_cost=cost, subscripts=None, shapes=(A.shape,)
    )
    return _np.linalg.matrix_rank(A, tol=tol, hermitian=hermitian)


attach_docstring(
    matrix_rank,
    _np.linalg.matrix_rank,
    "linalg",
    r"$m \cdot n \cdot \min(m,n)$ FLOPs (SVD)",
)
