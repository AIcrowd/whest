"""Truncated SVD with FLOP counting."""

from __future__ import annotations

import numpy as _np

from whest._flops import svd_cost
from whest._validation import check_nan_inf, require_budget


def svd(
    a: _np.ndarray, k: int | None = None
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """Truncated singular value decomposition.

    Computes the top-*k* singular values and corresponding vectors
    of a 2-D matrix, wrapping ``numpy.linalg.svd`` with FLOP counting.

    FLOP Cost
    ---------
    m × n × k FLOPs.

    Parameters
    ----------
    a : numpy.ndarray
        Input matrix of shape ``(m, n)``.
    k : int, optional
        Number of singular values/vectors to return. Must satisfy
        ``1 <= k <= min(m, n)``. If ``None``, returns all ``min(m, n)``
        components.

    Returns
    -------
    U : numpy.ndarray
        Left singular vectors, shape ``(m, k)``.
    S : numpy.ndarray
        Singular values in descending order, shape ``(k,)``.
    Vt : numpy.ndarray
        Right singular vectors (transposed), shape ``(k, n)``.

    Raises
    ------
    BudgetExhaustedError
        If the operation would exceed the FLOP budget.
    ValueError
        If *a* is not 2-D or *k* is out of range.

    Notes
    -----
    **whest cost:** m × n × k FLOPs.

    See Also
    --------
    numpy.linalg.svd : Full NumPy SVD documentation.
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"Input must be 2D, got {a.ndim}D")
    m, n = a.shape
    if k is None:
        k = min(m, n)
    if not (1 <= k <= min(m, n)):
        raise ValueError(f"k must satisfy 1 <= k <= min(m, n) = {min(m, n)}, got k={k}")
    cost = svd_cost(m, n, k)
    with budget.deduct("linalg.svd", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        U, S, Vt = _np.linalg.svd(a, full_matrices=False)
    U = U[:, :k]
    S = S[:k]
    Vt = Vt[:k, :]
    check_nan_inf(S, "linalg.svd")
    return U, S, Vt
