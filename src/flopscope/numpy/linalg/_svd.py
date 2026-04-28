"""SVD with FLOP counting."""

from __future__ import annotations

import numpy as _np
from numpy.linalg._linalg import SVDResult

from flopscope._flops import svd_cost
from flopscope._validation import check_nan_inf, require_budget


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


def svd(a, full_matrices=True, compute_uv=True, hermitian=False, *, k=None):
    """Singular value decomposition with FLOP counting.

    Matches ``numpy.linalg.svd`` signature with an optional *k* parameter
    for truncated SVD (flopscope extension).

    FLOP Cost
    ---------
    m * n * k FLOPs per matrix, where k defaults to min(m, n).

    Parameters
    ----------
    a : array_like
        Input array with shape ``(..., m, n)``.
    full_matrices : bool, optional
        If True (default), U and Vt have shapes ``(..., m, m)`` and
        ``(..., n, n)``. If False, shapes are ``(..., m, K)`` and
        ``(..., K, n)`` where K = min(m, n) or k if specified.
    compute_uv : bool, optional
        If True (default), compute U and Vt. If False, only compute
        singular values.
    hermitian : bool, optional
        If True, a is assumed to be Hermitian. Default is False.
    k : int, optional
        **flopscope extension.** Number of singular values/vectors to return.
        Must satisfy 1 <= k <= min(m, n). If None (default), returns
        all min(m, n) components.

    Returns
    -------
    When compute_uv is True (default):
        U : ndarray, S : ndarray, Vt : ndarray
    When compute_uv is False:
        S : ndarray
    """
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim < 2:
        raise ValueError(f"svd requires a 2D (or batched) array, got shape {a.shape}")
    m, n = a.shape[-2], a.shape[-1]
    if k is not None and k > min(m, n):
        raise ValueError(
            f"k={k} exceeds min(m, n)={min(m, n)} for array shape {a.shape}"
        )
    effective_k = k if k is not None else min(m, n)
    batch = _batch_size(a.shape)
    cost = svd_cost(m, n, effective_k) * batch if not _has_zero_dim(a.shape) else 0
    with budget.deduct(
        "linalg.svd", flop_cost=cost, subscripts=None, shapes=(a.shape,)
    ):
        if compute_uv:
            # When k is specified, always use economy decomposition then slice.
            fm = full_matrices if k is None else False
            U, S, Vt = _np.linalg.svd(a, full_matrices=fm, hermitian=hermitian)
            if k is not None:
                S = S[..., :k]
                U = U[..., :k]
                Vt = Vt[..., :k, :]
            check_nan_inf(S, "linalg.svd")
        else:
            S = _np.linalg.svd(a, compute_uv=False, hermitian=hermitian)
            if k is not None:
                S = S[..., :k]
            check_nan_inf(S, "linalg.svd")

    if compute_uv:
        return SVDResult(U, S, Vt)
    else:
        return S
