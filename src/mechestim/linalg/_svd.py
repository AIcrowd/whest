"""Truncated SVD with FLOP counting."""
from __future__ import annotations
import numpy as _np
from mechestim._flops import svd_cost
from mechestim._validation import check_nan_inf, require_budget, validate_ndarray


def svd(a: _np.ndarray, k: int | None = None) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """Truncated singular value decomposition.

    FLOP cost: m * n * k where a is (m, n). If k is None, k = min(m, n).

    Returns (U, S, Vt):
    - U: (m, k) left singular vectors
    - S: (k,) singular values descending
    - Vt: (k, n) right singular vectors transposed
    """
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2:
        raise ValueError(f"Input must be 2D, got {a.ndim}D")
    m, n = a.shape
    if k is None:
        k = min(m, n)
    if not (1 <= k <= min(m, n)):
        raise ValueError(f"k must satisfy 1 <= k <= min(m, n) = {min(m, n)}, got k={k}")
    cost = svd_cost(m, n, k)
    budget.deduct("linalg.svd", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    U, S, Vt = _np.linalg.svd(a, full_matrices=False)
    U = U[:, :k]
    S = S[:k]
    Vt = Vt[:k, :]
    check_nan_inf(S, "linalg.svd")
    return U, S, Vt
