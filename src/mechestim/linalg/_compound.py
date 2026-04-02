"""Compound linalg operations with FLOP counting."""
from __future__ import annotations
import math
import numpy as _np
from mechestim._docstrings import attach_docstring
from mechestim._validation import require_budget, validate_ndarray


def _popcount(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def multi_dot_cost(shapes: list[tuple[int, ...]]) -> int:
    """FLOP cost of optimal matrix chain multiplication.

    Parameters
    ----------
    shapes : list of tuple of int
        Shapes of the matrices to be multiplied.

    Returns
    -------
    int
        Estimated FLOP count using optimal parenthesization.

    Notes
    -----
    Uses dynamic programming for optimal parenthesization.
    Source: Cormen et al., *Introduction to Algorithms* (CLRS), §15.2.
    """
    n = len(shapes)
    if n < 2:
        return 0
    dims = [s[0] for s in shapes] + [shapes[-1][-1]]
    if n == 2:
        return dims[0] * dims[1] * dims[2]
    cost_table = [[0] * n for _ in range(n)]
    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            cost_table[i][j] = float("inf")
            for k in range(i, j):
                cost = cost_table[i][k] + cost_table[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < cost_table[i][j]:
                    cost_table[i][j] = cost
    return max(int(cost_table[0][n - 1]), 1)


def multi_dot(arrays, *, out=None):
    """Efficient multi-matrix dot product with FLOP counting."""
    budget = require_budget()
    for arr in arrays:
        validate_ndarray(arr)
    shapes = [arr.shape for arr in arrays]
    cost = multi_dot_cost(shapes)
    budget.deduct("linalg.multi_dot", flop_cost=cost, subscripts=None, shapes=tuple(shapes))
    return _np.linalg.multi_dot(arrays, out=out)

attach_docstring(multi_dot, _np.linalg.multi_dot, "linalg", "Optimal chain multiplication cost")


def matrix_power_cost(n: int, k: int) -> int:
    """FLOP cost of matrix power A**k.

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Exponent.

    Returns
    -------
    int
        Estimated FLOP count.

    Notes
    -----
    Uses exponentiation by repeated squaring. For $k < 0$, adds $n^3$ for
    the initial matrix inversion.
    """
    if k == 0 or k == 1:
        return 0
    if k < 0:
        return n ** 3 + matrix_power_cost(n, abs(k))
    num_ops = math.floor(math.log2(k)) + _popcount(k) - 1
    return max(num_ops * n ** 3, 1)


def matrix_power(a, n):
    """Matrix power with FLOP counting."""
    budget = require_budget()
    validate_ndarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be square 2D array, got shape {a.shape}")
    size = a.shape[0]
    cost = matrix_power_cost(size, n)
    budget.deduct("linalg.matrix_power", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.matrix_power(a, n)

attach_docstring(matrix_power, _np.linalg.matrix_power, "linalg",
    r"$n^3 \times \text{exponent}$ FLOPs (repeated squaring)")
