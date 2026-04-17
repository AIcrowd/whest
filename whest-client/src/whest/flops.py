"""FLOP cost estimation utilities.

Some cost functions can be computed locally (pure arithmetic); others
proxy to the server for more complex estimations.
"""

from __future__ import annotations

from collections.abc import Sequence

from whest._math_compat import prod as _prod

# ---------------------------------------------------------------------------
# Local cost functions (no server needed)
# ---------------------------------------------------------------------------


def pointwise_cost(shape: tuple[int, ...]) -> int:
    """Return the FLOP cost of a pointwise (element-wise) operation.

    Parameters
    ----------
    shape:
        Shape of the array the operation is applied to.

    Returns
    -------
    int
        Number of elements (``_prod(shape)``), which equals the number
        of FLOPs for a single pointwise operation.
    """
    return max(_prod(shape), 1)


def reduction_cost(input_shape: tuple[int, ...], axis: int | None = None) -> int:
    """Return the FLOP cost of a reduction operation.

    Parameters
    ----------
    input_shape:
        Shape of the input array.
    axis:
        Axis along which the reduction is performed.  ``None`` means
        reduce over all elements.

    Returns
    -------
    int
        Number of FLOPs for the reduction.
    """
    total = max(_prod(input_shape), 1)
    if axis is None:
        return total
    # Reduction along a single axis: cost is the total element count
    # (each element participates once).
    return total


# ---------------------------------------------------------------------------
# Server-proxied cost functions
# ---------------------------------------------------------------------------


def einsum_cost(subscripts: str, shapes: Sequence[tuple[int, ...]]) -> int:
    """Query the server for the FLOP cost of an einsum operation.

    Parameters
    ----------
    subscripts:
        Einstein summation subscript string.
    shapes:
        Shapes of the input arrays.

    Returns
    -------
    int
        Estimated FLOP cost.
    """
    from whest._connection import get_connection
    from whest._protocol import encode_request

    conn = get_connection()
    resp = conn.send_recv(
        encode_request(
            "flops.einsum_cost",
            kwargs={"subscripts": subscripts, "shapes": [list(s) for s in shapes]},
        )
    )
    result = resp.get("result", {})
    return int(result.get("value", 0))


def svd_cost(m: int, n: int, k: int = 0) -> int:
    """Query the server for the FLOP cost of an SVD operation.

    Parameters
    ----------
    m:
        Number of rows.
    n:
        Number of columns.
    k:
        Number of singular values to compute (0 means all).

    Returns
    -------
    int
        Estimated FLOP cost.
    """
    from whest._connection import get_connection
    from whest._protocol import encode_request

    conn = get_connection()
    resp = conn.send_recv(
        encode_request(
            "flops.svd_cost",
            kwargs={"m": m, "n": n, "k": k},
        )
    )
    result = resp.get("result", {})
    return int(result.get("value", 0))
