"""mechestim — transparent proxy to a remote mechestim server.

This module exposes a numpy-like API where every operation is dispatched
to a remote server over ZMQ.  Participants use it as::

    import mechestim as me

    with me.BudgetContext(flop_budget=1_000_000) as ctx:
        a = me.array([[1.0, 2.0], [3.0, 4.0]])
        b = me.zeros((2, 2))
        c = me.add(a, b)
"""
from __future__ import annotations

import builtins
import math
import struct
from typing import Any

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

from mechestim.errors import (  # noqa: E402
    BudgetExhaustedError,
    MechEstimError,
    MechEstimServerError,
    MechEstimWarning,
    NoBudgetContextError,
    SymmetryError,
)

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

from mechestim._budget import BudgetContext, OpRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Remote types
# ---------------------------------------------------------------------------

from mechestim._remote_array import (  # noqa: E402
    RemoteArray,
    RemoteScalar,
    _result_from_response,
    _DTYPE_INFO,
)

# Alias: ``me.ndarray`` refers to the RemoteArray class.
ndarray = RemoteArray

# ---------------------------------------------------------------------------
# Connection / protocol (private)
# ---------------------------------------------------------------------------

from mechestim._connection import get_connection  # noqa: E402
from mechestim._protocol import (  # noqa: E402
    encode_request,
    encode_create_from_data,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from mechestim._registry import (  # noqa: E402
    FUNCTION_CATEGORIES,
    BLACKLISTED,
    is_valid_op,
    get_category,
    iter_proxyable,
)
from mechestim._registry_data import FUNCTION_CATEGORIES as _FC  # noqa: E402

# ---------------------------------------------------------------------------
# Submodules (imported so ``me.linalg``, ``me.random``, ``me.fft`` work)
# ---------------------------------------------------------------------------

from mechestim import fft  # noqa: E402, F401
from mechestim import flops  # noqa: E402, F401
from mechestim import linalg  # noqa: E402, F401
from mechestim import random  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Constants (no server round-trip needed)
# ---------------------------------------------------------------------------

pi: float = math.pi
e: float = math.e
inf: float = math.inf
nan: float = math.nan
newaxis = None

# ---------------------------------------------------------------------------
# Dtype strings (mirror numpy dtype names as plain strings)
# ---------------------------------------------------------------------------

float16: str = "float16"
float32: str = "float32"
float64: str = "float64"
int8: str = "int8"
int16: str = "int16"
int32: str = "int32"
int64: str = "int64"
uint8: str = "uint8"
bool_: str = "bool"
complex64: str = "complex64"
complex128: str = "complex128"

# ---------------------------------------------------------------------------
# Proxy factory
# ---------------------------------------------------------------------------


def _make_proxy(op_name: str):
    """Create a proxy function that dispatches *op_name* to the server."""

    def proxy(*args: Any, **kwargs: Any):
        conn = get_connection()
        encoded_args = []
        for arg in args:
            if isinstance(arg, RemoteArray):
                encoded_args.append({"__handle__": arg.handle_id})
            elif isinstance(arg, RemoteScalar):
                encoded_args.append(arg._value)
            else:
                encoded_args.append(arg)
        encoded_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, RemoteArray):
                encoded_kwargs[k] = {"__handle__": v.handle_id}
            elif isinstance(v, RemoteScalar):
                encoded_kwargs[k] = v._value
            else:
                encoded_kwargs[k] = v
        resp = conn.send_recv(encode_request(op_name, args=encoded_args, kwargs=encoded_kwargs))
        return _result_from_response(resp)

    proxy.__name__ = op_name
    proxy.__qualname__ = op_name
    return proxy


# ---------------------------------------------------------------------------
# Special-case: array()
# ---------------------------------------------------------------------------


def _flatten(obj):
    """Recursively flatten a nested list/tuple and return ``(flat, shape)``."""
    if not isinstance(obj, (list, tuple)):
        return [obj], ()
    if len(obj) == 0:
        return [], (0,)
    first_flat, inner_shape = _flatten(obj[0])
    flat = list(first_flat)
    for item in obj[1:]:
        item_flat, item_shape = _flatten(item)
        if item_shape != inner_shape:
            raise ValueError(
                f"Inhomogeneous shape: expected inner shape {inner_shape}, "
                f"got {item_shape}"
            )
        flat.extend(item_flat)
    return flat, (len(obj),) + inner_shape


def _infer_dtype(values):
    """Infer a dtype string from a list of Python scalars."""
    # Use builtins.any/all to avoid collision with the proxy functions
    # that shadow these names at module level.
    _any = builtins.any
    _all = builtins.all
    has_float = _any(isinstance(v, float) for v in values)
    has_complex = _any(isinstance(v, complex) for v in values)
    if has_complex:
        return "complex128"
    if has_float:
        return "float64"
    if _all(isinstance(v, bool) for v in values):
        return "bool"
    return "float64"  # default to float64 for int lists (numpy convention)


def array(object, dtype=None, **kwargs):
    """Create a remote array from a Python list, tuple, or existing RemoteArray.

    Parameters
    ----------
    object:
        Data to create the array from.  May be a nested list/tuple of
        numbers or an existing :class:`RemoteArray`.
    dtype:
        Optional dtype string (e.g. ``"float64"``).  Inferred from data
        if not given.

    Returns
    -------
    RemoteArray
        A new remote array on the server.
    """
    if isinstance(object, RemoteArray):
        if dtype is None:
            return object
        # dtype cast: dispatch to server
        conn = get_connection()
        resp = conn.send_recv(
            encode_request("astype", args=[{"__handle__": object.handle_id}, dtype])
        )
        return _result_from_response(resp)

    if isinstance(object, (list, tuple)):
        flat, shape = _flatten(object)
        if not flat:
            # Empty array
            dtype_str = dtype if isinstance(dtype, str) else "float64"
            conn = get_connection()
            resp = conn.send_recv(
                encode_create_from_data(b"", list(shape), dtype_str)
            )
            return _result_from_response(resp)

        dtype_str = dtype if isinstance(dtype, str) else (dtype or _infer_dtype(flat))
        info = _DTYPE_INFO.get(dtype_str)
        if info is None:
            raise TypeError(f"Unsupported dtype: {dtype_str!r}")
        fmt_char, _ = info
        data = struct.pack(f"<{len(flat)}{fmt_char}", *flat)

        conn = get_connection()
        resp = conn.send_recv(encode_create_from_data(data, list(shape), dtype_str))
        return _result_from_response(resp)

    if isinstance(object, (int, float)):
        # Scalar -> 0-d array
        dtype_str = dtype if isinstance(dtype, str) else "float64"
        info = _DTYPE_INFO.get(dtype_str)
        if info is None:
            raise TypeError(f"Unsupported dtype: {dtype_str!r}")
        fmt_char, _ = info
        data = struct.pack(f"<1{fmt_char}", object)
        conn = get_connection()
        resp = conn.send_recv(encode_create_from_data(data, [], dtype_str))
        return _result_from_response(resp)

    raise TypeError(
        f"Cannot create array from {type(object).__name__}. "
        f"Expected list, tuple, int, float, or RemoteArray."
    )


# ---------------------------------------------------------------------------
# Special-case: einsum()
# ---------------------------------------------------------------------------


def einsum(subscripts, *operands, **kwargs):
    """Einstein summation on remote arrays.

    Parameters
    ----------
    subscripts:
        Subscript string (e.g. ``"ij,jk->ik"``).
    *operands:
        Input :class:`RemoteArray` objects.
    **kwargs:
        Additional keyword arguments forwarded to the server.

    Returns
    -------
    RemoteArray
        Result of the einsum operation.
    """
    conn = get_connection()
    encoded_args = [subscripts]
    for op in operands:
        if isinstance(op, RemoteArray):
            encoded_args.append({"__handle__": op.handle_id})
        elif isinstance(op, RemoteScalar):
            encoded_args.append(op._value)
        else:
            encoded_args.append(op)
    resp = conn.send_recv(encode_request("einsum", args=encoded_args, kwargs=kwargs))
    return _result_from_response(resp)


# ---------------------------------------------------------------------------
# Auto-generate proxy functions for all non-blacklisted top-level ops
# ---------------------------------------------------------------------------

# Functions that are special-cased above and should not be overwritten.
_SPECIAL_CASED = frozenset({"array", "einsum"})

# Functions that belong to submodules (contain a dot) are handled by the
# submodule packages themselves.
_generated_proxies: list[str] = []
for _op_name in iter_proxyable():
    if "." in _op_name:
        continue  # submodule function
    if _op_name in _SPECIAL_CASED:
        continue
    globals()[_op_name] = _make_proxy(_op_name)
    _generated_proxies.append(_op_name)

del _op_name  # clean up loop variable


# ---------------------------------------------------------------------------
# Module-level __getattr__ for blacklisted / unknown names
# ---------------------------------------------------------------------------

# We import the factory but define the function inline so we can also
# check against names that are already defined in the module namespace.

from mechestim._getattr import make_module_getattr as _make_module_getattr  # noqa: E402

_module_getattr = _make_module_getattr("", "mechestim")


def __getattr__(name: str):
    return _module_getattr(name)
