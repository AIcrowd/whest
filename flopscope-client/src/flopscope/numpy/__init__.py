"""flopscope.numpy — client-side counted-numpy drop-in (JAX-style).

Every numpy-shaped operation in this module is a thin proxy that dispatches
the call to a remote flopscope server.  The top-level :mod:`flopscope`
client module only exposes flopscope-specific primitives
(``BudgetContext``, ``configure``, ``SymmetricTensor``, …); the numpy-shaped
surface lives here so the client mirrors the core library's layout::

    import flopscope as flops
    import flopscope.numpy as fnp

    with flops.BudgetContext(flop_budget=1_000_000):
        a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = fnp.zeros((2, 2))
        c = fnp.add(a, b)
"""

from __future__ import annotations

import builtins
import struct
from typing import Any

from flopscope._connection import get_connection
from flopscope._math_compat import e, inf, nan, pi  # noqa: F401 (re-export)
from flopscope._protocol import encode_create_from_data, encode_request
from flopscope._registry import iter_proxyable
from flopscope._remote_array import (
    _DTYPE_INFO,
    RemoteArray,
    _encode_arg,
    _result_from_response,
)

# Alias: ``fnp.ndarray`` -> RemoteArray (matches flopscope core's pattern
# where fnp.ndarray aliases FlopscopeArray).
ndarray = RemoteArray

# --- Dtype strings (mirror numpy dtype names as plain strings) ---
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

newaxis = None


def _make_proxy(op_name: str):
    """Create a proxy function that dispatches *op_name* to the server."""

    def proxy(*args: Any, **kwargs: Any):
        conn = get_connection()
        encoded_args = [_encode_arg(a) for a in args]
        encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
        resp = conn.send_recv(
            encode_request(op_name, args=encoded_args, kwargs=encoded_kwargs)
        )
        return _result_from_response(resp)

    proxy.__name__ = op_name
    proxy.__qualname__ = op_name
    return proxy


# ---------------------------------------------------------------------------
# Special-cased constructors and ops
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
    if _all(isinstance(v, int) for v in values):
        return "int64"
    return "float64"


def array(object, dtype=None, **kwargs):
    """Create a remote array from a Python list, tuple, or existing RemoteArray."""
    if isinstance(object, RemoteArray):
        if dtype is None:
            return object
        conn = get_connection()
        resp = conn.send_recv(
            encode_request("astype", args=[{"__handle__": object.handle_id}, dtype])
        )
        return _result_from_response(resp)

    if isinstance(object, (list, tuple)):
        flat, shape = _flatten(object)
        if not flat:
            dtype_str = dtype if isinstance(dtype, str) else "float64"
            conn = get_connection()
            resp = conn.send_recv(encode_create_from_data(b"", list(shape), dtype_str))
            return _result_from_response(resp)

        dtype_str = dtype if isinstance(dtype, str) else (dtype or _infer_dtype(flat))
        info = _DTYPE_INFO.get(dtype_str)
        if info is None:
            raise TypeError(f"Unsupported dtype: {dtype_str!r}")
        fmt_char, _ = info

        if dtype_str in ("complex64", "complex128"):
            expanded = []
            for v in flat:
                c = complex(v)
                expanded.extend([c.real, c.imag])
            flat = expanded
            fmt_char = "f" if dtype_str == "complex64" else "d"
            data = struct.pack(f"<{len(flat)}{fmt_char}", *flat)
        else:
            data = struct.pack(f"<{len(flat)}{fmt_char}", *flat)

        conn = get_connection()
        resp = conn.send_recv(encode_create_from_data(data, list(shape), dtype_str))
        return _result_from_response(resp)

    if isinstance(object, (int, float, complex)):
        if isinstance(object, complex) and dtype is None:
            dtype_str = "complex128"
        else:
            dtype_str = dtype if isinstance(dtype, str) else "float64"
        info = _DTYPE_INFO.get(dtype_str)
        if info is None:
            raise TypeError(f"Unsupported dtype: {dtype_str!r}")
        fmt_char, _ = info

        if dtype_str in ("complex64", "complex128"):
            c = complex(object)
            pack_fmt = "f" if dtype_str == "complex64" else "d"
            data = struct.pack(f"<2{pack_fmt}", c.real, c.imag)
        else:
            data = struct.pack(f"<1{fmt_char}", object)
        conn = get_connection()
        resp = conn.send_recv(encode_create_from_data(data, [], dtype_str))
        return _result_from_response(resp)

    raise TypeError(
        f"Cannot create array from {type(object).__name__}. "
        f"Expected list, tuple, int, float, or RemoteArray."
    )


def einsum(subscripts, *operands, **kwargs):
    """Einstein summation on remote arrays."""
    conn = get_connection()
    encoded_args = [subscripts] + [_encode_arg(op) for op in operands]
    encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
    resp = conn.send_recv(
        encode_request("einsum", args=encoded_args, kwargs=encoded_kwargs)
    )
    return _result_from_response(resp)


# ---------------------------------------------------------------------------
# Auto-generate proxies for non-blacklisted top-level ops
# ---------------------------------------------------------------------------

_SPECIAL_CASED = frozenset({"array", "einsum"})
_generated_proxies: list[str] = []
for _op_name in iter_proxyable():
    if "." in _op_name:
        continue
    if _op_name in _SPECIAL_CASED:
        continue
    globals()[_op_name] = _make_proxy(_op_name)
    _generated_proxies.append(_op_name)

del _op_name

# ---------------------------------------------------------------------------
# Submodules: linalg, fft, random
# ---------------------------------------------------------------------------

from flopscope.numpy import fft, linalg, random  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Strict __getattr__: no transparent fallback to raw numpy. Unsupported
# names raise AttributeError so participants notice immediately rather than
# silently dispatch to an uncounted numpy operation.
# ---------------------------------------------------------------------------

from flopscope._getattr import make_module_getattr as _make_module_getattr  # noqa: E402

__getattr__ = _make_module_getattr("", "flopscope.numpy")
