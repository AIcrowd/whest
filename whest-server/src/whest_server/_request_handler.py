"""RequestHandler — dispatches decoded request dicts to whest functions."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

import whest as me
from whest._perm_group import SymmetryGroup, _Permutation
from whest_server._session import Session

_HANDLE_RE = re.compile(r"^a\d+$")


def _make_serializable(obj):
    """Convert a nested structure to be msgpack-safe (no numpy types)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    return obj


#: Maximum allowed array size in bytes (configurable via environment variable).
MAX_ARRAY_BYTES = int(os.environ.get("WHEST_MAX_ARRAY_BYTES", 100 * 1024 * 1024))


class RequestHandler:
    """Dispatch decoded request dicts to real whest functions.

    Parameters
    ----------
    session : Session
        The active session providing array storage, budget context, etc.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle(self, request: dict) -> dict:
        """Dispatch *request* and return a response dict.

        The ``request["op"]`` field determines which handler is invoked.
        """
        try:
            op = request["op"]

            if op == "fetch":
                return self._handle_fetch(request)
            if op == "fetch_slice":
                return self._handle_fetch_slice(request)
            if op == "free":
                return self._handle_free(request)
            if op == "budget_status":
                return self._handle_budget_status()
            if op == "create_from_data":
                return self._handle_create_from_data(request)
            if op == "__getitem__":
                return self._handle_getitem(request)

            # Any other op — whest function call
            return self._handle_whest_op(request)

        except me.BudgetExhaustedError as e:
            return {
                "status": "error",
                "error_type": "BudgetExhaustedError",
                "message": str(e),
            }
        except me.NoBudgetContextError as e:
            return {
                "status": "error",
                "error_type": "NoBudgetContextError",
                "message": str(e),
            }
        except me.SymmetryError as e:
            return {"status": "error", "error_type": "SymmetryError", "message": str(e)}
        except me.UnsupportedFunctionError as e:
            return {
                "status": "error",
                "error_type": "UnsupportedFunctionError",
                "message": str(e),
            }
        except (ValueError, TypeError) as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e),
            }
        except KeyError as e:
            return {"status": "error", "error_type": "KeyError", "message": str(e)}
        except Exception as e:
            return {
                "status": "error",
                "error_type": "WhestServerError",
                "message": f"internal server error: {type(e).__name__}: {e}",
            }

    # ------------------------------------------------------------------
    # Built-in ops
    # ------------------------------------------------------------------

    def _handle_fetch(self, request: dict) -> dict:
        # Support both direct "id" field and kwargs-based "handle_id"
        handle = request.get("id")
        if handle is None:
            kwargs = request.get("kwargs") or {}
            handle = kwargs.get("handle_id")
        if handle is None:
            raise KeyError("fetch requires 'id' or kwargs.handle_id")
        arr = self._session.get_array(handle)
        return {
            "status": "ok",
            "data": arr.tobytes(),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    def _handle_fetch_slice(self, request: dict) -> dict:
        arr = self._session.get_array(request["id"])
        slices = tuple(slice(*s) for s in request["slices"])
        sliced = arr[slices]

        if np.ndim(sliced) == 0:
            return {
                "status": "ok",
                "data": sliced.tobytes(),
                "shape": [],
                "dtype": str(sliced.dtype),
            }

        return {
            "status": "ok",
            "data": sliced.tobytes(),
            "shape": list(sliced.shape),
            "dtype": str(sliced.dtype),
        }

    def _handle_free(self, request: dict) -> dict:
        # Support both direct "ids" field and kwargs-based "handles"
        ids = request.get("ids")
        if ids is None:
            kwargs = request.get("kwargs") or {}
            ids = kwargs.get("handles", [])
        self._session.free_arrays(ids)
        return {"status": "ok"}

    def _handle_budget_status(self) -> dict:
        return {"status": "ok", "result": self._session.budget_status()}

    def _handle_create_from_data(self, request: dict) -> dict:
        # Support both direct fields and args-based [data, shape, dtype]
        if "data" in request:
            data = request["data"]
            shape = request["shape"]
            dtype = request["dtype"]
        else:
            args = request.get("args", [])
            if len(args) >= 3:
                data, shape, dtype = args[0], args[1], args[2]
            else:
                raise ValueError("create_from_data requires data, shape, dtype")
        # Ensure dtype is a string (may be bytes from msgpack)
        if isinstance(dtype, bytes):
            dtype = dtype.decode("utf-8")
        if len(data) > MAX_ARRAY_BYTES:
            return {
                "status": "error",
                "error_type": "ValueError",
                "message": f"array too large: {len(data)} bytes exceeds {MAX_ARRAY_BYTES} byte limit",
            }
        arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        handle = self._session.store_array(arr)
        meta = self._session.array_metadata(handle)
        return {
            "status": "ok",
            "result": meta,
            "budget": self._session.budget_status(),
        }

    # ------------------------------------------------------------------
    # __getitem__ dispatch
    # ------------------------------------------------------------------

    def _handle_getitem(self, request: dict) -> dict:
        """Handle array indexing: arr[key] on the server side."""
        args = request.get("args") or []
        if len(args) < 2:
            raise ValueError("__getitem__ requires [handle, key]")
        arr = self._resolve_arg(args[0])
        key = self._decode_index_key(args[1])
        result = arr[key]
        return self._pack_result(result)

    # ------------------------------------------------------------------
    # Whest function dispatch
    # ------------------------------------------------------------------

    def _handle_whest_op(self, request: dict) -> dict:
        op = request["op"]
        raw_args = request.get("args") or []
        kwargs = request.get("kwargs") or {}

        # Special-case: astype is an ndarray method, not a module function
        if op == "astype":
            arr = self._resolve_arg(raw_args[0])
            dtype = raw_args[1] if len(raw_args) > 1 else kwargs.get("dtype")
            if isinstance(dtype, bytes):
                dtype = dtype.decode("utf-8")
            result = arr.astype(dtype)
            return self._pack_result(result)

        func = _get_whest_func(op)
        resolved_args = [self._resolve_arg(a) for a in raw_args]
        resolved_kwargs = {k: self._resolve_arg(v) for k, v in kwargs.items()}

        result = func(*resolved_args, **resolved_kwargs)

        return self._pack_result(result)

    # ------------------------------------------------------------------
    # Argument resolution
    # ------------------------------------------------------------------

    def _resolve_arg(self, arg: Any) -> Any:
        """Resolve a single argument: handle IDs become arrays, rest pass through."""
        if isinstance(arg, str) and _HANDLE_RE.match(arg):
            return self._session.get_array(arg)
        # Support {"__handle__": "a0"} dict format from the client
        if isinstance(arg, dict):
            handle = arg.get("__handle__")
            if handle is None:
                # Try bytes key variant (msgpack may leave keys as bytes)
                handle = arg.get(b"__handle__")
            if handle is not None:
                if isinstance(handle, bytes):
                    handle = handle.decode("utf-8")
                return self._session.get_array(handle)
            # Permutation wire format
            perm_data = arg.get("__permutation__") or arg.get(b"__permutation__")
            if perm_data is not None:
                return _Permutation(list(perm_data))
            # SymmetryGroup wire format
            pg_data = arg.get("__symmetry_group__") or arg.get(b"__symmetry_group__")
            if pg_data is not None:
                if isinstance(pg_data, dict):
                    pg_data = {
                        (k.decode("utf-8") if isinstance(k, bytes) else k): v
                        for k, v in pg_data.items()
                    }
                return SymmetryGroup.from_payload(pg_data)
        # Recurse into lists/tuples so that e.g. concatenate([a, b]) works
        if isinstance(arg, (list, tuple)):
            resolved = [self._resolve_arg(item) for item in arg]
            return type(arg)(resolved) if isinstance(arg, tuple) else resolved
        return arg

    # ------------------------------------------------------------------
    # Result packing
    # ------------------------------------------------------------------

    def _decode_index_key(self, raw_key):
        """Decode a serialised index key from the client (instance method).

        Supports handle dicts for fancy indexing with RemoteArrays.
        """
        if isinstance(raw_key, dict):
            # Handle dict: {"__handle__": "a0"} for fancy indexing
            handle = raw_key.get("__handle__") or raw_key.get(b"__handle__")
            if handle is not None:
                if isinstance(handle, bytes):
                    handle = handle.decode()
                return self._session.get_array(handle)
            if "__slice__" in raw_key:
                parts = raw_key["__slice__"]
                return slice(*[None if p is None else int(p) for p in parts])
            if b"__slice__" in raw_key:
                parts = raw_key[b"__slice__"]
                return slice(*[None if p is None else int(p) for p in parts])
        if isinstance(raw_key, list):
            decoded = [self._decode_index_key(item) for item in raw_key]
            if any(isinstance(d, slice) for d in decoded) or len(decoded) > 1:
                return tuple(decoded)
            return decoded
        if isinstance(raw_key, (int, float)):
            return int(raw_key)
        return raw_key

    # ------------------------------------------------------------------
    # Result packing
    # ------------------------------------------------------------------

    def _pack_result(self, result: Any) -> dict:
        """Pack a whest function result into a response dict."""
        budget = self._session.budget_status()

        if isinstance(result, np.ndarray):
            if result.nbytes > MAX_ARRAY_BYTES:
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "message": f"result array too large: {result.nbytes} bytes exceeds {MAX_ARRAY_BYTES} byte limit",
                }
            handle = self._session.store_array(result)
            meta = self._session.array_metadata(handle)
            return {"status": "ok", "result": meta, "budget": budget}

        if isinstance(result, (tuple, list)):
            items = []
            for r in result:
                if isinstance(r, np.ndarray):
                    handle = self._session.store_array(r)
                    items.append(self._session.array_metadata(handle))
                elif isinstance(r, np.generic):
                    items.append({"value": r.item(), "dtype": str(r.dtype)})
                elif isinstance(r, (int, float)):
                    dtype_str = "float64" if isinstance(r, float) else "int64"
                    items.append({"value": r, "dtype": dtype_str})
                elif isinstance(r, str):
                    items.append({"value": r, "dtype": "str"})
                elif isinstance(r, (list, tuple)):
                    # Nested list/tuple (e.g., from einsum_path) — convert to JSON-safe
                    items.append({"value": _make_serializable(r), "dtype": "object"})
                else:
                    items.append({"value": r})
            return {"status": "ok", "result": {"multi": items}, "budget": budget}

        # Scalar or other value
        if isinstance(result, np.generic):
            dtype_str = str(result.dtype)
            return {
                "status": "ok",
                "result": {"value": result.item(), "dtype": dtype_str},
                "budget": budget,
            }
        if isinstance(result, bool):
            return {
                "status": "ok",
                "result": {"value": result, "dtype": "bool"},
                "budget": budget,
            }
        if isinstance(result, int):
            return {
                "status": "ok",
                "result": {"value": result, "dtype": "int64"},
                "budget": budget,
            }
        if isinstance(result, float):
            return {
                "status": "ok",
                "result": {"value": result, "dtype": "float64"},
                "budget": budget,
            }
        if isinstance(result, str):
            return {
                "status": "ok",
                "result": {"value": result, "dtype": "str"},
                "budget": budget,
            }
        if isinstance(result, np.dtype):
            return {
                "status": "ok",
                "result": {"value": str(result), "dtype": "str"},
                "budget": budget,
            }
        # Fallback: try to make it serializable
        try:
            serializable = _make_serializable(result)
            return {"status": "ok", "result": {"value": serializable}, "budget": budget}
        except Exception:
            return {
                "status": "ok",
                "result": {"value": str(result), "dtype": "str"},
                "budget": budget,
            }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_whest_func(op_name: str):
    """Look up a whest function by dotted name (e.g. 'linalg.svd')."""
    parts = op_name.split(".")
    obj = me
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _decode_index_key(raw_key):
    """Decode a serialised index key from the client.

    Supports:
    - int / float -> int
    - ``{"__slice__": [start, stop, step]}`` -> slice
    - list of the above -> tuple (for multi-dimensional indexing)
    """
    if isinstance(raw_key, dict):
        if "__slice__" in raw_key:
            parts = raw_key["__slice__"]
            return slice(*[None if p is None else int(p) for p in parts])
    if isinstance(raw_key, list):
        decoded = [_decode_index_key(item) for item in raw_key]
        # A list of slices/ints -> tuple for multi-dim indexing
        if any(isinstance(d, slice) for d in decoded) or len(decoded) > 1:
            return tuple(decoded)
        # Single-element list: could be the key itself being a list
        # (e.g., fancy indexing) -- keep as list
        return decoded
    if isinstance(raw_key, (int, float)):
        return int(raw_key)
    return raw_key
