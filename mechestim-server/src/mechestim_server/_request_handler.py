"""RequestHandler — dispatches decoded request dicts to mechestim functions."""

from __future__ import annotations

import os
import re
from typing import Any

import mechestim as me
import numpy as np

from mechestim_server._session import Session

_HANDLE_RE = re.compile(r"^a\d+$")

#: Maximum allowed array size in bytes (configurable via environment variable).
MAX_ARRAY_BYTES = int(os.environ.get("MECHESTIM_MAX_ARRAY_BYTES", 100 * 1024 * 1024))


class RequestHandler:
    """Dispatch decoded request dicts to real mechestim functions.

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

            # Any other op — mechestim function call
            return self._handle_mechestim_op(request)

        except me.BudgetExhaustedError as e:
            return {"status": "error", "error_type": "BudgetExhaustedError", "message": str(e)}
        except me.NoBudgetContextError as e:
            return {"status": "error", "error_type": "NoBudgetContextError", "message": str(e)}
        except me.SymmetryError as e:
            return {"status": "error", "error_type": "SymmetryError", "message": str(e)}
        except (ValueError, TypeError) as e:
            return {"status": "error", "error_type": type(e).__name__, "message": str(e)}
        except KeyError as e:
            return {"status": "error", "error_type": "KeyError", "message": str(e)}
        except Exception as e:
            return {
                "status": "error",
                "error_type": "MechEstimServerError",
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
            return {"status": "ok", "value": sliced.item()}

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
    # Mechestim function dispatch
    # ------------------------------------------------------------------

    def _handle_mechestim_op(self, request: dict) -> dict:
        op = request["op"]
        raw_args = request.get("args") or []
        kwargs = request.get("kwargs") or {}

        func = _get_mechestim_func(op)
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
        # Recurse into lists/tuples so that e.g. concatenate([a, b]) works
        if isinstance(arg, (list, tuple)):
            resolved = [self._resolve_arg(item) for item in arg]
            return type(arg)(resolved) if isinstance(arg, tuple) else resolved
        return arg

    # ------------------------------------------------------------------
    # Result packing
    # ------------------------------------------------------------------

    def _pack_result(self, result: Any) -> dict:
        """Pack a mechestim function result into a response dict."""
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

        if isinstance(result, tuple):
            items = []
            for r in result:
                if isinstance(r, np.ndarray):
                    handle = self._session.store_array(r)
                    items.append(self._session.array_metadata(handle))
                elif isinstance(r, (np.generic, int, float)):
                    items.append({"value": float(r) if isinstance(r, (np.floating, float)) else int(r)})
                else:
                    items.append(r)
            return {"status": "ok", "result": {"multi": items}, "budget": budget}

        # Scalar or other value
        if isinstance(result, (np.generic,)):
            result = result.item()
        return {"status": "ok", "result": {"value": result}, "budget": budget}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_mechestim_func(op_name: str):
    """Look up a mechestim function by dotted name (e.g. 'linalg.svd')."""
    parts = op_name.split(".")
    obj = me
    for part in parts:
        obj = getattr(obj, part)
    return obj
