"""RequestHandler — dispatches decoded request dicts to mechestim functions."""

from __future__ import annotations

import re
from typing import Any

import mechestim as me
import numpy as np

from mechestim_server._session import Session

_HANDLE_RE = re.compile(r"^a\d+$")


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
                "message": f"internal server error: {type(e).__name__}",
            }

    # ------------------------------------------------------------------
    # Built-in ops
    # ------------------------------------------------------------------

    def _handle_fetch(self, request: dict) -> dict:
        arr = self._session.get_array(request["id"])
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
        self._session.free_arrays(request["ids"])
        return {"status": "ok"}

    def _handle_budget_status(self) -> dict:
        return {"status": "ok", "result": self._session.budget_status()}

    def _handle_create_from_data(self, request: dict) -> dict:
        arr = np.frombuffer(request["data"], dtype=request["dtype"]).reshape(request["shape"]).copy()
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
        raw_args = request.get("args", [])
        kwargs = request.get("kwargs", {})

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
        return arg

    # ------------------------------------------------------------------
    # Result packing
    # ------------------------------------------------------------------

    def _pack_result(self, result: Any) -> dict:
        """Pack a mechestim function result into a response dict."""
        budget = self._session.budget_status()

        if isinstance(result, np.ndarray):
            handle = self._session.store_array(result)
            meta = self._session.array_metadata(handle)
            return {"status": "ok", "result": meta, "budget": budget}

        if isinstance(result, tuple) and all(isinstance(r, np.ndarray) for r in result):
            multi = []
            for arr in result:
                handle = self._session.store_array(arr)
                multi.append(self._session.array_metadata(handle))
            return {"status": "ok", "result": {"multi": multi}, "budget": budget}

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
