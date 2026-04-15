"""Session — ties together ArrayStore, BudgetContext, and CommsTracker for a single participant session."""

from __future__ import annotations

import numpy as np

import whest as me
from whest_server._array_store import ArrayStore
from whest_server._comms_tracker import CommsTracker


class Session:
    """A single participant session combining ArrayStore, BudgetContext, and CommsTracker.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed for this session.
    flop_multiplier : float
        Multiplier applied to each operation's raw FLOP cost.
    """

    def __init__(self, flop_budget: int, flop_multiplier: float = 1.0) -> None:
        self._store = ArrayStore()
        self._comms_tracker = CommsTracker()
        self._budget_ctx = me.BudgetContext(
            flop_budget=flop_budget,
            flop_multiplier=flop_multiplier,
            quiet=True,
        )
        self._budget_ctx.__enter__()
        self._is_open = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True if this session is still active (not yet closed)."""
        return self._is_open

    @property
    def budget_remaining(self) -> int:
        """FLOPs remaining in the current budget."""
        return self._budget_ctx.flops_remaining

    @property
    def budget_context(self) -> me.BudgetContext:
        """The active BudgetContext for this session.

        Raises
        ------
        RuntimeError
            If the session is already closed.
        """
        if not self._is_open:
            raise RuntimeError(
                "Session is closed; BudgetContext is no longer available."
            )
        return self._budget_ctx

    @property
    def comms_tracker(self) -> CommsTracker:
        """The CommsTracker for this session."""
        return self._comms_tracker

    # ------------------------------------------------------------------
    # Array operations (delegate to ArrayStore)
    # ------------------------------------------------------------------

    def store_array(self, arr: np.ndarray) -> str:
        """Store *arr* and return its handle ID.

        Delegates to :meth:`ArrayStore.put`.
        """
        return self._store.put(arr)

    def get_array(self, handle: str) -> np.ndarray:
        """Return the array for *handle*.

        Delegates to :meth:`ArrayStore.get`.

        Raises
        ------
        KeyError
            If *handle* is not in the store.
        """
        return self._store.get(handle)

    def array_metadata(self, handle: str) -> dict:
        """Return metadata dict for *handle*.

        Delegates to :meth:`ArrayStore.metadata`.

        Raises
        ------
        KeyError
            If *handle* is not in the store.
        """
        return self._store.metadata(handle)

    def free_arrays(self, handles: list) -> None:
        """Remove arrays by handle; silently ignore unknown handles.

        Delegates to :meth:`ArrayStore.free`.
        """
        self._store.free(handles)

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def budget_status(self) -> dict:
        """Return the current FLOP budget status.

        Returns
        -------
        dict with keys:
            flop_budget: total budget
            flops_used: FLOPs consumed so far
            flops_remaining: budget minus used
        """
        return {
            "flop_budget": self._budget_ctx.flop_budget,
            "flops_used": self._budget_ctx.flops_used,
            "flops_remaining": self._budget_ctx.flops_remaining,
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def close(self) -> dict:
        """Close the session, exiting the BudgetContext and clearing the ArrayStore.

        Returns
        -------
        dict with keys:
            budget_summary: str — human-readable FLOP budget summary,
                including a namespace section when labeled ops were recorded
            comms_summary: dict — CommsTracker summary

        Raises
        ------
        RuntimeError
            If the session is already closed.
        """
        if not self._is_open:
            raise RuntimeError("Session is already closed.")

        show_namespaces = any(
            op.namespace is not None for op in self._budget_ctx.op_log
        )

        self._budget_ctx.__exit__(None, None, None)
        budget_summary = self._budget_ctx.summary(by_namespace=show_namespaces)
        comms_summary = self._comms_tracker.summary()
        self._store.clear()
        self._is_open = False

        return {
            "budget_summary": budget_summary,
            "comms_summary": comms_summary,
        }
