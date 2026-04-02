"""Client-side BudgetContext proxy that delegates to the mechestim server."""

from __future__ import annotations

from mechestim._connection import get_connection
from mechestim._protocol import (
    encode_budget_close,
    encode_budget_open,
    encode_budget_status,
)

# Module-level guard: only one BudgetContext can be active at a time.
_active_context = None


class OpRecord:
    """Record of a single operation's FLOP cost.

    Parameters
    ----------
    op_name:
        Name of the operation (e.g. ``"dot"``).
    flop_cost:
        FLOPs charged for this operation.
    cumulative:
        Total FLOPs used after this operation.
    """

    def __init__(self, op_name: str, flop_cost: int, cumulative: int) -> None:
        self.op_name = op_name
        self.flop_cost = flop_cost
        self.cumulative = cumulative

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OpRecord(op_name={self.op_name!r}, "
            f"flop_cost={self.flop_cost}, cumulative={self.cumulative})"
        )


class BudgetContext:
    """Context manager that opens/closes a FLOP budget on the server.

    Parameters
    ----------
    flop_budget:
        Maximum FLOPs allowed within this context.
    flop_multiplier:
        Scaling factor applied to each operation's raw FLOP count before
        it is charged against the budget.  Defaults to ``1.0``.
    quiet:
        If ``True``, suppress informational output.  Defaults to ``False``.
    namespace:
        Optional label for grouping budget records.

    Example
    -------
    >>> with BudgetContext(flop_budget=1_000_000) as ctx:
    ...     result = mechestim.dot(a, b)
    ...     print(ctx.summary())
    """

    def __init__(
        self,
        flop_budget: int,
        flop_multiplier: float = 1.0,
        quiet: bool = False,
        namespace: str | None = None,
    ) -> None:
        self._flop_budget = flop_budget
        self._flop_multiplier = flop_multiplier
        self._quiet = quiet
        self._namespace = namespace
        self._flops_used: int = 0
        self._close_summary: str | None = None
        self._is_open: bool = False
        self._previous_context = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def flop_budget(self) -> int:
        """Maximum FLOP allowance for this context."""
        return self._flop_budget

    @property
    def flops_used(self) -> int:
        """FLOPs consumed so far (cached locally, updated from server responses)."""
        return self._flops_used

    @property
    def flops_remaining(self) -> int:
        """FLOPs remaining in the budget (``budget - used``)."""
        return self._flop_budget - self._flops_used

    @property
    def flop_multiplier(self) -> float:
        """FLOP scaling multiplier."""
        return self._flop_multiplier

    @property
    def quiet(self) -> bool:
        """Whether informational output is suppressed."""
        return self._quiet

    @property
    def namespace(self) -> str | None:
        """Optional namespace label for this context."""
        return self._namespace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_budget(self, budget_info: dict) -> None:
        """Update the local ``flops_used`` cache from a server-response dict.

        Parameters
        ----------
        budget_info:
            Dict that may contain a ``"flops_used"`` key.  Missing key is
            silently ignored.
        """
        if "flops_used" in budget_info:
            self._flops_used = int(budget_info["flops_used"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Query the server for current budget status and return a formatted string.

        Also updates the local ``flops_used`` cache.

        Returns
        -------
        str
            Human-readable summary of budget usage.
        """
        conn = get_connection()
        response = conn.send_recv(encode_budget_status())
        # Budget status is nested inside "result" key
        result = response.get("result", {})
        self._update_budget(result)
        budget = result.get("flop_budget", self._flop_budget)
        used = self._flops_used
        remaining = int(budget) - used
        return f"BudgetContext: {used}/{budget} FLOPs used ({remaining} remaining)"

    # ------------------------------------------------------------------
    # Decorator support
    # ------------------------------------------------------------------

    def __call__(self, func):
        """Use BudgetContext as a decorator."""
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> BudgetContext:
        """Open the budget on the server and update the local cache."""
        global _active_context
        if _active_context is not None and _active_context is not _global_default:
            raise RuntimeError(
                "Nested BudgetContext is not supported. "
                "Only one context can be active at a time."
            )
        self._previous_context = _active_context
        conn = get_connection()
        response = conn.send_recv(
            encode_budget_open(self._flop_budget, self._flop_multiplier)
        )
        self._update_budget(response)
        self._is_open = True
        _active_context = self
        return self

    def __exit__(self, *args: object) -> None:
        """Close the budget on the server and store the close summary."""
        global _active_context
        if self._is_open:
            conn = get_connection()
            response = conn.send_recv(encode_budget_close())
            self._update_budget(response)
            self._close_summary = (
                f"BudgetContext closed: {self._flops_used}/{self._flop_budget} "
                f"FLOPs used"
            )
            self._is_open = False
            _accumulator.record(self)
        _active_context = self._previous_context

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BudgetContext(flop_budget={self._flop_budget}, "
            f"flops_used={self._flops_used}, "
            f"flop_multiplier={self._flop_multiplier})"
        )


# ------------------------------------------------------------------
# Accumulator
# ------------------------------------------------------------------


class NamespaceRecord:
    """Snapshot of a BudgetContext's state at close time."""

    def __init__(self, namespace, flop_budget, flops_used):
        self.namespace = namespace
        self.flop_budget = flop_budget
        self.flops_used = flops_used


class BudgetAccumulator:
    """Collects budget records across multiple BudgetContext sessions."""

    def __init__(self):
        self._records = []

    def record(self, ctx):
        self._records.append(
            NamespaceRecord(
                namespace=ctx.namespace,
                flop_budget=ctx.flop_budget,
                flops_used=ctx.flops_used,
            )
        )

    def get_data(self, by_namespace=False):
        total_budget = sum(r.flop_budget for r in self._records)
        total_used = sum(r.flops_used for r in self._records)
        result = {
            "flop_budget": total_budget,
            "flops_used": total_used,
            "flops_remaining": total_budget - total_used,
            "operations": {},
        }
        if by_namespace:
            by_ns = {}
            for r in self._records:
                ns = r.namespace
                if ns not in by_ns:
                    by_ns[ns] = {"flop_budget": 0, "flops_used": 0, "operations": {}}
                by_ns[ns]["flop_budget"] += r.flop_budget
                by_ns[ns]["flops_used"] += r.flops_used
            result["by_namespace"] = by_ns
        return result

    def reset(self):
        self._records.clear()


_accumulator = BudgetAccumulator()


def budget(flop_budget, flop_multiplier=1.0, quiet=False, namespace=None):
    """Create a BudgetContext usable as both a context manager and decorator."""
    return BudgetContext(
        flop_budget=flop_budget,
        flop_multiplier=flop_multiplier,
        quiet=quiet,
        namespace=namespace,
    )


def budget_data(by_namespace=False):
    """Return aggregated budget data across all recorded contexts."""
    return _accumulator.get_data(by_namespace=by_namespace)


# Note: No budget_reset() in the client — participants must not clear usage.


_global_default = None


def _get_default_budget_amount():
    import os

    raw = os.environ.get("MECHESTIM_DEFAULT_BUDGET")
    if raw is not None:
        return int(float(raw))
    return int(1e15)


def _get_global_default():
    global _global_default, _active_context
    if _global_default is None:
        _global_default = BudgetContext(
            flop_budget=_get_default_budget_amount(),
            quiet=True,
            namespace=None,
        )
        # Open it on the server
        conn = get_connection()
        response = conn.send_recv(
            encode_budget_open(
                _global_default._flop_budget, _global_default._flop_multiplier
            )
        )
        _global_default._update_budget(response)
        _global_default._is_open = True
        _active_context = _global_default
    return _global_default


def _reset_global_default():
    global _global_default, _active_context
    if _global_default is not None and _active_context is _global_default:
        _active_context = None
    _global_default = None
