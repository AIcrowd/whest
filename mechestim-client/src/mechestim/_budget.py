"""Client-side BudgetContext proxy that delegates to the mechestim server."""
from __future__ import annotations

from mechestim._connection import get_connection
from mechestim._protocol import (
    encode_budget_open,
    encode_budget_close,
    encode_budget_status,
)


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
    ) -> None:
        self._flop_budget = flop_budget
        self._flop_multiplier = flop_multiplier
        self._quiet = quiet
        self._flops_used: int = 0
        self._close_summary: str | None = None

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
        self._update_budget(response)
        budget = response.get("flop_budget", self._flop_budget)
        used = self._flops_used
        remaining = int(budget) - used
        return (
            f"BudgetContext: {used}/{budget} FLOPs used "
            f"({remaining} remaining)"
        )

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> BudgetContext:
        """Open the budget on the server and update the local cache."""
        conn = get_connection()
        response = conn.send_recv(encode_budget_open(self._flop_budget))
        self._update_budget(response)
        return self

    def __exit__(self, *args: object) -> None:
        """Close the budget on the server and store the close summary."""
        conn = get_connection()
        response = conn.send_recv(encode_budget_close())
        self._update_budget(response)
        self._close_summary = (
            f"BudgetContext closed: {self._flops_used}/{self._flop_budget} "
            f"FLOPs used"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BudgetContext(flop_budget={self._flop_budget}, "
            f"flops_used={self._flops_used}, "
            f"flop_multiplier={self._flop_multiplier})"
        )
