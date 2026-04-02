"""Budget context manager and operation recording for mechestim."""

from __future__ import annotations

import threading
from typing import NamedTuple

from mechestim.errors import BudgetExhaustedError


class OpRecord(NamedTuple):
    """Record of a single counted operation."""

    op_name: str
    subscripts: str | None
    shapes: tuple
    flop_cost: int
    cumulative: int
    namespace: str | None = None


_thread_local = threading.local()


def get_active_budget() -> BudgetContext | None:
    """Return the active BudgetContext, or None if outside any context."""
    return getattr(_thread_local, "active_budget", None)


class BudgetContext:
    """Context manager for FLOP budget enforcement.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed. Must be > 0.
    flop_multiplier : float, optional
        Multiplier applied to all FLOP costs. Default 1.
    """

    def __init__(
        self,
        flop_budget: int,
        flop_multiplier: float = 1.0,
        quiet: bool = False,
        namespace: str | None = None,
    ):
        if flop_budget <= 0:
            raise ValueError(f"flop_budget must be > 0, got {flop_budget}")
        self._flop_budget = flop_budget
        self._flop_multiplier = flop_multiplier
        self._flops_used = 0
        self._op_log: list[OpRecord] = []
        self._quiet = quiet
        self._namespace = namespace

    @property
    def flop_budget(self) -> int:
        return self._flop_budget

    @property
    def flops_used(self) -> int:
        return self._flops_used

    @property
    def flops_remaining(self) -> int:
        return self._flop_budget - self._flops_used

    @property
    def flop_multiplier(self) -> float:
        return self._flop_multiplier

    @property
    def op_log(self) -> list[OpRecord]:
        return self._op_log

    @property
    def namespace(self) -> str | None:
        return self._namespace

    def deduct(
        self, op_name: str, *, flop_cost: int, subscripts: str | None, shapes: tuple
    ) -> None:
        """Deduct FLOPs from the budget."""
        adjusted_cost = int(flop_cost * self._flop_multiplier)
        if adjusted_cost > self.flops_remaining:
            raise BudgetExhaustedError(
                op_name, flop_cost=adjusted_cost, flops_remaining=self.flops_remaining
            )
        self._flops_used += adjusted_cost
        self._op_log.append(
            OpRecord(
                op_name=op_name,
                subscripts=subscripts,
                shapes=shapes,
                flop_cost=adjusted_cost,
                cumulative=self._flops_used,
                namespace=self._namespace,
            )
        )

    def summary(self) -> str:
        """Return a pretty-printed FLOP budget summary."""
        lines = [
            "mechestim FLOP Budget Summary",
            "=" * 30,
            f"  Total budget:  {self._flop_budget:>14,}",
            f"  Used:          {self._flops_used:>14,}  ({100 * self._flops_used / self._flop_budget:.1f}%)",
            f"  Remaining:     {self.flops_remaining:>14,}  ({100 * self.flops_remaining / self._flop_budget:.1f}%)",
            "",
            "  By operation:",
        ]
        from collections import Counter

        cost_by_op: dict[str, int] = {}
        count_by_op: Counter[str] = Counter()
        for rec in self._op_log:
            cost_by_op[rec.op_name] = cost_by_op.get(rec.op_name, 0) + rec.flop_cost
            count_by_op[rec.op_name] += 1
        for op_name, cost in sorted(cost_by_op.items(), key=lambda x: -x[1]):
            pct = 100 * cost / self._flops_used if self._flops_used > 0 else 0
            lines.append(
                f"    {op_name:<16} {cost:>12,}  ({pct:5.1f}%)  [{count_by_op[op_name]} call{'s' if count_by_op[op_name] != 1 else ''}]"
            )
        return "\n".join(lines)

    def __enter__(self) -> BudgetContext:
        if get_active_budget() is not None:
            raise RuntimeError("Cannot nest BudgetContexts")
        _thread_local.active_budget = self
        if not self._quiet:
            import sys

            import mechestim

            print(
                f"mechestim {mechestim.__version__} "
                f"(numpy {mechestim.__numpy_version__} backend) | "
                f"budget: {self._flop_budget:.2e} FLOPs",
                file=sys.stderr,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _thread_local.active_budget = None
        return None
