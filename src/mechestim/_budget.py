"""Budget context manager and operation recording for mechestim."""

from __future__ import annotations

import functools
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
        self._previous_budget: BudgetContext | None = None

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
        header = "mechestim FLOP Budget Summary"
        if self._namespace:
            header += f" [{self._namespace}]"
        lines = [
            header,
            "=" * len(header),
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
        current = get_active_budget()
        if current is not None and current is not _global_default:
            raise RuntimeError("Cannot nest BudgetContexts")
        self._previous_budget = current  # save (may be global default or None)
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
        _accumulator.record(self)
        _thread_local.active_budget = self._previous_budget  # restore previous
        return None

    def __call__(self, func):
        """Use BudgetContext as a decorator."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


def budget(
    flop_budget: int,
    flop_multiplier: float = 1.0,
    quiet: bool = False,
    namespace: str | None = None,
) -> BudgetContext:
    """Create a BudgetContext usable as both a context manager and decorator."""
    return BudgetContext(
        flop_budget=flop_budget,
        flop_multiplier=flop_multiplier,
        quiet=quiet,
        namespace=namespace,
    )


# ---------------------------------------------------------------------------
# Global default BudgetContext
# ---------------------------------------------------------------------------

_global_default: BudgetContext | None = None


def _get_default_budget_amount() -> int:
    """Read default budget from env var, falling back to 1e15."""
    import os

    raw = os.environ.get("MECHESTIM_DEFAULT_BUDGET")
    if raw is not None:
        return int(float(raw))
    return int(1e15)


def _get_global_default() -> BudgetContext:
    """Return the global default BudgetContext, creating it lazily."""
    global _global_default
    if _global_default is None:
        _global_default = BudgetContext(
            flop_budget=_get_default_budget_amount(),
            quiet=True,
            namespace=None,
        )
        _thread_local.active_budget = _global_default
    return _global_default


def _reset_global_default() -> None:
    """Reset the global default context. For testing and core library use."""
    global _global_default
    if (
        _global_default is not None
        and getattr(_thread_local, "active_budget", None) is _global_default
    ):
        _thread_local.active_budget = None
    _global_default = None


# ---------------------------------------------------------------------------
# Session-level accumulator
# ---------------------------------------------------------------------------


class NamespaceRecord(NamedTuple):
    """Snapshot of a BudgetContext's state at close time."""

    namespace: str | None
    flop_budget: int
    flops_used: int
    op_log: list[OpRecord]


class BudgetAccumulator:
    """Collects budget records across multiple BudgetContext sessions."""

    def __init__(self) -> None:
        self._records: list[NamespaceRecord] = []

    def record(self, ctx: BudgetContext) -> None:
        """Snapshot a BudgetContext and store it."""
        self._records.append(
            NamespaceRecord(
                namespace=ctx.namespace,
                flop_budget=ctx.flop_budget,
                flops_used=ctx.flops_used,
                op_log=list(ctx.op_log),
            )
        )

    def get_data(self, by_namespace: bool = False) -> dict:
        """Return aggregated budget data across all recorded contexts."""
        total_budget = 0
        total_used = 0
        ops: dict[str, dict] = {}

        for rec in self._records:
            total_budget += rec.flop_budget
            total_used += rec.flops_used
            for op in rec.op_log:
                if op.op_name not in ops:
                    ops[op.op_name] = {"flop_cost": 0, "calls": 0}
                ops[op.op_name]["flop_cost"] += op.flop_cost
                ops[op.op_name]["calls"] += 1

        result = {
            "flop_budget": total_budget,
            "flops_used": total_used,
            "flops_remaining": total_budget - total_used,
            "operations": ops,
        }

        if by_namespace:
            by_ns: dict[str | None, dict] = {}
            for rec in self._records:
                ns = rec.namespace
                if ns not in by_ns:
                    by_ns[ns] = {"flop_budget": 0, "flops_used": 0, "operations": {}}
                by_ns[ns]["flop_budget"] += rec.flop_budget
                by_ns[ns]["flops_used"] += rec.flops_used
                for op in rec.op_log:
                    if op.op_name not in by_ns[ns]["operations"]:
                        by_ns[ns]["operations"][op.op_name] = {
                            "flop_cost": 0,
                            "calls": 0,
                        }
                    by_ns[ns]["operations"][op.op_name]["flop_cost"] += op.flop_cost
                    by_ns[ns]["operations"][op.op_name]["calls"] += 1
            result["by_namespace"] = by_ns

        return result

    def reset(self) -> None:
        """Clear all recorded data."""
        self._records.clear()


_accumulator = BudgetAccumulator()


def budget_data(by_namespace: bool = False) -> dict:
    """Return aggregated budget data across all recorded contexts."""
    # Include the global default if it has been used
    if _global_default is not None and _global_default.flops_used > 0:
        acc_copy = BudgetAccumulator()
        acc_copy._records = list(_accumulator._records)
        acc_copy.record(_global_default)
        return acc_copy.get_data(by_namespace=by_namespace)
    return _accumulator.get_data(by_namespace=by_namespace)


def budget_reset() -> None:
    """Clear all accumulated budget data. Core library only."""
    _accumulator.reset()
