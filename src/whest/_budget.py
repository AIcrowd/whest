"""Budget context manager and operation recording for whest."""

from __future__ import annotations

import functools
import threading
import time
from typing import NamedTuple

from whest.errors import BudgetExhaustedError


class OpRecord(NamedTuple):
    """Record of a single counted operation."""

    op_name: str
    subscripts: str | None
    shapes: tuple
    flop_cost: int
    cumulative: int
    namespace: str | None = None
    timestamp: float | None = None   # seconds since context __enter__
    duration: float | None = None    # wall-clock seconds of the numpy call


class _OpTimer:
    """Lightweight timer returned by BudgetContext.deduct().

    Use as a context manager to measure wall-clock duration of the
    numpy operation that follows the FLOP deduction::

        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(...)):
            result = np_func(x)

    If used without ``with``, duration stays ``None`` on the OpRecord.
    """

    __slots__ = ("_budget", "_start")

    def __init__(self, budget: "BudgetContext"):
        self._budget = budget
        self._start: float | None = None

    def __enter__(self) -> "_OpTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> bool:
        if self._start is not None:
            duration = time.perf_counter() - self._start
            log = self._budget._op_log
            log[-1] = log[-1]._replace(duration=duration)
            self._budget._total_tracked_time += duration
        return False


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
        wall_time_limit_s: float | None = None,
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
        self._wall_time_limit_s = wall_time_limit_s
        self._start_time: float | None = None
        self._deadline: float | None = None
        self._wall_time_s: float | None = None
        self._total_tracked_time: float = 0.0

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

    @property
    def wall_time_limit_s(self) -> float | None:
        return self._wall_time_limit_s

    @property
    def wall_time_s(self) -> float | None:
        return self._wall_time_s

    @property
    def elapsed_s(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def total_tracked_time(self) -> float:
        return self._total_tracked_time

    @property
    def untracked_time(self) -> float | None:
        if self._wall_time_s is None:
            return None
        return self._wall_time_s - self._total_tracked_time

    def deduct(
        self, op_name: str, *, flop_cost: int, subscripts: str | None, shapes: tuple
    ) -> _OpTimer:
        """Deduct FLOPs from the budget and return a timer context manager."""
        from whest._weights import get_weight

        weight = get_weight(op_name)
        adjusted_cost = int(flop_cost * self._flop_multiplier * weight)
        if adjusted_cost > self.flops_remaining:
            raise BudgetExhaustedError(
                op_name, flop_cost=adjusted_cost, flops_remaining=self.flops_remaining
            )
        self._flops_used += adjusted_cost

        now = time.perf_counter()
        timestamp = now - self._start_time if self._start_time is not None else None

        self._op_log.append(
            OpRecord(
                op_name=op_name,
                subscripts=subscripts,
                shapes=shapes,
                flop_cost=adjusted_cost,
                cumulative=self._flops_used,
                namespace=self._namespace,
                timestamp=timestamp,
            )
        )

        if self._deadline is not None and now > self._deadline:
            from whest.errors import TimeExhaustedError

            raise TimeExhaustedError(
                op_name,
                elapsed_s=now - self._start_time,
                limit_s=self._wall_time_limit_s,
            )

        return _OpTimer(self)

    def summary(self) -> str:
        """Return a pretty-printed FLOP budget summary."""
        header = "whest FLOP Budget Summary"
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
        self._start_time = time.perf_counter()
        if self._wall_time_limit_s is not None:
            self._deadline = self._start_time + self._wall_time_limit_s
        if not self._quiet:
            import sys

            import whest

            print(
                f"whest {whest.__version__} "
                f"(numpy {whest.__numpy_version__} backend) | "
                f"budget: {self._flop_budget:.2e} FLOPs",
                file=sys.stderr,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_time is not None:
            self._wall_time_s = time.perf_counter() - self._start_time
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
    wall_time_limit_s: float | None = None,
) -> BudgetContext:
    """Create a BudgetContext usable as both a context manager and decorator."""
    return BudgetContext(
        flop_budget=flop_budget,
        flop_multiplier=flop_multiplier,
        quiet=quiet,
        namespace=namespace,
        wall_time_limit_s=wall_time_limit_s,
    )


# ---------------------------------------------------------------------------
# Global default BudgetContext
# ---------------------------------------------------------------------------

_global_default: BudgetContext | None = None


def _get_default_budget_amount() -> int:
    """Read default budget from env var, falling back to 1e15."""
    import os

    raw = os.environ.get("WHEST_DEFAULT_BUDGET")
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


def budget_summary_dict(by_namespace: bool = False) -> dict:
    """Return aggregated budget data across all recorded contexts.

    Parameters
    ----------
    by_namespace : bool, optional
        If ``True``, include a ``"by_namespace"`` key with per-namespace
        breakdowns. Default ``False``.

    Returns
    -------
    dict
        Dictionary with keys ``"flop_budget"``, ``"flops_used"``,
        ``"flops_remaining"``, ``"operations"``, and optionally
        ``"by_namespace"``.
    """
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
