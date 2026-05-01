"""Budget context manager and operation recording for flopscope."""

from __future__ import annotations

import functools
import threading
import time
import weakref
from typing import Any, Literal, NamedTuple

from flopscope.errors import BudgetExhaustedError


class OpRecord(NamedTuple):
    """Record of a single counted operation."""

    op_name: str
    subscripts: str | None
    shapes: tuple
    flop_cost: int
    cumulative: int
    namespace: str | None = None
    timestamp: float | None = None  # seconds since context __enter__
    duration: float | None = None  # wall-clock seconds of the numpy call
    flopscope_overhead: float | None = (
        None  # per-op flopscope dispatch time (preamble + deduct body + bookkeeping + postamble)
    )


# ---------------------------------------------------------------------------
# Why cooperative (not signal-based) deadline enforcement?
#
# We deliberately avoid SIGALRM / signal-based preemption because:
# 1. Python signal handlers only run between bytecodes — they cannot
#    interrupt C extensions (numpy/LAPACK/BLAS), which are exactly the
#    operations where time limits matter most.
# 2. signal.alarm() is POSIX-only (no Windows) and integer-second only.
# 3. Signals are main-thread-only and can interfere with numpy internals.
#
# The hard enforcement boundary is the container/OS level: flopscope
# submissions run inside Docker containers with kernel-level time limits
# (cgroups / rlimit) that deliver SIGKILL when exceeded.
#
# The in-library wall_time_limit_s is a UX feature: it gives participants
# a clean, informative TimeExhaustedError (with op name, elapsed time,
# and configured limit) rather than a brutal container kill.
#
# The deadline is checked:
# 1. Pre-op: in BudgetContext.deduct() before the numpy call starts.
# 2. Post-op: in _OpTimer.__exit__() after the numpy call completes.
#
# This bounds overshoot to the duration of a single numpy call.
# ---------------------------------------------------------------------------


class _OpTimer:
    """Timer for a counted op's numpy call window.

    Used as a context manager around the numpy call::

        with budget.deduct(op_name, ...):
            result = _call_numpy(np_func, ...)

    On __exit__, the block's wall time is split into:
      - tracked_time:  sum of _call_numpy durations reported during the block
      - in-block overhead: block_duration - tracked (view-casts, copyto, etc.)
    """

    __slots__ = ("_budget", "_op_index", "_block_t0", "_np_duration", "_prev_timer")

    def __init__(self, budget: BudgetContext, op_index: int):
        self._budget = budget
        self._op_index = op_index
        self._block_t0: float | None = None
        self._np_duration: float = 0.0
        self._prev_timer: _OpTimer | None = None

    def __enter__(self) -> _OpTimer:
        self._block_t0 = time.perf_counter()
        # Stack discipline supports the rare case of nested deduct() blocks
        self._prev_timer = self._budget._current_op_timer
        self._budget._current_op_timer = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self._block_t0 is not None:
            block_duration = time.perf_counter() - self._block_t0
            in_block_overhead = max(block_duration - self._np_duration, 0.0)

            self._budget._total_tracked_time += self._np_duration
            self._budget._total_flopscope_overhead_time += in_block_overhead

            op = self._budget._op_log[self._op_index]
            self._budget._op_log[self._op_index] = op._replace(
                duration=self._np_duration,
                flopscope_overhead=(op.flopscope_overhead or 0.0) + in_block_overhead,
            )

            # Post-op deadline check (preserves existing behavior)
            if (
                exc_type is None
                and self._budget._deadline is not None
                and time.perf_counter() > self._budget._deadline
            ):
                from flopscope.errors import TimeExhaustedError

                raise TimeExhaustedError(
                    op.op_name,
                    elapsed_s=time.perf_counter() - self._budget._start_time,  # type: ignore[operator]
                    limit_s=self._budget._wall_time_limit_s,  # type: ignore[arg-type]
                )

        self._budget._current_op_timer = self._prev_timer
        return False


def _call_numpy(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a numpy callable, attributing only its wall time to tracked_time.

    All numpy calls inside counted-op wrappers MUST go through this helper so
    that view-casts, copyto, errstate setup, and other flopscope-internal work
    surrounding the call are correctly bucketed as flopscope_overhead.

    Reports the call's duration to the active _OpTimer (via
    budget._current_op_timer). When no timer is active (e.g. helper called
    from a non-counted code path), it is a transparent passthrough.

    Annotated as ``Any -> Any`` to match the existing untyped-internal-helper
    convention used by ``_call_with_optional_out``. Wrappers' explicit return
    annotations carry the public type contract.
    """
    t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        d = time.perf_counter() - t0
        budget = get_active_budget()
        if budget is not None and budget._current_op_timer is not None:
            budget._current_op_timer._np_duration += d


def _counted_wrapper(fn):
    """Decorator that brackets a flopscope wrapper and bills its non-numpy,
    non-nested-overhead time to flopscope_overhead_time.

    Formula: wall - tracked_delta - overhead_delta. Handles nesting naturally
    (outer attributes only its own remainder), so no re-entrancy guard.

    Per-op attribution: wrapper-own overhead is distributed equally across
    ops created during this call (typically exactly 1 across this codebase).
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        from flopscope._validation import require_budget

        budget = require_budget()
        fs_t0 = time.perf_counter()
        tracked_baseline = budget._total_tracked_time
        overhead_baseline = budget._total_flopscope_overhead_time
        ops_before = len(budget._op_log)
        try:
            return fn(*args, **kwargs)
        finally:
            wall = time.perf_counter() - fs_t0
            tracked_delta = budget._total_tracked_time - tracked_baseline
            overhead_delta = budget._total_flopscope_overhead_time - overhead_baseline
            wrapper_own_overhead = max(wall - tracked_delta - overhead_delta, 0.0)
            budget._total_flopscope_overhead_time += wrapper_own_overhead
            ops_added = list(range(ops_before, len(budget._op_log)))
            if ops_added and wrapper_own_overhead > 0:
                per_op = wrapper_own_overhead / len(ops_added)
                for idx in ops_added:
                    op = budget._op_log[idx]
                    budget._op_log[idx] = op._replace(
                        flopscope_overhead=(op.flopscope_overhead or 0.0) + per_op
                    )

    return wrapped


_thread_local = threading.local()
_all_budget_contexts: weakref.WeakSet[BudgetContext] = weakref.WeakSet()


def get_active_budget() -> BudgetContext | None:
    """Return the active BudgetContext, or None if outside any context."""
    return getattr(_thread_local, "active_budget", None)


class _NamespaceScope:
    __slots__ = ("_budget", "_segment")

    def __init__(self, budget: BudgetContext, segment: str):
        self._budget = budget
        self._segment = segment

    def __enter__(self) -> BudgetContext:
        t0 = time.perf_counter()
        self._budget._push_namespace(self._segment)
        self._budget._total_flopscope_overhead_time += time.perf_counter() - t0
        return self._budget

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        t0 = time.perf_counter()
        self._budget._pop_namespace(self._segment)
        self._budget._total_flopscope_overhead_time += time.perf_counter() - t0
        return False


def _validate_namespace_segment(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("namespace segment must be a string")
    segment = name.strip()
    if not segment:
        raise ValueError("namespace segment must be non-empty")
    if "." in segment:
        raise ValueError("namespace segment must not contain '.'")
    return segment


def namespace(name: str) -> _NamespaceScope:
    """Create a nested namespace scope for the active budget context.

    Parameters
    ----------
    name : str
        Namespace segment to append to the active budget namespace. The
        segment must be non-empty and must not contain ``"."``.

    Returns
    -------
    _NamespaceScope
        Context manager that attributes counted operations to the nested
        namespace while the scope is active.

    Raises
    ------
    NoBudgetContextError
        If called outside an active :class:`BudgetContext`.
    ValueError
        If ``name`` is not a valid namespace segment.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>>
    >>> with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
    ...     with flops.namespace("encoder"):
    ...         _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    >>> budget.op_log[-1].namespace
    'encoder'
    """
    from flopscope.errors import NoBudgetContextError

    budget = get_active_budget()
    if budget is None:
        raise NoBudgetContextError()
    return _NamespaceScope(budget, _validate_namespace_segment(name))


def _update_operation_summary(ops: dict[str, dict], op: OpRecord) -> None:
    bucket = ops.setdefault(
        op.op_name,
        {
            "flop_cost": 0,
            "calls": 0,
            "duration": 0.0,
        },
    )
    bucket["flop_cost"] += op.flop_cost
    bucket["calls"] += 1
    if op.duration is not None:
        bucket["duration"] += op.duration


def _summarize_operations(op_log: list[OpRecord]) -> dict[str, dict]:
    ops: dict[str, dict] = {}
    for op in op_log:
        _update_operation_summary(ops, op)
    return ops


def _summarize_by_namespace(op_log: list[OpRecord]) -> dict[str | None, dict]:
    by_namespace: dict[str | None, dict] = {}
    for op in op_log:
        bucket = by_namespace.setdefault(
            op.namespace,
            {
                "flops_used": 0,
                "calls": 0,
                "tracked_time_s": 0.0,
                "flopscope_overhead_time_s": 0.0,
                "operations": {},
            },
        )
        bucket["flops_used"] += op.flop_cost
        bucket["calls"] += 1
        if op.duration is not None:
            bucket["tracked_time_s"] += op.duration
        if op.flopscope_overhead is not None:
            bucket["flopscope_overhead_time_s"] += op.flopscope_overhead
        _update_operation_summary(bucket["operations"], op)
    return by_namespace


def _timing_summary(
    wall_time_s: float | None,
    tracked_time_s: float | None,
    overhead_time_s: float | None,
) -> tuple[float | None, float, float, float | None]:
    tracked = tracked_time_s or 0.0
    overhead = overhead_time_s or 0.0
    if wall_time_s is None:
        return None, tracked, overhead, None
    untracked = wall_time_s - tracked - overhead
    if untracked < 0 and abs(untracked) < 1e-12:
        untracked = 0.0
    return wall_time_s, tracked, overhead, max(untracked, 0.0)


class BudgetContext:
    """Context manager for FLOP budget enforcement.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed. Must be > 0.
    flop_multiplier : float, optional
        Multiplier applied to all FLOP costs. Default 1.
    quiet : bool, optional
        When ``True``, suppress the startup banner printed on context entry.
    namespace : str | None, optional
        Root namespace prefix used for operation attribution inside this
        context. Nested ``flops.namespace(...)`` scopes append dotted segments.
    wall_time_limit_s : float | None, optional
        Cooperative wall-clock limit for the entire context. The timer starts
        when the context is entered and is checked before and after each
        counted NumPy call. If the deadline is exceeded, flopscope raises
        ``TimeExhaustedError`` at the next operation boundary. This is a
        diagnostic UX limit, not a hard preemptive kill.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>>
    >>> with flops.BudgetContext(flop_budget=100, quiet=True) as budget:
    ...     x = fnp.array([1.0, 2.0, 3.0])
    ...     _ = fnp.einsum("i->", x)
    >>> budget.summary_dict()["flops_used"] > 0
    True
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
        self._root_namespace = namespace
        self._namespace_stack: list[str] = []
        self._previous_budget: BudgetContext | None = None
        self._wall_time_limit_s = wall_time_limit_s
        self._start_time: float | None = None
        self._deadline: float | None = None
        self._wall_time_s: float | None = None
        self._total_tracked_time: float = 0.0
        self._total_flopscope_overhead_time: float = 0.0
        self._current_op_timer: _OpTimer | None = None
        self._recorded_flops_used = 0
        self._recorded_op_count = 0
        self._recorded_tracked_time = 0.0
        self._recorded_overhead_time: float = 0.0
        self._budget_recorded = False
        _all_budget_contexts.add(self)

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
        if not self._namespace_stack:
            return self._root_namespace
        suffix = ".".join(self._namespace_stack)
        if self._root_namespace is None:
            return suffix
        return f"{self._root_namespace}.{suffix}"

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
    def flopscope_overhead_time(self) -> float:
        """Wall-clock seconds spent inside flopscope's own dispatch code.

        Includes wrapper preambles, BudgetContext.deduct() body, _OpTimer
        bookkeeping, the flopscope-internal parts of the timed block (view-
        casts, copyto, dispatch), wrapper postambles (including
        maybe_check_nan_inf when opted in), and namespace push/pop.

        Measured per op via the @_counted_wrapper decorator. Aggregated per
        namespace via summary_dict(by_namespace=True).
        """
        return self._total_flopscope_overhead_time

    @property
    def untracked_time(self) -> float | None:
        """Wall time minus tracked_time minus flopscope_overhead_time.

        Genuinely 'neither' — user Python between ops, time.sleep, GC pauses,
        un-instrumented numpy.

        BREAKING (vs versions before issue #76): previously was
        wall_time - tracked_time. The new value subtracts measured framework
        overhead. To recover the prior 'all-unattributed' value, compute
        flopscope_overhead_time + untracked_time.
        """
        if self._wall_time_s is None:
            return None
        return self._wall_time_s - self._total_tracked_time

    def deduct(
        self, op_name: str, *, flop_cost: int, subscripts: str | None, shapes: tuple
    ) -> _OpTimer:
        """Deduct FLOPs from the budget and return a timer context manager."""
        fs_t0 = time.perf_counter()
        appended = False
        try:
            from flopscope._weights import get_weight

            weight = get_weight(op_name)
            adjusted_cost = int(flop_cost * self._flop_multiplier * weight)
            if adjusted_cost > self.flops_remaining:
                raise BudgetExhaustedError(
                    op_name,
                    flop_cost=adjusted_cost,
                    flops_remaining=self.flops_remaining,
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
                    namespace=self.namespace,
                    timestamp=timestamp,
                )
            )
            appended = True

            if self._deadline is not None and now > self._deadline:
                from flopscope.errors import TimeExhaustedError

                raise TimeExhaustedError(
                    op_name,
                    elapsed_s=now - self._start_time,  # type: ignore[operator]
                    limit_s=self._wall_time_limit_s,  # type: ignore[arg-type]
                )

            op_index = len(self._op_log) - 1
            return _OpTimer(self, op_index=op_index)
        finally:
            deduct_body_time = time.perf_counter() - fs_t0
            self._total_flopscope_overhead_time += deduct_body_time
            if appended:
                op = self._op_log[-1]
                self._op_log[-1] = op._replace(
                    flopscope_overhead=(op.flopscope_overhead or 0.0) + deduct_body_time
                )

    def summary_dict(self, by_namespace: bool = False) -> dict:
        """Return structured summary data for this budget context.

        Returns a dict with keys ``flop_budget``, ``flops_used``,
        ``flops_remaining``, ``operations``, ``wall_time_s``,
        ``tracked_time_s``, ``flopscope_overhead_time_s``,
        ``untracked_time_s``, and optionally ``by_namespace`` with per-
        namespace buckets that each include the same timing keys.

        Decomposition: ``wall_time_s == tracked_time_s
        + flopscope_overhead_time_s + untracked_time_s`` (within numerical
        tolerance).
        """
        wall_time = self._wall_time_s
        if wall_time is None and self._start_time is not None:
            wall_time = self.elapsed_s
        wall_time, tracked_time, overhead_time, untracked_time = _timing_summary(
            wall_time, self._total_tracked_time, self._total_flopscope_overhead_time
        )

        result = {
            "flop_budget": self._flop_budget,
            "flops_used": self._flops_used,
            "flops_remaining": self.flops_remaining,
            "operations": _summarize_operations(self._op_log),
            "wall_time_s": wall_time,
            "tracked_time_s": tracked_time,
            "flopscope_overhead_time_s": overhead_time,
            "untracked_time_s": untracked_time,
        }
        if by_namespace:
            result["by_namespace"] = _summarize_by_namespace(self._op_log)
        return result

    def summary(self, by_namespace: bool = False) -> str:
        """Return a pretty-printed FLOP budget summary."""
        from flopscope._display import _format_budget_summary_text

        header = "flopscope FLOP Budget Summary"
        if self.namespace:
            header += f" [{self.namespace}]"
        return _format_budget_summary_text(
            self.summary_dict(by_namespace=by_namespace),
            by_namespace=by_namespace,
            header=header,
        )

    def _push_namespace(self, segment: str) -> None:
        self._namespace_stack.append(segment)

    def _pop_namespace(self, expected: str) -> None:
        actual = self._namespace_stack.pop()
        if actual != expected:
            raise RuntimeError(
                f"Namespace stack corrupted: expected {expected!r}, got {actual!r}"
            )

    def _snapshot_record(self) -> NamespaceRecord:
        wall_time = self.wall_time_s
        if wall_time is None and self._start_time is not None:
            wall_time = self.elapsed_s

        tracked_delta = self._total_tracked_time - self._recorded_tracked_time
        if tracked_delta < 0 and abs(tracked_delta) < 1e-12:
            tracked_delta = 0.0

        overhead_delta = (
            self._total_flopscope_overhead_time - self._recorded_overhead_time
        )
        if overhead_delta < 0 and abs(overhead_delta) < 1e-12:
            overhead_delta = 0.0

        return NamespaceRecord(
            namespace=self.namespace,
            flop_budget=0 if self._budget_recorded else self.flop_budget,
            flops_used=max(self._flops_used - self._recorded_flops_used, 0),
            op_log=list(self._op_log[self._recorded_op_count :]),
            wall_time_s=wall_time,
            total_tracked_time=max(tracked_delta, 0.0),
            total_flopscope_overhead_time=max(overhead_delta, 0.0),
        )

    def _has_unrecorded_activity(self) -> bool:
        return self._flops_used > self._recorded_flops_used

    def _mark_recorded(self) -> None:
        self._recorded_flops_used = self._flops_used
        self._recorded_op_count = len(self._op_log)
        self._recorded_tracked_time = self._total_tracked_time
        self._recorded_overhead_time = self._total_flopscope_overhead_time
        self._budget_recorded = True

    def _mark_reset_baseline(self) -> None:
        self._recorded_flops_used = self._flops_used
        self._recorded_op_count = len(self._op_log)
        self._recorded_tracked_time = self._total_tracked_time
        self._recorded_overhead_time = self._total_flopscope_overhead_time
        self._budget_recorded = False

    def __enter__(self) -> BudgetContext:
        current = get_active_budget()
        if current is not None and current is not _global_default:
            raise RuntimeError("Cannot nest BudgetContexts")
        self._previous_budget = current  # save (may be global default or None)
        _thread_local.active_budget = self
        self._start_time = time.perf_counter()
        self._wall_time_s = None
        self._deadline = None
        if self._wall_time_limit_s is not None:
            self._deadline = self._start_time + self._wall_time_limit_s
        if not self._quiet:
            import sys

            import flopscope

            banner = (
                f"flopscope {flopscope.__version__} "
                f"(numpy {flopscope.__numpy_version__} backend) | "
                f"budget: {self._flop_budget:.2e} FLOPs"
            )
            if self._wall_time_limit_s is not None:
                banner += f" | time limit: {self._wall_time_limit_s:.1f}s"
            print(banner, file=sys.stderr)
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
    """Create a ``BudgetContext`` usable as a context manager or decorator.

    This helper accepts the same arguments as ``BudgetContext(...)`` and is
    convenient when you want a short, function-style entrypoint.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed inside the context.
    flop_multiplier : float, optional
        Multiplier applied to every charged FLOP cost. Default ``1.0``.
    quiet : bool, optional
        If ``True``, suppress the startup banner on context entry.
    namespace : str or None, optional
        Root namespace prefix for attribution.
    wall_time_limit_s : float or None, optional
        Cooperative wall-clock limit checked at operation boundaries.

    Returns
    -------
    BudgetContext
        A context manager that can also be used as a decorator.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> with flops.budget(1_000):
    ...     _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    """
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

    raw = os.environ.get("FLOPSCOPE_DEFAULT_BUDGET")
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
    wall_time_s: float | None = None
    total_tracked_time: float | None = None
    total_flopscope_overhead_time: float | None = None


def _snapshot_namespace_record(ctx: BudgetContext) -> NamespaceRecord:
    return ctx._snapshot_record()


class BudgetAccumulator:
    """Collects budget records across multiple BudgetContext sessions."""

    def __init__(self) -> None:
        self._records: list[NamespaceRecord] = []

    def record(self, ctx: BudgetContext) -> None:
        """Snapshot a BudgetContext and store it."""
        self._records.append(_snapshot_namespace_record(ctx))
        ctx._mark_recorded()

    def get_data(self, by_namespace: bool = False) -> dict:
        """Return aggregated budget data across all recorded contexts."""
        total_budget = 0
        total_used = 0
        total_wall_time: float | None = None
        total_tracked: float | None = None
        total_overhead: float | None = None
        all_ops: list[OpRecord] = []

        for rec in self._records:
            total_budget += rec.flop_budget
            total_used += rec.flops_used
            all_ops.extend(rec.op_log)
            if rec.wall_time_s is not None:
                total_wall_time = (total_wall_time or 0.0) + rec.wall_time_s
            if rec.total_tracked_time is not None:
                total_tracked = (total_tracked or 0.0) + rec.total_tracked_time
            if rec.total_flopscope_overhead_time is not None:
                total_overhead = (
                    total_overhead or 0.0
                ) + rec.total_flopscope_overhead_time

        wall_time, tracked_time, overhead_time, untracked_time = _timing_summary(
            total_wall_time, total_tracked, total_overhead
        )

        result = {
            "flop_budget": total_budget,
            "flops_used": total_used,
            "flops_remaining": total_budget - total_used,
            "operations": _summarize_operations(all_ops),
            "wall_time_s": wall_time,
            "tracked_time_s": tracked_time,
            "flopscope_overhead_time_s": overhead_time,
            "untracked_time_s": untracked_time,
        }

        if by_namespace:
            result["by_namespace"] = _summarize_by_namespace(all_ops)

        return result

    def reset(self) -> None:
        """Clear all recorded data."""
        self._records.clear()


_accumulator = BudgetAccumulator()


def _snapshot_records() -> list[NamespaceRecord]:
    records = list(_accumulator._records)
    active = get_active_budget()
    if _global_default is not None and _global_default._has_unrecorded_activity():
        records.append(_snapshot_namespace_record(_global_default))
    if (
        active is not None
        and active is not _global_default
        and active._has_unrecorded_activity()
    ):
        records.append(_snapshot_namespace_record(active))
    return records


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
        ``"flops_remaining"``, ``"operations"``, ``"wall_time_s"``,
        ``"tracked_time_s"``, ``"flopscope_overhead_time_s"``,
        ``"untracked_time_s"``, and optionally ``"by_namespace"``.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>> with flops.BudgetContext(flop_budget=100):
    ...     _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    >>> summary = flops.budget_summary_dict()
    >>> sorted(summary)
    ['flop_budget', 'flops_remaining', 'flops_used', 'flopscope_overhead_time_s', 'operations', 'tracked_time_s', 'untracked_time_s', 'wall_time_s']
    """
    acc_copy = BudgetAccumulator()
    acc_copy._records = _snapshot_records()
    return acc_copy.get_data(by_namespace=by_namespace)


def budget_reset() -> None:
    """Clear accumulated session-wide budget data.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Removes all recorded budget summaries and resets live baselines for
        active contexts.

    Notes
    -----
    This is primarily useful in tests, notebooks, and long-lived processes
    where you want a fresh session-wide summary.

    Examples
    --------
    >>> import flopscope as flops
    >>> import flopscope.numpy as fnp
    >>>
    >>> flops.budget_reset()
    >>> with flops.BudgetContext(flop_budget=100, quiet=True):
    ...     _ = fnp.add(fnp.array([1.0]), fnp.array([2.0]))
    >>> flops.budget_summary_dict()["flops_used"] > 0
    True
    >>> flops.budget_reset()
    >>> flops.budget_summary_dict()["flops_used"]
    0
    """
    _accumulator.reset()
    for ctx in list(_all_budget_contexts):
        ctx._mark_reset_baseline()
