# BudgetContext Wall-Clock Time Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add wall-clock time tracking and cooperative deadline enforcement to whest's BudgetContext, with per-op timestamps and durations.

**Architecture:** `deduct()` returns an `_OpTimer` context manager. Callers wrap their numpy calls with `with budget.deduct(...)`. OpRecord gains `timestamp` and `duration` fields. BudgetContext tracks total wall time, tracked time, and enforces an optional deadline.

**Tech Stack:** Python stdlib only (`time.perf_counter`, `threading.local`). No new dependencies.

**Spec:** `.aicrowd/superpowers/specs/2026-04-14-budget-time-tracking-design.md`

---

### Task 1: Add `TimeExhaustedError` to `errors.py`

**Files:**
- Modify: `src/whest/errors.py`
- Modify: `src/whest/__init__.py:444-452`
- Test: `tests/test_budget.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_budget.py`, add:

```python
def test_time_exhausted_error_attributes():
    from whest.errors import TimeExhaustedError
    err = TimeExhaustedError("matmul", elapsed_s=1.5, limit_s=1.0)
    assert err.op_name == "matmul"
    assert err.elapsed_s == 1.5
    assert err.limit_s == 1.0
    assert "matmul" in str(err)
    assert "1.500" in str(err)
    assert "1.000" in str(err)


def test_time_exhausted_error_is_whest_error():
    from whest.errors import TimeExhaustedError, WhestError
    err = TimeExhaustedError("add", elapsed_s=2.0, limit_s=1.0)
    assert isinstance(err, WhestError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py::test_time_exhausted_error_attributes tests/test_budget.py::test_time_exhausted_error_is_whest_error -v`
Expected: FAIL with `ImportError: cannot import name 'TimeExhaustedError'`

- [ ] **Step 3: Add `TimeExhaustedError` to `errors.py`**

In `src/whest/errors.py`, after `BudgetExhaustedError` (after line 23), add:

```python
class TimeExhaustedError(WhestError):
    """Raised when an operation exceeds the wall-clock time limit."""

    def __init__(self, op_name: str, *, elapsed_s: float, limit_s: float):
        self.op_name = op_name
        self.elapsed_s = elapsed_s
        self.limit_s = limit_s
        super().__init__(
            f"{op_name}: wall-clock time {elapsed_s:.3f}s exceeds "
            f"limit {limit_s:.3f}s. "
            f"See: {_DOCS_BASE}/#timeexhaustederror"
        )
```

- [ ] **Step 4: Export from `__init__.py`**

In `src/whest/__init__.py`, add `TimeExhaustedError` to the errors import block (line 444-452):

```python
from whest.errors import (  # noqa: F401
    BudgetExhaustedError,
    NoBudgetContextError,
    SymmetryError,
    SymmetryLossWarning,
    TimeExhaustedError,
    UnsupportedFunctionError,
    WhestError,
    WhestWarning,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_budget.py::test_time_exhausted_error_attributes tests/test_budget.py::test_time_exhausted_error_is_whest_error -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/whest/errors.py src/whest/__init__.py tests/test_budget.py
git commit -m "feat: add TimeExhaustedError for wall-clock deadline enforcement"
```

---

### Task 2: Add `_OpTimer` and Timing to `BudgetContext` Core

**Files:**
- Modify: `src/whest/_budget.py`
- Test: `tests/test_budget.py`

This is the core change. `OpRecord` gains `timestamp` and `duration`. `BudgetContext` gains timing state. `deduct()` returns `_OpTimer`.

- [ ] **Step 1: Write failing tests for OpRecord new fields**

In `tests/test_budget.py`, add:

```python
def test_oprecord_has_timestamp_and_duration_fields():
    """OpRecord has timestamp and duration fields defaulting to None."""
    import whest
    rec = whest.OpRecord(
        op_name="add", subscripts=None, shapes=((3,),),
        flop_cost=3, cumulative=3,
    )
    assert rec.timestamp is None
    assert rec.duration is None


def test_oprecord_with_timestamp_and_duration():
    import whest
    rec = whest.OpRecord(
        op_name="add", subscripts=None, shapes=((3,),),
        flop_cost=3, cumulative=3, timestamp=0.001, duration=0.0005,
    )
    assert rec.timestamp == 0.001
    assert rec.duration == 0.0005
```

- [ ] **Step 2: Write failing tests for BudgetContext timing**

In `tests/test_budget.py`, add:

```python
import time


def test_budget_context_tracks_wall_time():
    """wall_time_s is populated after context exit."""
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        _ = whest.ones((100,))
        time.sleep(0.01)
    assert b.wall_time_s is not None
    assert b.wall_time_s >= 0.01


def test_budget_context_wall_time_none_before_exit():
    """wall_time_s is None before context exit."""
    import whest
    b = whest.BudgetContext(flop_budget=int(1e9))
    assert b.wall_time_s is None


def test_budget_context_elapsed_s_live():
    """elapsed_s is available while context is active."""
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        time.sleep(0.01)
        assert b.elapsed_s >= 0.01


def test_budget_context_wall_time_limit_s_property():
    import whest
    b = whest.BudgetContext(flop_budget=int(1e9), wall_time_limit_s=5.0)
    assert b.wall_time_limit_s == 5.0
    b2 = whest.BudgetContext(flop_budget=int(1e9))
    assert b2.wall_time_limit_s is None


def test_budget_context_total_tracked_time():
    """total_tracked_time accumulates op durations."""
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        # Do some counted ops — their durations are tracked via _OpTimer
        _ = whest.add(whest.ones((1000,)), whest.ones((1000,)))
    assert b.total_tracked_time >= 0
    assert b.total_tracked_time <= b.wall_time_s


def test_budget_context_untracked_time():
    """untracked_time = wall_time_s - total_tracked_time."""
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        _ = whest.add(whest.ones((1000,)), whest.ones((1000,)))
        time.sleep(0.01)  # untracked time
    assert b.untracked_time is not None
    assert b.untracked_time >= 0
```

- [ ] **Step 3: Write failing test for deadline enforcement**

In `tests/test_budget.py`, add:

```python
def test_wall_time_limit_raises_time_exhausted():
    """TimeExhaustedError raised when deadline exceeded."""
    import whest
    from whest.errors import TimeExhaustedError
    with pytest.raises(TimeExhaustedError) as exc_info:
        with whest.BudgetContext(
            flop_budget=int(1e15), wall_time_limit_s=0.001
        ):
            # Tight loop of counted ops until deadline fires
            a = whest.ones((10,))
            for _ in range(100_000):
                a = whest.add(a, a)
    assert exc_info.value.limit_s == 0.001
    assert exc_info.value.elapsed_s >= 0.001
```

- [ ] **Step 4: Write failing test for OpRecord timestamps**

In `tests/test_budget.py`, add:

```python
def test_oprecord_timestamps_monotonic():
    """OpRecord timestamps are monotonically increasing."""
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        a = whest.ones((10,))
        for _ in range(5):
            a = whest.add(a, a)
    timestamps = [r.timestamp for r in b.op_log if r.timestamp is not None]
    assert len(timestamps) >= 5
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


def test_oprecord_durations_populated():
    """OpRecord durations are populated for ops using with-deduct pattern."""
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        a = whest.ones((10,))
        _ = whest.add(a, a)
    # At least the add op should have a duration
    add_records = [r for r in b.op_log if r.op_name == "add"]
    assert len(add_records) >= 1
    assert all(r.duration is not None for r in add_records)
    assert all(r.duration >= 0 for r in add_records)
```

- [ ] **Step 5: Run all new tests to verify they fail**

Run: `uv run pytest tests/test_budget.py -k "test_oprecord_has_timestamp or test_oprecord_with_timestamp or test_budget_context_tracks_wall or test_budget_context_wall_time_none or test_budget_context_elapsed or test_budget_context_wall_time_limit or test_budget_context_total_tracked or test_budget_context_untracked or test_wall_time_limit_raises or test_oprecord_timestamps_mono or test_oprecord_durations_pop" -v`
Expected: Multiple FAILs

- [ ] **Step 6: Implement `OpRecord` extension in `_budget.py`**

In `src/whest/_budget.py`, update the `OpRecord` NamedTuple (lines 12-20):

```python
class OpRecord(NamedTuple):
    """Record of a single counted operation."""

    op_name: str
    subscripts: str | None
    shapes: tuple
    flop_cost: int
    cumulative: int
    namespace: str | None = None
    timestamp: float | None = None
    duration: float | None = None
```

- [ ] **Step 7: Implement `_OpTimer` class in `_budget.py`**

Add after the `OpRecord` class, before `_thread_local`:

```python
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
```

- [ ] **Step 8: Add timing state to `BudgetContext.__init__`**

Update `BudgetContext.__init__` (lines 42-57) to add `wall_time_limit_s` parameter and new fields:

```python
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
```

- [ ] **Step 9: Add timing properties to `BudgetContext`**

After the existing properties (after `flop_multiplier` property, around line 75), add:

```python
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
```

- [ ] **Step 10: Update `__enter__` to start the clock**

Update `BudgetContext.__enter__` to set `_start_time` and `_deadline`:

```python
def __enter__(self) -> BudgetContext:
    current = get_active_budget()
    if current is not None and current is not _global_default:
        raise RuntimeError("Cannot nest BudgetContexts")
    self._previous_budget = current
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
```

- [ ] **Step 11: Update `__exit__` to record wall time**

Update `BudgetContext.__exit__`:

```python
def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    if self._start_time is not None:
        self._wall_time_s = time.perf_counter() - self._start_time
    _accumulator.record(self)
    _thread_local.active_budget = self._previous_budget
    return None
```

- [ ] **Step 12: Update `deduct()` to record timestamps and check deadline**

Update the `deduct` method to add timestamp, deadline check, and return `_OpTimer`:

```python
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
```

- [ ] **Step 13: Add `import time` at top of `_budget.py`**

Add `import time` to the imports at the top of `src/whest/_budget.py` (after `import threading`):

```python
import time
```

- [ ] **Step 14: Run all new tests**

Run: `uv run pytest tests/test_budget.py -k "test_oprecord_has_timestamp or test_oprecord_with_timestamp or test_budget_context_tracks_wall or test_budget_context_wall_time_none or test_budget_context_elapsed or test_budget_context_wall_time_limit or test_budget_context_total_tracked or test_budget_context_untracked or test_wall_time_limit_raises or test_oprecord_timestamps_mono or test_oprecord_durations_pop" -v`
Expected: Most PASS. `test_oprecord_durations_populated` may still fail because factory callers haven't been updated yet — that's expected and fixed in Task 3.

- [ ] **Step 15: Run the full existing test suite to check backward compat**

Run: `uv run pytest tests/test_budget.py -v`
Expected: All existing tests PASS (deduct() returning _OpTimer is backward-compatible since old callers just discard the return value).

- [ ] **Step 16: Commit**

```bash
git add src/whest/_budget.py tests/test_budget.py
git commit -m "feat: add wall-clock timing and deadline to BudgetContext"
```

---

### Task 3: Update Factory Helpers to Use `with budget.deduct(...)`

**Files:**
- Modify: `src/whest/_pointwise.py:27-46` (`_counted_unary`)
- Modify: `src/whest/_pointwise.py:49-68` (`_counted_unary_multi`)
- Modify: `src/whest/_pointwise.py:71-156` (`_counted_binary`)
- Modify: `src/whest/_pointwise.py:157-183` (`_counted_binary_multi`)
- Modify: `src/whest/_pointwise.py:184-236` (`_counted_reduction`)
- Modify: `src/whest/random/__init__.py:45-70` (`_counted_sampler`)
- Modify: `src/whest/random/__init__.py:73-90` (`_counted_dims_sampler`)
- Modify: `src/whest/random/__init__.py:161-176` (`_counted_size_only_sampler`)
- Test: `tests/test_budget.py`

Each factory gets the `with budget.deduct(...)` pattern. This covers ~80% of all ops automatically.

**Important:** Two factories (`_counted_reduction`, `_counted_sampler`) call numpy *before* `deduct()` because they need the result to compute cost. For these, the `with` block wraps both the numpy call and the deduct, so duration captures the full operation.

- [ ] **Step 1: Update `_counted_unary`**

In `src/whest/_pointwise.py`, change the wrapper in `_counted_unary` (lines 28-41):

```python
def _counted_unary(np_func, op_name: str):
    def wrapper(x):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        sym_info = x.symmetry_info if isinstance(x, SymmetricTensor) else None
        cost = pointwise_cost(x.shape, symmetry_info=sym_info)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,)):
            result = np_func(x)
        check_nan_inf(result, op_name)
        if sym_info is not None:
            result = SymmetricTensor(result, symmetric_axes=sym_info.symmetric_axes)
        if sym_info is None:
            result = _aswhest(result)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(output) FLOPs")
    return wrapper
```

- [ ] **Step 2: Update `_counted_unary_multi`**

Change the wrapper in `_counted_unary_multi` (lines 52-63):

```python
def _counted_unary_multi(np_func, op_name: str):
    """Factory for unary functions that return multiple arrays (e.g., modf, frexp)."""

    def wrapper(x):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        cost = pointwise_cost(x.shape)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,)):
            result = np_func(x)
        if isinstance(result, tuple):
            result = tuple(_aswhest(r) for r in result)
        else:
            result = _aswhest(result)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(input) FLOPs")
    return wrapper
```

- [ ] **Step 3: Update `_counted_binary`**

In `_counted_binary`, wrap the `np_func` call (the key lines are 112-118). Replace:

```python
        budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        )
        # Call the underlying ufunc with the ORIGINAL inputs so that
        # Python-scalar dtype promotion (NEP 50) and FloatingPointError
        # propagation (np.errstate) work exactly as in plain numpy.
        result = np_func(x_orig, y_orig)
```

with:

```python
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        ):
            # Call the underlying ufunc with the ORIGINAL inputs so that
            # Python-scalar dtype promotion (NEP 50) and FloatingPointError
            # propagation (np.errstate) work exactly as in plain numpy.
            result = np_func(x_orig, y_orig)
```

- [ ] **Step 4: Update `_counted_binary_multi`**

Same pattern. Replace the `budget.deduct(...)` + `result = np_func(x, y)` lines with `with budget.deduct(...):` wrapping the numpy call.

- [ ] **Step 5: Update `_counted_reduction`**

This one is special: numpy runs *before* deduct because `extra_output` needs the result shape for cost. Wrap the entire compute+deduct section:

Replace lines 193-197:

```python
        result = np_func(a, axis=axis, **kwargs)
        if extra_output:
            out_shape = _np.asarray(result).shape
            cost += pointwise_cost(out_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,))
```

with:

```python
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,)):
            result = np_func(a, axis=axis, **kwargs)
            if extra_output:
                out_shape = _np.asarray(result).shape
                extra_cost = pointwise_cost(out_shape)
                # Deduct extra cost directly (already inside timer)
                budget._flops_used += extra_cost
                budget._op_log[-1] = budget._op_log[-1]._replace(
                    flop_cost=budget._op_log[-1].flop_cost + extra_cost,
                    cumulative=budget._flops_used,
                )
```

Wait — this breaks the clean `deduct()` API. A simpler approach: compute the cost conservatively upfront (with `extra_output` always adding `pointwise_cost` of input shape as upper bound), then deduct-and-wrap. But that changes FLOP semantics.

Actually the cleanest fix: keep the existing cost calculation order but move `deduct` before `np_func` and accept that for `extra_output=True` reductions the cost is slightly approximate (uses input shape instead of output shape for the extra cost). Let me check if `extra_output` is actually used...

- [ ] **Step 5 (revised): Check `extra_output` usage and update `_counted_reduction`**

Search the codebase for `extra_output=True` to understand the impact. Then apply the simplest correct approach:

For `_counted_reduction`, use the timer around just the `np_func` call. The extra_output cost adjustment happens outside the timer — this is fine because it's a cheap shape inspection, not a numpy computation:

```python
        if extra_output:
            # Pre-compute with np_func to get output shape for cost
            result = np_func(a, axis=axis, **kwargs)
            out_shape = _np.asarray(result).shape
            cost += pointwise_cost(out_shape)
            with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,)):
                pass  # numpy already executed; timer captures ~0 but FLOP cost is correct
        else:
            with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,)):
                result = np_func(a, axis=axis, **kwargs)
```

Actually this is awkward too. The simplest correct approach: for `extra_output` reductions, wrap the numpy call in the timer but do the deduct before with the base cost, then adjust the cost after:

The cleanest approach is actually to just wrap the numpy call and accept that for the `extra_output=True` case (which is rare — only used for `std` and `var`), the timer captures the numpy execution time correctly even though the cost adjustment happens after:

```python
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,)):
            result = np_func(a, axis=axis, **kwargs)
        if extra_output:
            out_shape = _np.asarray(result).shape
            extra = pointwise_cost(out_shape)
            budget._flops_used += extra
            # Update the last OpRecord with corrected cost
            last = budget._op_log[-1]
            budget._op_log[-1] = last._replace(
                flop_cost=last.flop_cost + extra,
                cumulative=budget._flops_used,
            )
```

This preserves exact FLOP accounting and accurate timing. The `extra_output` cost fixup is a metadata adjustment, not a numpy computation.

- [ ] **Step 6: Update `_counted_sampler` in `random/__init__.py`**

`_counted_sampler` calls numpy before deduct (needs result to compute size). Restructure to wrap both:

```python
def _counted_sampler(np_func, op_name):
    """Factory for simple samplers: cost = numel(output)."""

    def wrapper(*args, **kwargs):
        budget = require_budget()
        # Must call np_func first to determine output size
        result = np_func(*args, **kwargs)
        if isinstance(result, _np.ndarray):
            n = _builtins.max(result.size, 1)
        elif isinstance(result, (int, float, _np.integer, _np.floating)):
            n = 1
        else:
            n = 1
        with budget.deduct(op_name, flop_cost=n, subscripts=None, shapes=((n,),)):
            pass  # numpy already executed; deduct records the FLOP cost
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    return wrapper
```

Note: for samplers, the `_OpTimer` duration will be ~0 since numpy already ran. This is accurate — the FLOP cost is recorded, and the actual wall-clock of the numpy call is captured in `untracked_time`. This is acceptable because random sampling is typically fast and the FLOP cost is the meaningful metric.

- [ ] **Step 7: Update `_counted_dims_sampler`**

```python
def _counted_dims_sampler(np_func, op_name):
    """Factory for rand/randn that take *dims instead of size=."""

    def wrapper(*dims):
        budget = require_budget()
        n = int(_np.prod(dims)) if dims else 1
        cost = _builtins.max(n, 1)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=((n,),)):
            if dims:
                result = np_func(*dims)
            else:
                result = np_func()
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    return wrapper
```

- [ ] **Step 8: Update `_counted_size_only_sampler`**

```python
def _counted_size_only_sampler(np_func, op_name):
    """Factory for samplers where the only arg is ``size`` (positional or kw)."""

    def wrapper(size=None):
        budget = require_budget()
        n = _output_size(size=size)
        cost = _builtins.max(n, 1)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=((n,),)):
            result = np_func(size=size)
        return result

    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    wrapper.__doc__ = (
        f"Counted version of ``numpy.random.{op_name}``. Cost: numel(output) FLOPs."
    )
    return wrapper
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest tests/test_budget.py -v`
Expected: All tests PASS including `test_oprecord_durations_populated`

Run: `uv run pytest tests/ -x --timeout=120`
Expected: Full suite PASS

- [ ] **Step 10: Commit**

```bash
git add src/whest/_pointwise.py src/whest/random/__init__.py
git commit -m "feat: update factory helpers to use with-deduct timer pattern"
```

---

### Task 4: Update Standalone Functions to Use `with budget.deduct(...)`

**Files:**
- Modify: `src/whest/_pointwise.py` (standalone functions: around, sort_complex, isclose, vecdot, matvec, vecmat, clip, ptp, dot, matmul, inner, outer, tensordot, vdot, kron, cross, diff, gradient, round)
- Modify: `src/whest/_counting_ops.py` (trace, allclose, array_equal, array_equiv, histogram*, bincount, logspace, geomspace, vander, apply_along_axis, apply_over_axes, piecewise)
- Modify: `src/whest/_free_ops.py` (all charged "free" ops: array, full, diag, arange, linspace, concatenate, stack, etc.)
- Modify: `src/whest/_einsum.py` (einsum, einsum_path)
- Modify: `src/whest/_sorting_ops.py`
- Modify: `src/whest/_polynomial.py`
- Modify: `src/whest/_window.py`
- Modify: `src/whest/_unwrap.py`
- Modify: `src/whest/linalg/_svd.py`
- Modify: `src/whest/linalg/_decompositions.py`
- Modify: `src/whest/linalg/_solvers.py`
- Modify: `src/whest/linalg/_properties.py`
- Modify: `src/whest/linalg/_compound.py`
- Modify: `src/whest/fft/_transforms.py`
- Modify: `src/whest/stats/_base.py`
- Test: `tests/test_budget.py`

This is a mechanical change across ~200 call sites. Every `budget.deduct(...)` followed by a numpy call becomes `with budget.deduct(...): numpy_call`.

- [ ] **Step 1: Write a comprehensive duration test**

In `tests/test_budget.py`, add:

```python
def test_durations_populated_across_op_types():
    """Verify that durations are populated for various operation types."""
    import whest

    with whest.BudgetContext(flop_budget=int(1e12)) as b:
        a = whest.array([1.0, 2.0, 3.0])    # free_ops
        b_arr = whest.ones((3,))             # truly free, no deduct
        c = whest.add(a, b_arr)              # pointwise binary
        d = whest.exp(c)                     # pointwise unary
        e = whest.sum(d)                     # reduction
        f = whest.concatenate([a, c])        # free_ops (charged)
        g = whest.linspace(0, 1, 10)         # free_ops (charged)

    records_with_duration = [r for r in b.op_log if r.duration is not None]
    records_without = [r for r in b.op_log if r.duration is None]
    # All records should have durations after this task
    assert len(records_without) == 0, (
        f"Ops missing duration: {[r.op_name for r in records_without]}"
    )
    assert all(r.duration >= 0 for r in records_with_duration)
```

- [ ] **Step 2: Apply the pattern to `_pointwise.py` standalone functions**

For each standalone function, the change is mechanical. Example for `matmul` (line 744):

Replace:
```python
    budget.deduct("matmul", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.matmul(a, b)
```
With:
```python
    with budget.deduct("matmul", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape)):
        result = _np.matmul(a, b)
```

Apply this to all ~19 standalone functions in `_pointwise.py`.

- [ ] **Step 3: Apply the pattern to `_counting_ops.py`**

Same mechanical change for all 15 functions. Example for `trace` (line 26):

Replace:
```python
    budget.deduct("trace", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.trace(a, **kwargs)
```
With:
```python
    with budget.deduct("trace", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.trace(a, **kwargs)
    return result
```

- [ ] **Step 4: Apply the pattern to `_free_ops.py`**

Same mechanical change for all ~69 charged "free" ops. Many use `return` directly after deduct, so they need a temp variable:

Replace:
```python
    budget.deduct("array", flop_cost=cost, subscripts=None, shapes=((n,),))
    return WhestArray(...)
```
With:
```python
    with budget.deduct("array", flop_cost=cost, subscripts=None, shapes=((n,),)):
        result = WhestArray(...)
    return result
```

- [ ] **Step 5: Apply to `_einsum.py`**

For `einsum()` (lines 151-159), wrap `_execute_pairwise`:

Replace:
```python
    budget.deduct(
        "einsum",
        flop_cost=path_info.optimized_cost,
        subscripts=subscripts,
        shapes=tuple(shapes),
    )

    # Execute pairwise steps
    result = _execute_pairwise(path_info, list(operands))
```
With:
```python
    with budget.deduct(
        "einsum",
        flop_cost=path_info.optimized_cost,
        subscripts=subscripts,
        shapes=tuple(shapes),
    ):
        result = _execute_pairwise(path_info, list(operands))
```

- [ ] **Step 6: Apply to linalg modules**

Update all deduct calls in:
- `linalg/_svd.py` (1 call)
- `linalg/_decompositions.py` (7 calls)
- `linalg/_solvers.py` (6 calls)
- `linalg/_properties.py` (8 calls)
- `linalg/_compound.py` (2 calls)

Same mechanical pattern for each.

- [ ] **Step 7: Apply to remaining modules**

Update all deduct calls in:
- `fft/_transforms.py` (14 calls)
- `stats/_base.py` (1 call)
- `_sorting_ops.py` (18 calls)
- `_polynomial.py` (10 calls)
- `_window.py` (5 calls)
- `_unwrap.py` (1 call)

- [ ] **Step 8: Run the comprehensive duration test**

Run: `uv run pytest tests/test_budget.py::test_durations_populated_across_op_types -v`
Expected: PASS

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest tests/ -x --timeout=120`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add src/whest/_pointwise.py src/whest/_counting_ops.py src/whest/_free_ops.py src/whest/_einsum.py src/whest/_sorting_ops.py src/whest/_polynomial.py src/whest/_window.py src/whest/_unwrap.py src/whest/linalg/ src/whest/fft/ src/whest/stats/
git commit -m "feat: add with-deduct timer to all standalone op functions"
```

---

### Task 5: Update `NamespaceRecord`, `BudgetAccumulator`, and Summary

**Files:**
- Modify: `src/whest/_budget.py:231-335`
- Test: `tests/test_budget.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_budget.py`, add:

```python
def test_namespace_record_includes_time():
    import whest
    with whest.BudgetContext(flop_budget=int(1e9), namespace="test") as b:
        _ = whest.add(whest.ones((10,)), whest.ones((10,)))
    data = whest.budget_summary_dict(by_namespace=True)
    assert "wall_time_s" in data
    assert data["wall_time_s"] is not None
    assert data["wall_time_s"] > 0
    assert "total_tracked_time" in data
    ns_data = data["by_namespace"]["test"]
    assert "wall_time_s" in ns_data
    whest.budget_reset()


def test_summary_includes_time_section():
    import whest
    with whest.BudgetContext(flop_budget=int(1e9)) as b:
        _ = whest.add(whest.ones((10,)), whest.ones((10,)))
    summary = b.summary()
    assert "Wall time:" in summary
    assert "Tracked time:" in summary


def test_budget_factory_passes_wall_time_limit():
    import whest
    b = whest.budget(flop_budget=int(1e9), wall_time_limit_s=2.0)
    assert b.wall_time_limit_s == 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_budget.py -k "test_namespace_record_includes_time or test_summary_includes_time or test_budget_factory" -v`
Expected: FAIL

- [ ] **Step 3: Update `NamespaceRecord`**

In `src/whest/_budget.py`, update `NamespaceRecord` (around line 231):

```python
class NamespaceRecord(NamedTuple):
    """Snapshot of a BudgetContext's state at close time."""

    namespace: str | None
    flop_budget: int
    flops_used: int
    op_log: list[OpRecord]
    wall_time_s: float | None = None
    total_tracked_time: float | None = None
```

- [ ] **Step 4: Update `BudgetAccumulator.record()`**

```python
def record(self, ctx: BudgetContext) -> None:
    """Snapshot a BudgetContext and store it."""
    self._records.append(
        NamespaceRecord(
            namespace=ctx.namespace,
            flop_budget=ctx.flop_budget,
            flops_used=ctx.flops_used,
            op_log=list(ctx.op_log),
            wall_time_s=ctx.wall_time_s,
            total_tracked_time=ctx.total_tracked_time,
        )
    )
```

- [ ] **Step 5: Update `get_data()` to include time**

In `BudgetAccumulator.get_data()`, add time aggregation:

```python
def get_data(self, by_namespace: bool = False) -> dict:
    total_budget = 0
    total_used = 0
    total_wall_time: float | None = None
    total_tracked: float | None = None
    ops: dict[str, dict] = {}

    for rec in self._records:
        total_budget += rec.flop_budget
        total_used += rec.flops_used
        if rec.wall_time_s is not None:
            total_wall_time = (total_wall_time or 0.0) + rec.wall_time_s
        if rec.total_tracked_time is not None:
            total_tracked = (total_tracked or 0.0) + rec.total_tracked_time
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
        "wall_time_s": total_wall_time,
        "total_tracked_time": total_tracked,
    }

    if by_namespace:
        by_ns: dict[str | None, dict] = {}
        for rec in self._records:
            ns = rec.namespace
            if ns not in by_ns:
                by_ns[ns] = {
                    "flop_budget": 0, "flops_used": 0, "operations": {},
                    "wall_time_s": None, "total_tracked_time": None,
                }
            by_ns[ns]["flop_budget"] += rec.flop_budget
            by_ns[ns]["flops_used"] += rec.flops_used
            if rec.wall_time_s is not None:
                by_ns[ns]["wall_time_s"] = (by_ns[ns]["wall_time_s"] or 0.0) + rec.wall_time_s
            if rec.total_tracked_time is not None:
                by_ns[ns]["total_tracked_time"] = (
                    (by_ns[ns]["total_tracked_time"] or 0.0) + rec.total_tracked_time
                )
            for op in rec.op_log:
                if op.op_name not in by_ns[ns]["operations"]:
                    by_ns[ns]["operations"][op.op_name] = {"flop_cost": 0, "calls": 0}
                by_ns[ns]["operations"][op.op_name]["flop_cost"] += op.flop_cost
                by_ns[ns]["operations"][op.op_name]["calls"] += 1
        result["by_namespace"] = by_ns

    return result
```

- [ ] **Step 6: Update `summary()` to show time data**

In `BudgetContext.summary()`, append time section when available:

```python
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
    time_by_op: dict[str, float] = {}
    for rec in self._op_log:
        cost_by_op[rec.op_name] = cost_by_op.get(rec.op_name, 0) + rec.flop_cost
        count_by_op[rec.op_name] += 1
        if rec.duration is not None:
            time_by_op[rec.op_name] = time_by_op.get(rec.op_name, 0.0) + rec.duration
    for op_name, cost in sorted(cost_by_op.items(), key=lambda x: -x[1]):
        pct = 100 * cost / self._flops_used if self._flops_used > 0 else 0
        lines.append(
            f"    {op_name:<16} {cost:>12,}  ({pct:5.1f}%)  [{count_by_op[op_name]} call{'s' if count_by_op[op_name] != 1 else ''}]"
        )

    if self._wall_time_s is not None:
        lines.append("")
        lines.append(f"  Wall time:       {self._wall_time_s:.3f}s")
        tracked_pct = (
            100 * self._total_tracked_time / self._wall_time_s
            if self._wall_time_s > 0
            else 0
        )
        untracked = self._wall_time_s - self._total_tracked_time
        untracked_pct = 100 - tracked_pct
        lines.append(
            f"  Tracked time:    {self._total_tracked_time:.3f}s  ({tracked_pct:.1f}%)"
        )
        lines.append(
            f"  Untracked time:  {untracked:.3f}s  ({untracked_pct:.1f}%)"
        )
        if time_by_op:
            lines.append("")
            lines.append("  By operation (time):")
            for op_name, t in sorted(time_by_op.items(), key=lambda x: -x[1]):
                t_pct = (
                    100 * t / self._total_tracked_time
                    if self._total_tracked_time > 0
                    else 0
                )
                lines.append(
                    f"    {op_name:<16} {t:.3f}s  ({t_pct:5.1f}%)  [{count_by_op[op_name]} call{'s' if count_by_op[op_name] != 1 else ''}]"
                )

    return "\n".join(lines)
```

- [ ] **Step 7: Update `budget()` factory**

Update the `budget()` function to pass through `wall_time_limit_s`:

```python
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
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_budget.py -v`
Expected: All PASS

- [ ] **Step 9: Run full suite**

Run: `uv run pytest tests/ -x --timeout=120`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add src/whest/_budget.py
git commit -m "feat: add time data to NamespaceRecord, accumulator, and summary"
```

---

### Task 6: Thread Safety Test and Final Verification

**Files:**
- Test: `tests/test_budget.py`

- [ ] **Step 1: Write thread isolation test**

In `tests/test_budget.py`, add:

```python
import threading


def test_thread_isolation_time_tracking():
    """Two threads with separate BudgetContexts track time independently."""
    import whest
    from whest._budget import _reset_global_default

    results = {}

    def worker(name, sleep_time):
        _reset_global_default()
        with whest.BudgetContext(flop_budget=int(1e9), quiet=True) as b:
            _ = whest.add(whest.ones((10,)), whest.ones((10,)))
            time.sleep(sleep_time)
            _ = whest.add(whest.ones((10,)), whest.ones((10,)))
        results[name] = b.wall_time_s

    t1 = threading.Thread(target=worker, args=("fast", 0.01))
    t2 = threading.Thread(target=worker, args=("slow", 0.05))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["fast"] < results["slow"]
    assert results["fast"] >= 0.01
    assert results["slow"] >= 0.05
```

- [ ] **Step 2: Run thread test**

Run: `uv run pytest tests/test_budget.py::test_thread_isolation_time_tracking -v`
Expected: PASS

- [ ] **Step 3: Run complete test suite**

Run: `uv run pytest tests/ -x --timeout=120`
Expected: All PASS

- [ ] **Step 4: Run budget-specific tests comprehensively**

Run: `uv run pytest tests/test_budget.py tests/test_budget_accumulator.py tests/test_budget_decorator.py tests/test_budget_display.py tests/test_budget_namespace.py tests/test_budget_weights.py tests/test_global_budget.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_budget.py
git commit -m "test: add thread isolation test for wall-clock time tracking"
```
