# BudgetContext Wall-Clock Time Tracking

## Problem

Whestbench runs participant-submitted estimators under a FLOP budget enforced by
whest's `BudgetContext`. Currently there is no visibility into **wall-clock time**
consumed by individual operations, and no way to enforce a cooperative time
deadline from within the budget system.

Participants can spend arbitrary wall-clock time in "free" operations (reshape,
transpose, zeros, copy, memory allocation) that bypass FLOP accounting entirely.
Contest organizers need:

1. **Observability** — per-op wall-clock duration so whestbench can compute
   tracked vs untracked time, and identify which operations dominate wall-clock.
2. **Enforcement** — a cooperative wall-time deadline checked at every counted
   operation, raising `TimeExhaustedError` the same way `BudgetExhaustedError`
   works today (per-MLP failure, evaluation continues).

## Scope

This spec covers **whest (the library) only**. Whestbench integration (passing
`wall_time_limit_s` from runners, catching `TimeExhaustedError`, populating
per-MLP timing in reports) is a follow-up.

## Design Decision: Where Timing Lives

**Timing state lives inside `BudgetContext`** (not a separate context manager).
Rationale: BudgetContext already owns the execution lifecycle via
`__enter__`/`__exit__`, and every counted operation passes through `deduct()`.
Adding timing here gives a natural heartbeat checkpoint with no new thread-local
state or context nesting.

## Architecture

### 1. `deduct()` Returns an `_OpTimer` Context Manager

The central design change: `deduct()` returns a lightweight `_OpTimer` that
callers wrap around their numpy execution using `with`:

```python
# In a factory helper or standalone function:
with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(...)):
    result = np_func(x)
```

`deduct()` performs all existing work (weight lookup, FLOP check, OpRecord
creation, deadline check) and returns `_OpTimer`. The timer's `__enter__`
records `perf_counter()`. Its `__exit__` computes duration, writes it to the
OpRecord, and accumulates `_total_tracked_time` on the budget.

**Backward compatibility:** If `deduct()` is called without `with`, the returned
`_OpTimer` is discarded. The OpRecord's `duration` field stays `None`. No crash,
no behavior change for existing code.

### 2. `_OpTimer` Class

```python
class _OpTimer:
    """Lightweight timer returned by BudgetContext.deduct().

    Use as a context manager to measure wall-clock duration of the
    numpy operation that follows the FLOP deduction.
    """
    __slots__ = ('_budget', '_start')

    def __init__(self, budget: BudgetContext):
        self._budget = budget
        self._start: float | None = None

    def __enter__(self) -> _OpTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        duration = time.perf_counter() - self._start
        log = self._budget._op_log
        log[-1] = log[-1]._replace(duration=duration)
        self._budget._total_tracked_time += duration
        return False  # never suppress exceptions
```

- Two `perf_counter()` calls per timed op (~100ns total overhead).
- `__exit__` does not suppress exceptions — if numpy raises, the error
  propagates normally and the duration captures the time until failure.
- `log[-1]` is safe because BudgetContext prohibits nesting, so no other
  `deduct()` call can interleave between `_OpTimer.__enter__` and `__exit__`.

### 3. `OpRecord` Extension

```python
class OpRecord(NamedTuple):
    op_name: str
    subscripts: str | None
    shapes: tuple
    flop_cost: int
    cumulative: int
    namespace: str | None = None
    timestamp: float | None = None   # seconds since context __enter__
    duration: float | None = None    # wall-clock seconds of the numpy call
```

- `timestamp`: set by `deduct()` as `perf_counter() - _start_time`. Records
  **when** the op was deducted relative to context entry.
- `duration`: set by `_OpTimer.__exit__`. Records **how long** the numpy
  execution took.
- Both default to `None` for backward compatibility.

### 4. `BudgetContext` Changes

#### New Constructor Parameter

```python
def __init__(
    self,
    flop_budget: int,
    flop_multiplier: float = 1.0,
    quiet: bool = False,
    namespace: str | None = None,
    wall_time_limit_s: float | None = None,  # NEW
):
```

`wall_time_limit_s` sets the cooperative deadline. `None` means no limit.

#### New Instance State

| Field | Type | Set in | Purpose |
|-------|------|--------|---------|
| `_wall_time_limit_s` | `float \| None` | `__init__` | Configured deadline |
| `_start_time` | `float \| None` | `__enter__` | `perf_counter()` at entry |
| `_deadline` | `float \| None` | `__enter__` | `_start_time + limit` |
| `_wall_time_s` | `float \| None` | `__exit__` | Total wall-clock duration |
| `_total_tracked_time` | `float` | accumulated | Sum of all `_OpTimer` durations |

#### New Properties

| Property | Type | Description |
|----------|------|-------------|
| `wall_time_s` | `float \| None` | Total wall-clock after context exit |
| `wall_time_limit_s` | `float \| None` | Configured limit |
| `elapsed_s` | `float` | Live elapsed while context is active |
| `total_tracked_time` | `float` | Sum of all op durations |
| `untracked_time` | `float \| None` | `wall_time_s - total_tracked_time` (after exit) |

#### Lifecycle

**`__enter__`:**
```python
self._start_time = time.perf_counter()
if self._wall_time_limit_s is not None:
    self._deadline = self._start_time + self._wall_time_limit_s
```

**`__exit__`:**
```python
self._wall_time_s = time.perf_counter() - self._start_time
```

#### `deduct()` Changes

After the existing FLOP deduction logic:

```python
now = time.perf_counter()
timestamp = now - self._start_time if self._start_time is not None else None

# ... create OpRecord with timestamp=timestamp, duration=None ...

# Cooperative deadline check (after FLOP check — BudgetExhaustedError
# takes priority if both are exceeded on the same op)
if self._deadline is not None and now > self._deadline:
    raise TimeExhaustedError(
        op_name,
        elapsed_s=now - self._start_time,
        limit_s=self._wall_time_limit_s,
    )

return _OpTimer(self)
```

One `perf_counter()` call serves both the timestamp and deadline check.

### 5. `TimeExhaustedError`

New exception in `errors.py`, parallel to `BudgetExhaustedError`:

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

Same failure semantics as `BudgetExhaustedError`: in whestbench, caught
per-MLP, that MLP gets zeroed, evaluation continues to the next.

### 6. `summary()` Extension

When `_wall_time_s` is not None, the existing summary output appends:

```
  Wall time:       0.342s
  Tracked time:    0.298s  (87.1%)
  Untracked time:  0.044s  (12.9%)

  By operation (time):
    matmul           0.251s  (84.2%)  [12 calls]
    add              0.032s  (10.7%)  [48 calls]
    einsum           0.015s  ( 5.0%)  [3 calls]
```

### 7. `NamespaceRecord` and `BudgetAccumulator`

`NamespaceRecord` gains:

```python
class NamespaceRecord(NamedTuple):
    namespace: str | None
    flop_budget: int
    flops_used: int
    op_log: list[OpRecord]
    wall_time_s: float | None = None         # NEW
    total_tracked_time: float | None = None   # NEW
```

`BudgetAccumulator.record()` captures both new fields from the context.
`budget_summary_dict()` includes them in its output when present.

### 8. `budget()` Factory and `__init__.py`

- `budget()` factory passes through `wall_time_limit_s`.
- `__init__.py` exports `TimeExhaustedError` alongside existing errors.

### 9. Caller Integration Pattern

All callers of `budget.deduct()` adopt the `with` pattern. Two categories:

**Factory helpers** (~5 functions, each covering many ops):
- `_counted_unary()` in `_pointwise.py`
- `_counted_binary()` in `_pointwise.py`
- Similar factories in `_counting_ops.py`, `_free_ops.py`

Change:
```python
# Before:
budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(...))
result = np_func(x)

# After:
with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(...)):
    result = np_func(x)
```

One change per factory covers ~80% of operations automatically.

**Standalone functions** (~20-30 functions):
- `einsum()`, `matmul()` in `_pointwise.py` / `_einsum.py`
- `svd()`, `solve()`, `inv()`, etc. in `linalg/`
- FFT transforms in `fft/_transforms.py`
- Stats functions in `stats/`
- Sorting, polynomial ops

Same mechanical `with` wrapping around the numpy call(s).

**Truly-free ops** (reshape, transpose, zeros, ones, etc.):
No changes. They don't call `deduct()` and remain untracked at the individual
level. Their wall-clock time is captured in `untracked_time`.

## Thread Safety

All timing state lives on the `BudgetContext` instance, which is already
per-thread via `_thread_local.active_budget`. No new thread-local state.

`time.perf_counter()` is thread-safe and monotonic.

If a participant spawns threads: each thread's `BudgetContext` (if any) tracks
time independently. Wall-clock time may include GIL-blocked time — acceptable
for contest fairness and documented.

## Overhead

| Source | Cost | Frequency |
|--------|------|-----------|
| `perf_counter()` in `deduct()` | ~50ns | Every counted op |
| `perf_counter()` pair in `_OpTimer` | ~100ns | Every timed op |
| `NamedTuple._replace()` | ~200ns | Every timed op |
| **Total per op** | **~350ns** | |

For a predict() with 10,000 operations: ~3.5ms overhead — negligible compared
to numpy computation.

## Files Modified

| File | Change |
|------|--------|
| `src/whest/errors.py` | Add `TimeExhaustedError` |
| `src/whest/_budget.py` | `_OpTimer`, timing in `BudgetContext`, `OpRecord`, `NamespaceRecord` |
| `src/whest/__init__.py` | Export `TimeExhaustedError` |
| `src/whest/_pointwise.py` | `with budget.deduct(...)` in factories + standalone ops |
| `src/whest/_counting_ops.py` | `with budget.deduct(...)` in all ops |
| `src/whest/_free_ops.py` | `with budget.deduct(...)` in charged "free" ops |
| `src/whest/_einsum.py` | `with budget.deduct(...)` around `_execute_pairwise` |
| `src/whest/_sorting_ops.py` | `with budget.deduct(...)` |
| `src/whest/_polynomial.py` | `with budget.deduct(...)` |
| `src/whest/linalg/*.py` | `with budget.deduct(...)` in svd, solvers, decompositions, etc. |
| `src/whest/fft/_transforms.py` | `with budget.deduct(...)` |
| `src/whest/stats/*.py` | `with budget.deduct(...)` |
| `src/whest/random/__init__.py` | `with budget.deduct(...)` |
| `src/whest/_display.py` | Update summary formatting for time data |
| `tests/test_budget.py` | Time tracking, deadline enforcement, OpRecord timestamps |

## Verification

1. `uv run pytest tests/test_budget.py -v` — existing tests pass
2. New test: `BudgetContext(flop_budget=1e9, wall_time_limit_s=0.001)` with
   enough ops to exceed 1ms — `TimeExhaustedError` raised
3. New test: `budget.wall_time_s` is populated and positive after context exit
4. New test: `OpRecord.timestamp` values are monotonically increasing
5. New test: `OpRecord.duration` is populated when using `with budget.deduct(...)`
6. New test: `OpRecord.duration` is `None` when calling `deduct()` without `with`
7. New test: `budget.total_tracked_time <= budget.wall_time_s`
8. New test: `budget.untracked_time >= 0`
9. New test: thread isolation — two threads with separate budgets track time
   independently
10. `uv run pytest tests/ -x` — full suite passes
