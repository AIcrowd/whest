# Timing Budget Tightening & Duration Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten wall-clock enforcement, complete duration tracking across all 72 missing call sites, remove all timing-related xfails, and add timing data to participant-facing output.

**Architecture:** Cooperative deadline checking via `_OpTimer.__exit__` post-op check. Duration tracking via `with budget.deduct(...)` context manager pattern across all op modules. Display layer reads existing `budget_summary_dict()` timing fields plus new per-op duration aggregation.

**Tech Stack:** Python, numpy, Rich (optional display dependency)

---

### Task 1: Post-op deadline check in `_OpTimer.__exit__`

**Files:**
- Modify: `src/whest/_budget.py:26-55` (`_OpTimer` class)
- Test: `tests/test_budget.py`

- [ ] **Step 1: Write failing test for post-op deadline enforcement**

Add to `tests/test_budget.py`:

```python
def test_post_op_deadline_check():
    """_OpTimer.__exit__ raises TimeExhaustedError if deadline passed during op."""
    import time

    import pytest

    import whest
    from whest.errors import TimeExhaustedError

    with pytest.raises(TimeExhaustedError) as exc_info:
        with whest.BudgetContext(flop_budget=int(1e15), wall_time_limit_s=0.05) as b:
            a = whest.ones((10,))
            # Burn through the time limit with a sleep inside a timed op
            # The deduct() pre-check passes, but __exit__ should catch the overshoot
            timer = b.deduct("test_op", flop_cost=1, subscripts=None, shapes=((10,),))
            with timer:
                time.sleep(0.1)  # Exceeds 0.05s limit
    assert exc_info.value.elapsed_s >= 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py::test_post_op_deadline_check -xvs`
Expected: FAIL — currently `_OpTimer.__exit__` doesn't check the deadline.

- [ ] **Step 3: Add cooperative enforcement comment and post-op check**

In `src/whest/_budget.py`, add a comment block above `_OpTimer` and modify `__exit__`:

```python
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
# The hard enforcement boundary is the container/OS level: whest
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

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._start is not None:
            duration = time.perf_counter() - self._start
            log = self._budget._op_log
            log[-1] = log[-1]._replace(duration=duration)
            self._budget._total_tracked_time += duration

            # Post-op deadline check: only if no exception is already propagating
            if (
                exc_type is None
                and self._budget._deadline is not None
                and time.perf_counter() > self._budget._deadline
            ):
                from whest.errors import TimeExhaustedError

                raise TimeExhaustedError(
                    log[-1].op_name,
                    elapsed_s=time.perf_counter() - self._budget._start_time,
                    limit_s=self._budget._wall_time_limit_s,
                )
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_budget.py::test_post_op_deadline_check -xvs`
Expected: PASS

- [ ] **Step 5: Run full budget test suite**

Run: `uv run pytest tests/test_budget.py -x --timeout=30`
Expected: All existing tests pass (the two xfail'd duration tests still xfail — that's fine, fixed in Task 3).

- [ ] **Step 6: Commit**

```bash
git add src/whest/_budget.py tests/test_budget.py
git commit -m "feat: add post-op deadline check in _OpTimer.__exit__

Cooperative wall-clock enforcement now checks the deadline both
before (in deduct()) and after (in __exit__()) each numpy call.
Includes comment documenting why we use cooperative checking
rather than signal-based preemption."
```

---

### Task 2: Duration tracking — pointwise ops (`_pointwise.py`)

**Files:**
- Modify: `src/whest/_pointwise.py:28-244` (5 factory functions)
- Test: `tests/test_budget.py` (existing xfail'd tests cover this)

This is the biggest single change — 32 call sites, but all go through 5 factory functions. The reduction factory (`_counted_reduction`) has a wrinkle: `budget.deduct()` is called *after* `np_func()` because it needs `result` to compute `extra_output` cost.

- [ ] **Step 1: Write a targeted test for pointwise duration tracking**

Add to `tests/test_budget.py`:

```python
def test_pointwise_ops_have_duration():
    """Pointwise factory ops (add, exp, sum) must record duration."""
    import whest

    with whest.BudgetContext(flop_budget=int(1e12)) as b:
        a = whest.ones((100,))
        _ = whest.add(a, a)
        _ = whest.exp(a)
        _ = whest.sum(a)

    for rec in b.op_log:
        if rec.op_name in ("add", "exp", "sum"):
            assert rec.duration is not None, f"{rec.op_name} missing duration"
            assert rec.duration >= 0, f"{rec.op_name} has negative duration"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py::test_pointwise_ops_have_duration -xvs`
Expected: FAIL — `add`, `exp`, `sum` all have `duration=None`.

- [ ] **Step 3: Wrap `_counted_unary` factory**

In `src/whest/_pointwise.py`, change `_counted_unary` (lines 28-47). The `budget.deduct()` call and the `np_func(x)` call must be wrapped together:

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

- [ ] **Step 4: Wrap `_counted_unary_multi` factory**

In `src/whest/_pointwise.py`, change `_counted_unary_multi` (lines 50-69):

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

- [ ] **Step 5: Wrap `_counted_binary` factory**

In `src/whest/_pointwise.py`, change `_counted_binary` (lines 72-155). The key change is wrapping the `budget.deduct()` and `np_func()` call:

Replace the bare deduct + call (lines 113-119):
```python
        budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        )
        # Call the underlying ufunc with the ORIGINAL inputs so that
        # Python-scalar dtype promotion (NEP 50) and FloatingPointError
        # propagation (np.errstate) work exactly as in plain numpy.
        result = np_func(x_orig, y_orig)
```

With:
```python
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        ):
            # Call the underlying ufunc with the ORIGINAL inputs so that
            # Python-scalar dtype promotion (NEP 50) and FloatingPointError
            # propagation (np.errstate) work exactly as in plain numpy.
            result = np_func(x_orig, y_orig)
```

- [ ] **Step 6: Wrap `_counted_binary_multi` factory**

In `src/whest/_pointwise.py`, change `_counted_binary_multi` (lines 158-182). Replace lines 169-172:

```python
        budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        )
        result = np_func(x, y)
```

With:
```python
        with budget.deduct(
            op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape)
        ):
            result = np_func(x, y)
```

- [ ] **Step 7: Wrap `_counted_reduction` factory**

In `src/whest/_pointwise.py`, change `_counted_reduction` (lines 185-244). This one is trickier — `budget.deduct()` is currently called *after* `np_func()` because `extra_output` needs the result shape. Restructure to call numpy first, then wrap deduct around a no-op (just to get timing), or better: compute cost before, run numpy inside the timer:

Replace lines 193-198:
```python
        cost = reduction_cost(a.shape, axis, symmetry_info=sym_info) * cost_multiplier
        result = np_func(a, axis=axis, **kwargs)
        if extra_output:
            out_shape = _np.asarray(result).shape
            cost += pointwise_cost(out_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,))
```

With:
```python
        cost = reduction_cost(a.shape, axis, symmetry_info=sym_info) * cost_multiplier
        if extra_output:
            # Pre-compute output shape for the extra cost without running numpy yet.
            # Use numpy's shape inference: reduce a.shape along axis.
            if axis is None:
                extra_cost = 1  # scalar output
            else:
                ax = axis if axis >= 0 else axis + a.ndim
                keepdims = kwargs.get("keepdims", False)
                if keepdims:
                    out_shape = a.shape[:ax] + (1,) + a.shape[ax + 1:]
                else:
                    out_shape = a.shape[:ax] + a.shape[ax + 1:]
                extra_cost = pointwise_cost(out_shape)
            cost += extra_cost
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,)):
            result = np_func(a, axis=axis, **kwargs)
```

- [ ] **Step 8: Wrap standalone functions `around`, `round`, `sort_complex`**

These are non-factory functions in `_pointwise.py` that also have bare `deduct()` calls. For each one, wrap the deduct + numpy call:

For `around` (lines 290-292), replace:
```python
    budget.deduct("around", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    result = _np.around(a, decimals=decimals, out=out)
```
With:
```python
    with budget.deduct("around", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.around(a, decimals=decimals, out=out)
```

Apply the same pattern to `round` (lines 367-369) and `sort_complex` (lines 398-399).

- [ ] **Step 9: Run test to verify it passes**

Run: `uv run pytest tests/test_budget.py::test_pointwise_ops_have_duration -xvs`
Expected: PASS

- [ ] **Step 10: Run full test suite for pointwise**

Run: `uv run pytest tests/ -x -k "pointwise or budget" --timeout=30`
Expected: All pass (except the two remaining xfails which need linalg too).

- [ ] **Step 11: Commit**

```bash
git add src/whest/_pointwise.py tests/test_budget.py
git commit -m "feat: add duration tracking to all pointwise ops

Wrap numpy calls in 'with budget.deduct(...)' context manager in
all 5 pointwise factory functions and 3 standalone functions.
Covers add, exp, multiply, sum, and all other unary/binary/reduction ops."
```

---

### Task 3: Duration tracking — linalg ops

**Files:**
- Modify: `src/whest/linalg/_decompositions.py` (7 sites)
- Modify: `src/whest/linalg/_properties.py` (8 sites)
- Modify: `src/whest/linalg/_solvers.py` (6 sites)
- Modify: `src/whest/linalg/_compound.py` (2 sites)
- Modify: `src/whest/linalg/_svd.py` (1 site)
- Modify: `src/whest/linalg/_aliases.py` (1 site — `cross`)
- Test: `tests/test_budget.py`

All linalg sites follow the same pattern: bare `budget.deduct()` followed by a numpy call. Wrap each in `with`.

- [ ] **Step 1: Write test for linalg duration tracking**

Add to `tests/test_budget.py`:

```python
def test_linalg_ops_have_duration():
    """Linalg ops must record duration."""
    import whest

    with whest.BudgetContext(flop_budget=int(1e12)) as b:
        A = whest.array([[1.0, 2.0], [3.0, 4.0]])
        _ = whest.linalg.det(A)
        _ = whest.linalg.solve(A, whest.array([1.0, 2.0]))
        _ = whest.linalg.svd(A)
        _ = whest.linalg.cholesky(A @ A.T + 2 * whest.eye(2))  # ensure positive definite

    linalg_records = [r for r in b.op_log if r.op_name.startswith("linalg.")]
    assert len(linalg_records) >= 4
    for rec in linalg_records:
        assert rec.duration is not None, f"{rec.op_name} missing duration"
        assert rec.duration >= 0, f"{rec.op_name} has negative duration"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py::test_linalg_ops_have_duration -xvs`
Expected: FAIL — linalg ops have `duration=None`.

- [ ] **Step 3: Wrap `_decompositions.py`**

For each function in `src/whest/linalg/_decompositions.py`, wrap the `budget.deduct()` + numpy call. Example for `cholesky` (lines 46-47):

Replace:
```python
    budget.deduct("linalg.cholesky", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.linalg.cholesky(a, upper=upper)
```
With:
```python
    with budget.deduct("linalg.cholesky", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.linalg.cholesky(a, upper=upper)
    return result
```

Apply this pattern to all 7 functions: `cholesky`, `qr`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `svdvals`.

- [ ] **Step 4: Wrap `_properties.py`**

Apply the same wrapping pattern to all 8 functions in `src/whest/linalg/_properties.py`: `trace`, `det`, `slogdet`, `norm`, `vector_norm`, `matrix_norm`, `cond`, `matrix_rank`.

For functions that have early-return paths (like `norm` which returns early on invalid axis), keep the early return outside the `with` block.

- [ ] **Step 5: Wrap `_solvers.py`**

Apply the pattern to all 6 functions in `src/whest/linalg/_solvers.py`: `solve`, `lstsq`, `inv`, `pinv`, `tensorsolve`, `tensorinv`.

- [ ] **Step 6: Wrap `_compound.py`**

Apply the pattern to both functions in `src/whest/linalg/_compound.py`: `multi_dot`, `matrix_power`.

- [ ] **Step 7: Wrap `_svd.py`**

Apply the pattern to `svd` in `src/whest/linalg/_svd.py` (line 75). This function has branching logic after deduct (compute_uv True/False), so wrap the entire branch:

Replace:
```python
    budget.deduct("linalg.svd", flop_cost=cost, subscripts=None, shapes=(a.shape,))

    if compute_uv:
        ...
    else:
        ...
```
With:
```python
    with budget.deduct("linalg.svd", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        if compute_uv:
            ...
        else:
            ...
```

- [ ] **Step 8: Wrap `_aliases.py` — `cross` function**

In `src/whest/linalg/_aliases.py`, wrap the `cross` function (lines 33-45). Note: `cross` calls numpy *before* deduct (to get the result size). Restructure similarly to the reduction pattern — compute cost from input shapes, then wrap:

Replace:
```python
    result = _np.linalg.cross(x1, x2, axis=axis)
    r = _np.asarray(result)
    budget.deduct(
        "linalg.cross",
        flop_cost=_builtins.max(r.size * 3, 1),
        subscripts=None,
        shapes=(_np.asarray(x1).shape, _np.asarray(x2).shape),
    )
```
With:
```python
    x1_arr = _np.asarray(x1)
    x2_arr = _np.asarray(x2)
    out_shape = _np.broadcast_shapes(x1_arr.shape, x2_arr.shape)
    out_size = 1
    for d in out_shape:
        out_size *= d
    with budget.deduct(
        "linalg.cross",
        flop_cost=_builtins.max(out_size * 3, 1),
        subscripts=None,
        shapes=(x1_arr.shape, x2_arr.shape),
    ):
        result = _np.linalg.cross(x1, x2, axis=axis)
```

- [ ] **Step 9: Run test to verify it passes**

Run: `uv run pytest tests/test_budget.py::test_linalg_ops_have_duration -xvs`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/whest/linalg/
git commit -m "feat: add duration tracking to all linalg ops

Wrap numpy calls in 'with budget.deduct(...)' context manager
across _decompositions.py (7), _properties.py (8), _solvers.py (6),
_compound.py (2), _svd.py (1), _aliases.py (1). Total: 25 sites."
```

---

### Task 4: Duration tracking — counting ops (`_counting_ops.py`)

**Files:**
- Modify: `src/whest/_counting_ops.py` (15 sites)
- Test: `tests/test_budget.py`

- [ ] **Step 1: Write test for counting ops duration**

Add to `tests/test_budget.py`:

```python
def test_counting_ops_have_duration():
    """Counting ops (trace, histogram, bincount, etc.) must record duration."""
    import whest

    with whest.BudgetContext(flop_budget=int(1e12)) as b:
        a = whest.array([1.0, 2.0, 3.0, 4.0])
        _ = whest.trace(whest.eye(3))
        _ = whest.histogram(a, bins=5)
        _ = whest.logspace(0, 1, 10)

    counting_ops = {"trace", "histogram", "logspace"}
    for rec in b.op_log:
        if rec.op_name in counting_ops:
            assert rec.duration is not None, f"{rec.op_name} missing duration"
            assert rec.duration >= 0, f"{rec.op_name} has negative duration"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py::test_counting_ops_have_duration -xvs`
Expected: FAIL

- [ ] **Step 3: Wrap all 15 call sites in `_counting_ops.py`**

Apply the standard wrapping pattern to each function. For every bare `budget.deduct()` + numpy call, wrap:

Functions to wrap: `trace`, `allclose`, `array_equal`, `array_equiv`, `histogram`, `histogram2d`, `histogramdd`, `histogram_bin_edges`, `bincount`, `logspace`, `geomspace`, `vander`, `count_nonzero`, `correlate`, `convolve`.

Example for `trace` (lines 27-28):
```python
    budget.deduct("trace", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return _np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)
```
Becomes:
```python
    with budget.deduct("trace", flop_cost=cost, subscripts=None, shapes=(a.shape,)):
        result = _np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_budget.py::test_counting_ops_have_duration -xvs`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_counting_ops.py tests/test_budget.py
git commit -m "feat: add duration tracking to all counting ops

Wrap numpy calls in 'with budget.deduct(...)' across all 15
counting op functions: trace, histogram, bincount, logspace, etc."
```

---

### Task 5: Remove duration tracking xfails

**Files:**
- Modify: `tests/test_budget.py:230,244` (remove 2 xfail markers)

- [ ] **Step 1: Remove xfail from `test_oprecord_durations_populated`**

In `tests/test_budget.py`, remove line 230:
```python
@pytest.mark.xfail(reason="Duration tracking not yet implemented for all op types")
```

- [ ] **Step 2: Remove xfail from `test_durations_populated_across_op_types`**

In `tests/test_budget.py`, remove line 244:
```python
@pytest.mark.xfail(reason="Duration tracking not yet implemented for all op types")
```

- [ ] **Step 3: Run both tests to verify they pass**

Run: `uv run pytest tests/test_budget.py::test_oprecord_durations_populated tests/test_budget.py::test_durations_populated_across_op_types -xvs`
Expected: PASS — both tests should now pass with all ops returning durations.

- [ ] **Step 4: Run full budget test suite**

Run: `uv run pytest tests/test_budget.py -x --timeout=30`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_budget.py
git commit -m "fix: remove duration tracking xfails — all ops now have duration"
```

---

### Task 6: Add `TimeExhaustedError` to client package

**Files:**
- Modify: `scripts/sync_client.py:72-77,80-118` (`_generate_errors()`)
- Generated: `whest-client/src/whest/errors.py` (auto-generated, do not edit directly)
- Modify: `tests/test_client_server_parity.py:151,161` (remove 2 xfails)

- [ ] **Step 1: Add `TimeExhaustedError` to `_generate_errors()` in `scripts/sync_client.py`**

In `scripts/sync_client.py`, add to `_CUSTOM_INIT_CLASSES` (line 72-77):

```python
    _CUSTOM_INIT_CLASSES = {
        "BudgetExhaustedError",
        "NoBudgetContextError",
        "SymmetryError",
        "TimeExhaustedError",
        "UnsupportedFunctionError",
    }
```

And add to `_CLASSES` list (after `BudgetExhaustedError`, before `NoBudgetContextError`):

```python
        (
            "TimeExhaustedError",
            "WhestError",
            "Raised when an operation exceeds the wall-clock time limit.",
        ),
```

- [ ] **Step 2: Regenerate client errors**

Run: `uv run scripts/sync_client.py`
Expected: Prints "wrote whest-client/src/whest/errors.py" (among others).

- [ ] **Step 3: Verify the generated file contains `TimeExhaustedError`**

Run: `grep TimeExhaustedError whest-client/src/whest/errors.py`
Expected: Shows class definition and entries in `_WHEST_ERRORS` and `_ERROR_MAP`.

- [ ] **Step 4: Remove xfails from `test_client_server_parity.py`**

Remove the xfail markers at lines 151 and 161:
```python
    @pytest.mark.xfail(reason="TimeExhaustedError not yet added to client package")
```

- [ ] **Step 5: Run client-server parity tests**

Run: `uv run pytest tests/test_client_server_parity.py::TestErrorParity -xvs`
Expected: PASS — both tests should now find `TimeExhaustedError` in the client.

- [ ] **Step 6: Commit**

```bash
git add scripts/sync_client.py whest-client/src/whest/errors.py tests/test_client_server_parity.py
git commit -m "feat: add TimeExhaustedError to client package

Update sync_client.py generator and regenerate client errors.py.
Remove client-server parity xfails."
```

---

### Task 7: Banner — show time limit

**Files:**
- Modify: `src/whest/_budget.py:252-262` (`__enter__` banner)
- Test: `tests/test_budget.py`

- [ ] **Step 1: Write test for banner with time limit**

Add to `tests/test_budget.py`:

```python
def test_banner_shows_time_limit(capsys):
    """Banner should include time limit when wall_time_limit_s is set."""
    import whest

    with whest.BudgetContext(flop_budget=int(1e6), wall_time_limit_s=5.0):
        pass
    captured = capsys.readouterr()
    assert "time limit: 5.0s" in captured.err


def test_banner_no_time_limit(capsys):
    """Banner should not mention time limit when wall_time_limit_s is None."""
    import whest

    with whest.BudgetContext(flop_budget=int(1e6)):
        pass
    captured = capsys.readouterr()
    assert "time limit" not in captured.err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py::test_banner_shows_time_limit -xvs`
Expected: FAIL — banner doesn't include time limit.

- [ ] **Step 3: Update banner in `__enter__`**

In `src/whest/_budget.py`, modify the `__enter__` method (lines 252-262):

```python
        if not self._quiet:
            import sys

            import whest

            banner = (
                f"whest {whest.__version__} "
                f"(numpy {whest.__numpy_version__} backend) | "
                f"budget: {self._flop_budget:.2e} FLOPs"
            )
            if self._wall_time_limit_s is not None:
                banner += f" | time limit: {self._wall_time_limit_s:.1f}s"
            print(banner, file=sys.stderr)
```

- [ ] **Step 4: Run both banner tests**

Run: `uv run pytest tests/test_budget.py::test_banner_shows_time_limit tests/test_budget.py::test_banner_no_time_limit -xvs`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_budget.py tests/test_budget.py
git commit -m "feat: show time limit in budget banner

When wall_time_limit_s is set, the entry banner now includes
'| time limit: 5.0s' so participants know they're on a clock."
```

---

### Task 8: Display — add per-op duration to `BudgetAccumulator.get_data()`

**Files:**
- Modify: `src/whest/_budget.py:376-438` (`get_data()`)
- Test: `tests/test_budget.py`

The display layer needs per-op duration data. Currently `get_data()` aggregates `flop_cost` and `calls` per op but not `duration`. Add a `"duration"` key to each op entry.

- [ ] **Step 1: Write test for per-op duration in get_data()**

Add to `tests/test_budget.py`:

```python
def test_budget_summary_dict_includes_op_duration():
    """budget_summary_dict() should include per-op duration."""
    import whest

    whest.budget_reset()
    with whest.BudgetContext(flop_budget=int(1e12), namespace="test", quiet=True):
        a = whest.ones((100,))
        _ = whest.add(a, a)

    data = whest.budget_summary_dict(by_namespace=True)
    ops = data["operations"]
    assert "add" in ops
    assert "duration" in ops["add"]
    assert ops["add"]["duration"] >= 0

    ns_ops = data["by_namespace"]["test"]["operations"]
    assert "add" in ns_ops
    assert "duration" in ns_ops["add"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py::test_budget_summary_dict_includes_op_duration -xvs`
Expected: FAIL — no `"duration"` key in op dicts.

- [ ] **Step 3: Add duration aggregation to `get_data()`**

In `src/whest/_budget.py`, modify `get_data()`. In both the top-level and per-namespace loops, add duration tracking:

In the top-level loop (around line 388-391):
```python
            for op in rec.op_log:
                if op.op_name not in ops:
                    ops[op.op_name] = {"flop_cost": 0, "calls": 0, "duration": 0.0}
                ops[op.op_name]["flop_cost"] += op.flop_cost
                ops[op.op_name]["calls"] += 1
                if op.duration is not None:
                    ops[op.op_name]["duration"] += op.duration
```

In the per-namespace loop (around line 429-435):
```python
                for op in rec.op_log:
                    if op.op_name not in by_ns[ns]["operations"]:
                        by_ns[ns]["operations"][op.op_name] = {
                            "flop_cost": 0,
                            "calls": 0,
                            "duration": 0.0,
                        }
                    by_ns[ns]["operations"][op.op_name]["flop_cost"] += op.flop_cost
                    by_ns[ns]["operations"][op.op_name]["calls"] += 1
                    if op.duration is not None:
                        by_ns[ns]["operations"][op.op_name]["duration"] += op.duration
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_budget.py::test_budget_summary_dict_includes_op_duration -xvs`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_budget.py tests/test_budget.py
git commit -m "feat: add per-op duration to budget_summary_dict()

BudgetAccumulator.get_data() now aggregates OpRecord.duration into
each operation's dict, enabling display layer to show timing breakdowns."
```

---

### Task 9: Display — plain-text timing

**Files:**
- Modify: `src/whest/_display.py:32-79` (`_plain_text_summary()`)
- Test: `tests/test_budget.py`

- [ ] **Step 1: Write test for plain-text timing display**

Add to `tests/test_budget.py`:

```python
def test_plain_text_summary_includes_timing():
    """Plain-text summary should show wall time and tracked/untracked."""
    import whest
    from whest._display import _plain_text_summary

    whest.budget_reset()
    with whest.BudgetContext(flop_budget=int(1e12), namespace="test", quiet=True):
        a = whest.ones((100,))
        _ = whest.add(a, a)

    text = _plain_text_summary()
    assert "Wall time:" in text
    assert "Tracked time:" in text
    assert "Untracked time:" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py::test_plain_text_summary_includes_timing -xvs`
Expected: FAIL

- [ ] **Step 3: Add timing section to `_plain_text_summary()`**

In `src/whest/_display.py`, add timing data after the operations section in `_plain_text_summary()`. Add before the final `return "\n".join(lines)` (around line 79):

```python
    # Timing section
    wall_time = data.get("wall_time_s")
    tracked_time = data.get("total_tracked_time")
    if wall_time is not None:
        untracked = wall_time - (tracked_time or 0.0)
        tracked_pct = 100 * (tracked_time or 0.0) / wall_time if wall_time > 0 else 0.0
        untracked_pct = 100 * untracked / wall_time if wall_time > 0 else 0.0
        lines += [
            "",
            f"  Wall time:       {wall_time:.3f}s",
            f"  Tracked time:    {(tracked_time or 0.0):.3f}s  ({tracked_pct:.1f}%)",
            f"  Untracked time:  {untracked:.3f}s  ({untracked_pct:.1f}%)",
        ]

        # Per-op timing breakdown
        op_durations = {}
        op_calls = {}
        for op_name, op_info in ops.items():
            dur = op_info.get("duration", 0.0)
            if dur > 0:
                op_durations[op_name] = dur
                op_calls[op_name] = op_info["calls"]
        if op_durations:
            lines += ["", "  By operation (time):"]
            for op_name, dur in sorted(op_durations.items(), key=lambda x: -x[1]):
                op_pct = 100 * dur / (tracked_time or 1.0)
                n = op_calls[op_name]
                call_word = "call" if n == 1 else "calls"
                lines.append(
                    f"    {op_name:<20} {dur:.3f}s  ({op_pct:5.1f}%)  [{n} {call_word}]"
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_budget.py::test_plain_text_summary_includes_timing -xvs`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/whest/_display.py tests/test_budget.py
git commit -m "feat: add timing data to plain-text budget summary

Shows wall time, tracked/untracked breakdown, and per-op timing
in _plain_text_summary() output."
```

---

### Task 10: Display — Rich timing

**Files:**
- Modify: `src/whest/_display.py:90-221` (`_rich_namespace_table()`, `_rich_summary()`)
- Test: manual verification (Rich output is complex to test programmatically)

- [ ] **Step 1: Add timing rows to Rich session totals**

In `src/whest/_display.py`, modify `_rich_summary()`. After the "Used" / "Remaining" rows in the totals table (around line 186), add timing data:

```python
    # Timing rows
    wall_time = data.get("wall_time_s")
    tracked_time = data.get("total_tracked_time")
    if wall_time is not None:
        tracked_pct = 100 * (tracked_time or 0.0) / wall_time if wall_time > 0 else 0.0
        untracked = wall_time - (tracked_time or 0.0)
        untracked_pct = 100 * untracked / wall_time if wall_time > 0 else 0.0
        totals.add_section()
        totals.add_row("Wall time", f"{wall_time:.3f}s")
        totals.add_row(
            "Tracked",
            Text(f"{(tracked_time or 0.0):.3f}s  ({tracked_pct:.1f}%)", style="dim"),
        )
        totals.add_row(
            "Untracked",
            Text(f"{untracked:.3f}s  ({untracked_pct:.1f}%)", style="dim"),
        )
```

- [ ] **Step 2: Add duration column to `_rich_namespace_table()`**

In `src/whest/_display.py`, modify `_rich_namespace_table()`. Add a "Time" column and populate it:

```python
    table.add_column("Operation", style="dim")
    table.add_column("FLOPs", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Time", justify="right", style="dim")
    table.add_column("Calls", justify="right", style="dim")

    ops = ns_data.get("operations", {})
    for op_name, op_info in sorted(ops.items(), key=lambda x: -x[1]["flop_cost"]):
        op_pct = _pct(op_info["flop_cost"], ns_data["flops_used"])
        call_word = "call" if op_info["calls"] == 1 else "calls"
        dur = op_info.get("duration", 0.0)
        time_str = f"{dur:.3f}s" if dur > 0 else ""
        table.add_row(
            op_name,
            _format_flops(op_info["flop_cost"]),
            op_pct,
            time_str,
            f"{op_info['calls']} {call_word}",
        )
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -x --timeout=60`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/whest/_display.py
git commit -m "feat: add timing data to Rich budget summary

Rich session totals now show wall time, tracked/untracked.
Namespace tables include per-op duration column."
```

---

### Task 11: Final verification — all xfails removed, full test suite green

**Files:** None (verification only)

- [ ] **Step 1: Verify no timing-related xfails remain**

Run: `grep -rn "xfail.*[Dd]uration\|xfail.*[Tt]ime[Ee]xhausted" tests/`
Expected: No matches.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x --timeout=120`
Expected: All pass.

- [ ] **Step 3: Run linting**

Run: `uv run ruff check src/whest/_budget.py src/whest/_pointwise.py src/whest/_counting_ops.py src/whest/linalg/ src/whest/_display.py scripts/sync_client.py`
Expected: No errors.

- [ ] **Step 4: Run format check**

Run: `uv run ruff format --check src/whest/_budget.py src/whest/_pointwise.py src/whest/_counting_ops.py src/whest/linalg/ src/whest/_display.py scripts/sync_client.py`
Expected: All files formatted.
