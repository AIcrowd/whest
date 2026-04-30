# Whestbench pyright cleanup guide

This document is a hand-off for the **whestbench maintainer**. It catalogs every
pyright error remaining on the `migrate-to-flopscope` branch after pinning
flopscope at `chore/tighten-public-type-stubs` (the type-stub-tightening PR
that pre-empts this work).

After flopscope's PR lands and you re-pin, the whestbench-side count drops from
**84 → 22 errors** in `standard` mode. Of those 22:

- **2 are still flopscope-stub-shape but require whestbench-side fixes** —
  flopscope cannot fix them without violating LSP. Section 1.
- **20 are whestbench-internal type debt** unrelated to flopscope.
  Sections 2–5 by area.

Each error has a recommended fix that matches the codebase's existing style.

---

## How to reproduce the count

```bash
git checkout migrate-to-flopscope
# Pin flopscope at the merged-PR commit:
sed -i 's|flopscope.git@.*"|flopscope.git@<MERGED-SHA>"|' pyproject.toml
uv lock && uv sync --all-groups
uv run pyright | tail -3
# Expect: 22 errors, 0 warnings, 0 informations
```

Once you're done with this guide, the count should be **0**.

---

## Section 1 — The 2 flopscope-stub-shape errors that need a whestbench fix

`flopscope.numpy` exposes typed wrappers; using bare `numpy` in code that hands
results back into flopscope-typed APIs creates ndarray-vs-FlopscopeArray
mismatches that flopscope can't fix at its boundary.

### 1.1 `tests/test_scoring_module.py:506` and `:507`

**Error**
```
"ndarray[_AnyShape, dtype[floating[_32Bit]]]" is not assignable to "FlopscopeArray"
  in DatasetBundle.all_layer_means / final_means
```

**Cause**
```python
import numpy as np
...
all_layer_means = np.stack([...]).astype(np.float32)   # ndarray
final_means = np.stack([...]).astype(np.float32)        # ndarray
DatasetBundle(all_layer_means=all_layer_means, final_means=final_means, ...)
```

`DatasetBundle.all_layer_means` and `.final_means` are typed `FlopscopeArray`.
`numpy.stack(...).astype(...)` returns plain `ndarray`. Plain ndarray is the
parent class of `FlopscopeArray`, not the child — so the assignment is
contravariant and pyright (correctly) refuses it.

**Fix**

Use `flopscope.numpy.stack` instead of `numpy.stack`:

```python
import flopscope.numpy as fnp
...
all_layer_means = fnp.stack([...]).astype(fnp.float32)
final_means = fnp.stack([...]).astype(fnp.float32)
```

Both `fnp.stack` and `FlopscopeArray.astype` are now typed `-> FlopscopeArray`
(after the flopscope PR), so the chain produces `FlopscopeArray` end-to-end.

If for some reason a particular call site genuinely needs bare numpy (e.g.,
the test is verifying ndarray interop), wrap with `fnp.asarray(...)`:

```python
all_layer_means = fnp.asarray(np.stack([...]).astype(np.float32))
```

---

## Section 2 — Presentation layer (5 errors)

These errors are about narrowing string literals and list element types in the
presentation/UI code. They are unrelated to flopscope.

### 2.1 `src/whestbench/presentation/adapters.py:` (3 errors)

```
"str" is not assignable to "ChecklistStatus"
"list[StepsSection]" is not assignable to "list[KeyValueSection | StepsSection]"
"list[str]" is not assignable to "list[str | StepItem]"
```

**Fix patterns**

For the `ChecklistStatus` error: the value being assigned is a runtime string
that should be one of a fixed set (`"pending" | "passed" | "failed"`). Either:

```python
status: ChecklistStatus = cast(ChecklistStatus, raw_status)
# or — better — use a Literal narrowing:
if raw_status not in ("pending", "passed", "failed"):
    raise ValueError(f"unexpected status: {raw_status}")
status: ChecklistStatus = raw_status  # narrows after the guard
```

For the `list[X]` invariance errors: change the source declaration from
`list[StepsSection]` to `list[KeyValueSection | StepsSection]` (the wider
union) at the construction site, OR change the consumer parameter from
`list[KeyValueSection | StepsSection]` to `Sequence[KeyValueSection | StepsSection]`
(covariant).

The `Sequence` change is the lower-friction fix because it doesn't require
upcasting at every call site:

```python
# Before
def render_sections(sections: list[KeyValueSection | StepsSection]) -> str: ...

# After
from collections.abc import Sequence
def render_sections(sections: Sequence[KeyValueSection | StepsSection]) -> str: ...
```

### 2.2 `src/whestbench/presentation/output.py:` (1 error)

```
"str" is not assignable to "OutputFormat"
```

Same pattern as 2.1 — narrow with a guard or `cast`.

### 2.3 `src/whestbench/presentation/presenters.py:` (1 error)

```
"list[object]" is not assignable to "list[RenderableType]" in render_document
```

Same `list[X]` invariance pattern. Use `Sequence[RenderableType]` on the
parameter, or upcast the producer to `list[RenderableType]` instead of
`list[object]`.

---

## Section 3 — Reporting (3 errors)

### 3.1 `src/whestbench/reporting.py:` (3 errors at distinct call sites)

```
"list[object]" is not assignable to "list[RenderableType]" in render_blocks
```

Same as 2.3. Apply the `Sequence[RenderableType]` change to `render_blocks`
parameter signature; the three call sites all pass heterogeneous lists.

---

## Section 4 — CLI (6 errors)

### 4.1 `tests/test_cli.py:` (6 errors, 3 lines × 2 duplicates)

```
"_ExitCode" is not assignable to "ConvertibleToInt" in int.__new__
```

`_ExitCode` is presumably an `enum.Enum` with int values. `int(some_exit_code)`
fails statically because pyright doesn't see the enum's `__int__`/`__index__`.

**Fix**

If `_ExitCode` is an `IntEnum`, the conversion is implicit and the test should
just compare directly:

```python
# Before
assert int(result.exit_code) == int(_ExitCode.SUCCESS)

# After
assert result.exit_code == _ExitCode.SUCCESS
```

If `_ExitCode` is a regular `Enum` whose `.value` is the int, use that:

```python
assert result.exit_code.value == _ExitCode.SUCCESS.value
```

If neither works (e.g., `_ExitCode` is a custom class), add a `__int__`
method to it OR use `cast(int, exit_code)`.

---

## Section 5 — Tests / fixtures (8 errors)

### 5.1 `tests/test_cli_participant_commands.py:` (1 error)

```
"NDArray[Any]" is not assignable to parameter "allow_pickle: bool" in np.savez
```

The argument order in `np.savez` is `np.savez(file, *args, **kwds)` where
`*args` are arrays. The error suggests an ndarray is being passed where
`allow_pickle` is expected. Most likely:

```python
# Wrong (positional after the file)
np.savez("foo.npz", arr1, arr2, allow_pickle)

# Right
np.savez("foo.npz", arr1=arr1, arr2=arr2)
# or
np.savez("foo.npz", arr1, arr2)  # without an extra positional `allow_pickle`
```

Check the call site for an extra positional argument that's an ndarray.

### 5.2 `tests/test_presentation_renderers.py:` (2 errors)

```
"list[Panel]" is not assignable to "list[RenderableType]"
```

Same as 2.3 / 3.1 — `Sequence[RenderableType]` on the consumer signature.

### 5.3 `tests/test_reporting.py:` (1 error)

```
"list[str | StepItem]" is not assignable to "Sequence[StepItem]"
```

The consumer expected `Sequence[StepItem]` but got a wider union. Two fixes:

1. **Narrow the producer** — if the test generates only StepItem instances,
   declare the literal as `list[StepItem]` (drop the `str` from the union):
   ```python
   items: list[StepItem] = [StepItem(...), StepItem(...)]
   ```

2. **Filter at the call site** — `[x for x in items if isinstance(x, StepItem)]`.

### 5.4 `tests/test_subprocess_worker.py:` (2 errors)

```
"_BudgetEstimator" is not assignable to "BaseEstimator" in _handle_predict
"_TimeEstimator" is not assignable to "BaseEstimator" in _handle_predict
```

The two custom estimator classes (`_BudgetEstimator`, `_TimeEstimator`) don't
declare inheritance from `BaseEstimator`. Either:

1. **Make them inherit** — add `(BaseEstimator)` to the class definition.
2. **Make `BaseEstimator` a `Protocol`** — if the actual contract is duck-typed,
   convert `BaseEstimator` to a `typing.Protocol` so any class satisfying its
   methods is structurally compatible.

(2) is cleaner if the test estimators are only created in tests for testing
purposes; (1) is cleaner if there's a real inheritance relationship being
modeled.

---

## Recommended landing order

1. **Section 1 (1 commit, 1 file).** Trivial: replace `np.stack` with `fnp.stack`
   in `tests/test_scoring_module.py`. This drops the count from 22 to 20 and
   removes the only flopscope-shaped category.
2. **Section 2 (1 commit, 3 files).** Apply `Sequence` covariance on
   presentation layer signatures. Three call sites get cleanly narrower.
3. **Section 3 (1 commit, 1 file).** Same `Sequence` pattern on `render_blocks`.
4. **Section 4 (1 commit, 1 file).** Drop the `int(_ExitCode.X)` conversion in
   `tests/test_cli.py`.
5. **Section 5 (1 commit per file or 1 batched commit).** The remaining test
   fixes.

After step 5, `uv run pyright` should report `0 errors`. Once green:

- Drop `reportPrivateImportUsage = "none"` from your `[tool.pyright]` config
  (added because flopscope's old re-export pattern was triggering false
  positives — flopscope's PR adds `__all__` and you no longer need it).
- Optionally tighten your config from `typeCheckingMode = "standard"` to
  `"strict"` for ongoing rigor.

---

## Reference: flopscope's typed surface (post-PR)

The flopscope PR makes the following promises that whestbench can rely on:

| flopscope API | Returns | Notes |
|---|---|---|
| `fnp.zeros(shape, dtype=float, **kw)` | `FlopscopeArray` | dtype accepts `DTypeLike` |
| `fnp.ones(shape, dtype=float, **kw)` | `FlopscopeArray` | same |
| `fnp.array(object, dtype=None, **kw)` | `FlopscopeArray` | object: `ArrayLike` |
| `fnp.asarray(a, dtype=None, **kw)` | `FlopscopeArray` | |
| `fnp.eye(N, M=None, k=0, dtype=float, **kw)` | `FlopscopeArray` | |
| `fnp.stack(arrays, axis=0)` | `FlopscopeArray` | arrays: `Sequence[ArrayLike]` |
| `fnp.concatenate(arrays, axis=0, **kw)` | `FlopscopeArray` | same |
| `fnp.einsum(*operands, **kw)` | `FlopscopeArray` | |
| `fnp.linalg.solve(a, b)` | `FlopscopeArray` | |
| `fnp.linalg.norm(x, ord=None, …)` | `FlopscopeArray` | |
| `fnp.linalg.qr(a, mode="reduced")` | `tuple[FlopscopeArray, FlopscopeArray]` | |
| `fnp.linalg.svd(a, full_matrices=True, …)` | `tuple[FlopscopeArray, FlopscopeArray, FlopscopeArray] \| FlopscopeArray` | |
| `fnp.fft.fft(a, n=None, axis=-1, norm=None)` | `FlopscopeArray` | |
| `fnp.random.default_rng(seed=None)` | `Generator` | (numpy's `Generator`) |
| `fnp.random.symmetric(shape, …)` | `FlopscopeArray` | shape: `int \| Sequence[int]` |
| `fnp.histogram(a, bins=10, …)` | `tuple[FlopscopeArray, FlopscopeArray]` | |
| `fnp.allclose(a, b, **kw)` | `bool` | |
| `fnp.array_equal(a, b, **kw)` | `bool` | |
| `arr.sum(...)`, `.mean(...)`, `.astype(...)` etc. | `FlopscopeArray` | dunders too: `+ - * / @ == < & | <<` etc. |
| `with fnp.namespace(...)` | safe — `__exit__` declared `Literal[False]` so names bound inside stay bound after | |

When in doubt, prefer `flopscope.numpy.<X>` over `numpy.<X>` in code that hands
the result to a flopscope-typed API.
