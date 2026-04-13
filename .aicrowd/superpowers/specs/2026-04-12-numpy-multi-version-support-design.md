# Multi-Version NumPy Support for Mechestim

**Date:** 2026-04-12
**Status:** Approved
**Scope:** Widen numpy dependency from `>=2.1.0,<2.2.0` to `>=2.0.0,<2.3.0`

## Context

Mechestim wraps ~482 NumPy functions with FLOP-counting instrumentation. The project currently pins numpy tightly to `>=2.1.0,<2.2.0` because the operation registry was built against numpy 2.1.3.

An audit of numpy 2.0.2, 2.1.3, and 2.2.6 shows the public API is purely additive across these versions — no functions were removed. Only 7 functions differ:

| Function | Available from | Category |
|---|---|---|
| `cumulative_sum` | 2.1+ | counted_reduction |
| `cumulative_prod` | 2.1+ | counted_reduction |
| `unstack` | 2.1+ | counted_custom |
| `bitwise_count` | 2.1+ | counted_unary |
| `vecdot` | 2.1+ | counted_binary |
| `matvec` | 2.2+ | counted_binary |
| `vecmat` | 2.2+ | counted_binary |

All other 477 functions work identically across 2.0, 2.1, and 2.2.

## Decisions

- **Supported range:** `numpy >=2.0.0, <2.3.0`
- **Default version:** numpy 2.2 (what `pip install whest` resolves to)
- **Tested versions in CI:** 2.0.2, 2.1.3, 2.2.6
- **Unavailable function behavior:** Error at call time (not import time) with actionable message telling the user which numpy version to upgrade to
- **Approach:** Conditional `hasattr` guards at function definition sites (same pattern as existing `ptp` guard)

## 1. Conditional Guard Pattern

When a function doesn't exist in the running numpy, define a stub that raises `UnsupportedFunctionError` at call time. Import always succeeds — only calling the function raises.

### Error class

New `UnsupportedFunctionError(Exception)` in a shared exceptions module (alongside existing `MechEstimWarning`). Carries `func_name` and `min_version` attributes.

### Error message format

```
whest.UnsupportedFunctionError: numpy.matvec requires numpy >= 2.2
(you have numpy 2.1.3). To use it: uv pip install 'numpy>=2.2'
```

### Guard pattern

```python
if hasattr(_np, "matvec"):
    matvec = _counted_binary(_np.matvec, "matvec")
else:
    def matvec(*args, **kwargs):
        raise UnsupportedFunctionError("matvec", min_version="2.2")
```

### Functions requiring guards

**In `_pointwise.py`:**
- `bitwise_count` (min 2.1)
- `cumulative_sum` (min 2.1)
- `cumulative_prod` (min 2.1)
- `vecdot` (min 2.1)
- `matvec` (min 2.2, new implementation)
- `vecmat` (min 2.2, new implementation)

**In `_free_ops.py`:**
- `unstack` (min 2.1)

## 2. New Function Implementations

### `matvec` and `vecmat`

Both are ufuncs added in numpy 2.2 for matrix-vector and vector-matrix products.

**Semantics:**
- `matvec(A, v)`: A is `(..., m, n)`, v is `(..., n)`, result is `(..., m)`. Equivalent to `A @ v`.
- `vecmat(v, A)`: v is `(..., n)`, A is `(..., n, m)`, result is `(..., m)`. Equivalent to `v @ A`.

**Cost formula:** `result.size * contracted_axis_size` — same structure as `vecdot`. Each output element is a dot product of length n along the contracted axis.

**Weight:** 1.0 (same as all dot/matmul family ops in empirical weights data).

**Implementation:** Custom functions following the `vecdot` pattern:
- `matvec(A, v)`: compute result via `_np.matvec`, deduct `result.size * A.shape[-1]` FLOPs (contracted over A's last axis).
- `vecmat(v, A)`: compute result via `_np.vecmat`, deduct `result.size * v.shape[-1]` FLOPs (contracted over v's last axis).

No `linalg.matvec`/`linalg.vecmat` aliases exist in numpy 2.2 — only top-level.

## 3. Registry Changes

### New entries in `_registry.py`

```python
"matvec": {
    "category": "counted_binary",
    "module": "numpy",
    "min_numpy": "2.2",
    "notes": "Matrix-vector product. Cost = output_size * contracted_axis.",
},
"vecmat": {
    "category": "counted_binary",
    "module": "numpy",
    "min_numpy": "2.2",
    "notes": "Vector-matrix product. Cost = output_size * contracted_axis.",
},
```

### `min_numpy` metadata field

Add `"min_numpy"` to the 7 version-gated registry entries. This is informational metadata — runtime wiring uses `hasattr` guards, not registry lookups. Useful for the audit script and documentation generation.

### Registry metadata update

```python
REGISTRY_META = {
    "numpy_version": "2.2.6",    # reference version (was "2.1.3")
    "numpy_supported": ">=2.0.0,<2.3.0",  # new field
    "last_updated": "2026-04-12",
}
```

### weights.json

Add entries: `"matvec": 1.0`, `"vecmat": 1.0`.

## 4. Version Check Changes

### `_version_check.py`

Replace exact major.minor comparison with a range check. Warn if installed numpy is outside `[2.0, 2.3)`, not if it differs from one pinned version.

### `__init__.py`

- Keep `__numpy_version__` = installed numpy version (dynamic)
- Add `__numpy_supported__` = `">=2.0.0,<2.3.0"` (from registry metadata)
- `__numpy_pinned__` updated to `"2.2.6"` (reference version)

### `pyproject.toml` (root + whest-server)

Change `"numpy>=2.1.0,<2.2.0"` to `"numpy>=2.0.0,<2.3.0"`.

## 5. CI and Testing

### CI matrix (`.github/workflows/ci.yml`)

Add numpy version dimension:

```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
    numpy-version: ["2.0", "2.1", "2.2"]
```

After `uv sync`, override: `uv pip install 'numpy~={numpy-version}.0'`

### Core tests

All 1699 existing tests should pass on all three numpy versions. Tests for version-gated functions use skip markers:

```python
@pytest.mark.skipif(not hasattr(np, "matvec"), reason="requires numpy >= 2.2")
def test_matvec_cost(): ...
```

### Compat tests (`tests/numpy_compat/`)

Make xfails version-aware. Change `xfails.py` from a flat list to a structure keyed by numpy minor version range, so different versions can have different expected failures.

### New tests

- Verify `UnsupportedFunctionError` is raised with correct message when calling a stub
- Verify `matvec`/`vecmat` cost accounting matches expected formula on numpy 2.2+

## 6. Documentation

### Installation docs

Update `docs/getting-started/installation.md`:
- Document supported range `>=2.0.0,<2.3.0`
- Default install gets numpy 2.2
- Users can pin lower: `uv pip install 'whest' 'numpy>=2.0,<2.1'`
- Note which functions are unavailable on older versions

### Changelog

Document the version range expansion as a feature.

## Files Changed

| File | Change |
|---|---|
| `pyproject.toml` | Widen numpy pin |
| `whest-server/pyproject.toml` | Widen numpy pin |
| `src/whest/_registry.py` | Add `matvec`/`vecmat`, add `min_numpy` fields, update meta |
| `src/whest/_version_check.py` | Range check instead of exact match |
| `src/whest/__init__.py` | Add `__numpy_supported__`, update `__numpy_pinned__` |
| `src/whest/_pointwise.py` | Add guards for 6 functions, implement `matvec`/`vecmat` |
| `src/whest/_free_ops.py` | Add guard for `unstack` |
| `src/whest/data/weights.json` | Add `matvec`, `vecmat` entries |
| `src/whest/data/weights.csv` | Add `matvec`, `vecmat` entries |
| `.github/workflows/ci.yml` | Add numpy version matrix |
| `tests/` | Add version-skip markers, stub error tests, matvec/vecmat cost tests |
| `tests/numpy_compat/xfails.py` | Make version-aware |
| `docs/getting-started/installation.md` | Update version docs |
| `docs/changelog.md` | Document feature |
| New: `src/whest/_exceptions.py` | `UnsupportedFunctionError` class |
