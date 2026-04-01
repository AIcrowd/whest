# Exhaustive NumPy Coverage for mechestim

**Date:** 2026-04-01
**Status:** Draft
**Pinned NumPy version:** 2.1.x (latest stable in the 2.1 series)

## Problem

mechestim currently wraps 79 numpy functions (31 counted + 48 free) via hand-written code. The rest of numpy's ~300+ public functions hit a generic `AttributeError`. This creates two problems:

1. **Gaps by accident** — functions that *should* be available (e.g., `arcsin`, `nanmean`, `median`) are missing simply because nobody added them yet
2. **No systematic tracking** — there's no way to know what's covered vs. what's intentionally excluded vs. what was just forgotten

## Goals

- Every public callable in numpy 2.1.x is explicitly categorized: `counted`, `free`, `blacklisted`, or `unclassified`
- Zero `unclassified` entries allowed on the main branch (enforced by CI)
- The numpy backend version is visible in mechestim's version string and at runtime
- Wrappers inherit numpy's docstrings for discoverability
- A rich audit report makes coverage status easy to navigate

## Non-Goals

- Custom array classes or numpy subclassing
- Supporting multiple numpy versions simultaneously
- Auto-generating FLOP cost formulas (these remain hand-written)

---

## Design

### 1. The NumPy Registry (`src/mechestim/_registry.py`)

A Python dict mapping every public numpy function (pinned to 2.1.x) to a category.

**Categories:**
- `counted_unary` — unary math ops, cost = `numel(output)`
- `counted_binary` — binary math ops, cost = `numel(output)` after broadcast
- `counted_reduction` — reductions, cost = `numel(input)` * cost_multiplier
- `counted_custom` — ops with bespoke cost formulas (einsum, dot, linalg ops)
- `free` — zero FLOP cost (reshaping, indexing, type casting, creation)
- `blacklisted` — intentionally unsupported, with a reason
- `unclassified` — not yet triaged (temporary, blocked from main branch by CI)

**Entry format:**
```python
REGISTRY = {
    "exp": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Pointwise exponential. Cost: numel(output) FLOPs.",
    },
    "add": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise addition with broadcasting. Cost: numel(output) FLOPs.",
    },
    "sum": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Sum reduction. Cost: numel(input) FLOPs.",
    },
    "reshape": {
        "category": "free",
        "module": "numpy",
        "notes": "View-based shape change. No computation.",
    },
    "fft.fft": {
        "category": "blacklisted",
        "module": "numpy.fft",
        "notes": "O(n log n) cost model not yet implemented. Blacklisted to prevent untracked compute.",
    },
    "arcsin": {
        "category": "unclassified",
        "module": "numpy",
        "notes": "Needs triage — likely counted_unary.",
    },
}
```

**Registry metadata** (stored at the top of the file):
```python
REGISTRY_META = {
    "numpy_version": "2.1.3",  # exact version this was built against
    "generated_by": "scripts/numpy_audit.py",
    "last_updated": "2026-04-01",
}
```

**Key invariant:** Every public callable in numpy 2.1.x has an entry. The audit script + CI enforce this.

### 2. The Audit Script (`scripts/numpy_audit.py`)

A standalone script that introspects numpy and compares against the registry.

**What it scans:**
- `numpy` top-level namespace
- `numpy.linalg`
- `numpy.fft`
- `numpy.random`

**Explicitly excluded** (documented in the script):
- `numpy.testing` — test utilities, not math ops
- `numpy.lib` — internal helpers
- `numpy.dtypes` — type system internals
- `numpy.ma` — masked arrays (blacklisted as a whole submodule)
- `numpy.polynomial` — polynomial fitting (blacklisted)
- `numpy.strings` / `numpy.char` — string ops (blacklisted)
- `numpy.rec` — record arrays (blacklisted)
- `numpy.ctypeslib` — C interop (blacklisted)

**Introspection method:**
- Walk each scanned module's `dir()`, filter to public names (no leading `_`)
- Filter to callables (`inspect.isfunction`, `inspect.isbuiltin`, or numpy ufunc instances)
- Skip non-callable attributes (constants, types, modules handled separately)

**Coverage report — rich terminal table:**

Columns: Function | Module | Category | Status | Notes (truncated)

Color coding:
- Green — covered (in registry + wrapper implemented)
- Yellow — registered but not yet implemented
- Red — unclassified (in numpy but missing from registry)
- Gray/dim — blacklisted
- Magenta — stale (in registry but not in current numpy)

Rows grouped by submodule.

Summary footer:
```
Coverage: 79/312 implemented | 0 unclassified | 45 blacklisted | numpy 2.1.3
```

**CLI flags:**
- `--filter <category>` — show only specific categories (e.g., `--filter unclassified`)
- `--module <name>` — show only a specific submodule
- `--json` — machine-readable JSON output (no colors)
- `--ci` — plain text, exits non-zero if any unclassified entries exist

**Dependency:** `rich` added as optional dev dependency.

### 3. Version Identity & Runtime Banner

**Version string:**
- `me.__version__` returns `"0.2.0+np2.1.3"` (PEP 440 local version identifier)
- `me.__numpy_version__` returns the runtime numpy version (e.g., `"2.1.3"`)
- `me.__numpy_pinned__` returns the version from `REGISTRY_META["numpy_version"]`

**Runtime version check:**
- On import, mechestim compares installed numpy version against `__numpy_pinned__`
- If major.minor doesn't match, emits `MechEstimWarning: mechestim registry was built for numpy 2.1.3 but numpy {installed} is installed. Some functions may be missing or behave differently.`

**Runtime banner:**
- Printed to stderr when entering `BudgetContext`:
  ```
  mechestim 0.2.0 (numpy 2.1.3 backend) | budget: 1.00e+09 FLOPs
  ```
- Suppressible via `BudgetContext(budget, quiet=True)`

**pyproject.toml changes:**
- `numpy>=2.1.0,<2.2.0` (pin to 2.1.x compatible range)
- Version bumped to `0.2.0`
- `rich` added to `[project.optional-dependencies] dev`

### 4. Expanded Wrapper Architecture

The existing factory pattern is preserved and extended.

**New factories:**
- `_counted_unary_multi(np_func, op_name)` — for unary functions returning multiple arrays (e.g., `modf`, `frexp`). Cost = `numel(input)`.
- `_counted_binary_multi(np_func, op_name)` — binary ops returning tuples (e.g., `divmod`). Cost = `numel(output)`.

**Existing factories reused for:**
- Additional unary math (`arcsin`, `arccos`, `sinh`, `cosh`, `exp2`, `log1p`, `expm1`, `rint`, `trunc`, `degrees`, `radians`, etc.) — `_counted_unary`
- Additional binary ops (`fmod`, `remainder`, `heaviside`, `logaddexp`, `logaddexp2`, etc.) — `_counted_binary`
- Additional reductions (`any`, `all`, `nansum`, `nanmean`, `nanstd`, `nanmax`, `nanmin`, `median`, `percentile`, `count_nonzero`, etc.) — `_counted_reduction`
- Additional free ops (`rot90`, `fliplr`, `flipud`, `atleast_1d/2d/3d`, `column_stack`, `dstack`, `flatnonzero`, `nonzero`, `indices`, etc.) — manual passthrough

**Linear algebra (incremental):**
- Each linalg function gets a custom wrapper with a specific cost formula
- e.g., `linalg.solve` → `O(n^3)`, `linalg.norm` → reduction cost, `linalg.det` → `O(n^3)`
- Added one-by-one, staying `unclassified` until implemented

**Registry-driven `__getattr__`:**

Each module that exposes mechestim functions has its own `__getattr__` that consults the registry for its namespace. The top-level `mechestim/__init__.py` handles top-level numpy functions. `mechestim/linalg/__init__.py` handles `numpy.linalg` functions. `mechestim/fft/__init__.py` handles `numpy.fft` functions. Each follows the same logic:

- `blacklisted` → `AttributeError` with `notes` from registry
- `unclassified` → `AttributeError: "mechestim has not yet classified '{name}'. Please report this."`
- `counted`/`free` entries that aren't yet wrapped → `AttributeError: "'{name}' is registered but not yet implemented in mechestim."`

### 5. Docstring Inheritance

Every wrapper inherits the original numpy docstring, prefixed with mechestim metadata.

**Format:**
```
[mechestim] Cost: numel(output) FLOPs | Category: counted_unary

--- numpy docstring ---
{original numpy docstring}
```

**Implementation:**
- A helper function `_attach_docstring(wrapper, np_func, category, cost_description)` called in each factory
- Reads `np_func.__doc__` at import time
- For free ops: header says `Cost: 0 FLOPs | Category: free`
- If numpy's docstring is `None`, falls back to mechestim header only

### 6. Submodule Coverage

**`numpy.linalg`** — existing `me.linalg` expanded incrementally. Each function gets a custom cost formula in its own file under `linalg/`. `me.linalg.__getattr__` consults the registry.

**`numpy.random`** — existing passthrough. Audit confirms all functions are covered.

**`numpy.fft`** — stub module. `me.fft.__getattr__` gives clear blacklist errors from registry.

**Blacklisted entirely:**
- `numpy.ma` — masked arrays not needed for competition
- `numpy.polynomial` — polynomial fitting not needed
- `numpy.strings` / `numpy.char` — string operations
- `numpy.rec` — record arrays
- `numpy.ctypeslib` — C interop

**Excluded from registry** (not user-facing math):
- `numpy.testing`, `numpy.lib`, `numpy.dtypes`

Note: "blacklisted" submodules have individual function entries in the registry (so the audit tracks them and `__getattr__` gives specific error messages). "Excluded" submodules are not in the registry at all — they're filtered out during introspection because they aren't math operations.

The audit script maintains explicit lists of scanned and excluded submodules.

---

## Implementation Order

1. **Registry + audit script** — build the registry, run the audit, see the full gap report
2. **Version identity + runtime banner** — update pyproject.toml, add version attributes, banner
3. **Docstring inheritance** — add helper, retrofit existing wrappers
4. **Expand wrappers** — category by category, starting with the mechanical ones (unary, binary, free)
5. **Linalg expansion** — incrementally, with custom cost formulas
6. **CI integration** — audit script in CI, zero-unclassified gate on main

---

## Testing Strategy

- **Registry completeness test:** import numpy, walk public API, assert every function is in registry
- **Wrapper correctness tests:** for each new wrapper, test that (a) it produces same output as numpy, (b) it charges the correct FLOP cost
- **Version tests:** assert `__version__`, `__numpy_version__`, `__numpy_pinned__` are well-formed
- **Banner test:** assert BudgetContext prints banner to stderr (and `quiet=True` suppresses it)
- **Audit script test:** run audit in `--ci` mode, assert exit code 0
