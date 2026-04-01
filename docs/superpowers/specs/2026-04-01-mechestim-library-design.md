# mechestim Library Specification

> **Status:** Draft
> **Date:** 2026-04-01
> **Author:** S.P. Mohanty / AIcrowd
> **Stakeholders:** Paul Christiano, Jacob Hilton, Wilson Wu (ARC)

## 1. Purpose

mechestim is a Python library providing a constrained set of mathematical primitives with analytical FLOP counting for ARC's Mechanistic Estimation Challenge. The challenge asks participants to estimate the expected output of randomly initialized ReLU MLPs more efficiently than brute-force sampling.

The library exists to make the competition about **algorithmic innovation** rather than **performance engineering**. It does this by:

1. Providing a controlled set of operations participants must use
2. Counting FLOPs analytically for each operation (decoupled from actual runtime)
3. Enforcing a computational budget so participants optimize algorithms, not code

## 2. Design Principles

- **NumPy-native:** All tensors are plain `numpy.ndarray`. No custom array class.
- **NumPy-familiar API:** Function signatures mirror NumPy/SciPy where possible.
- **Analytical FLOP counting:** FLOP cost is computed from shapes, not measured from execution. The cost model is deterministic and hardware-independent.
- **Single context model:** Everything runs inside a `BudgetContext`. Non-compute ops (reshape, index, etc.) have zero FLOP cost. No separate precomputation phase.
- **Documentation-first:** Every public function has a comprehensive NumPy-style docstring. Professional documentation site auto-generated from docstrings.
- **Transparent diagnostics:** Op log and FLOP breakdown always available, same behavior locally and in remote evaluation.

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     PARTICIPANT CODE                           │
│                                                                │
│  with mechestim.BudgetContext(flop_budget=N) as budget:        │
│      # Free ops: tensor creation, reshaping, indexing (0 FLOP) │
│      # Counted ops: einsum, pointwise math, SVD (deduct FLOPs) │
│      result = my_estimator(weights, budget)                    │
│      print(budget.summary())                                   │
├────────────────────────────────────────────────────────────────┤
│                     mechestim (public API)                      │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Counted Ops   │  │ Free Ops     │  │ Budget + Diagnostics │ │
│  │              │  │              │  │                      │ │
│  │ einsum       │  │ zeros, ones  │  │ BudgetContext        │ │
│  │ exp,log,max  │  │ eye, diag    │  │ budget.flops_used    │ │
│  │ add,mul,div  │  │ reshape      │  │ budget.flops_remaining│ │
│  │ sum,mean     │  │ transpose    │  │ budget.op_log        │ │
│  │ svd          │  │ concatenate  │  │ budget.summary()     │ │
│  │              │  │ stack, split │  │                      │ │
│  └──────┬───────┘  │ copy, where  │  └──────────────────────┘ │
│         │          │ astype, etc. │                            │
│         ▼          └──────┬───────┘                            │
│  ┌──────────────┐         │                                    │
│  │ FLOP Counter  │◀───────┘ (cost = 0)                        │
│  │              │                                              │
│  │ Per-op cost  │                                              │
│  │ calculators  │                                              │
│  └──────┬───────┘                                              │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────┐                                              │
│  │ NumPy backend │                                              │
│  │ (np.einsum,  │                                              │
│  │  np.exp, etc)│                                              │
│  └──────────────┘                                              │
└────────────────────────────────────────────────────────────────┘
```

### 3.1 Data Flow for Counted Operations

```
mechestim.einsum(subscripts, *operands)
    │
    ├── 1. Validate inputs (types, shapes, subscript syntax)
    │       └── Raise TypeError / ValueError on failure
    │
    ├── 2. Calculate FLOP cost (shape-based, analytical)
    │       ├── Parse subscript → index labels + dimensions
    │       ├── Detect repeated operands (Python `is` check)
    │       ├── Apply symmetry savings if applicable
    │       └── Return integer FLOP count
    │
    ├── 3. Check budget (cost <= remaining?)
    │       └── Raise BudgetExhaustedError if exceeded
    │
    ├── 4. Execute via NumPy backend (np.einsum)
    │
    ├── 5. Check result for NaN/Inf → warn if present
    │
    ├── 6. Record in op_log (OpRecord)
    │
    └── 7. Deduct cost from budget, return result (ndarray)
```

### 3.2 Data Flow for Free Operations

```
mechestim.reshape(x, shape)
    │
    ├── 1. Validate inputs
    │
    ├── 2. Execute via NumPy backend (np.reshape)
    │
    └── 3. Return result (ndarray) — no budget interaction
```

## 4. Counted Operations

### 4.1 einsum

```python
mechestim.einsum(subscripts: str, *operands: np.ndarray, symmetric_dims: list[tuple[int, ...]] | None = None) -> np.ndarray
```

The primary workhorse operation. Wraps `np.einsum`.

**FLOP cost formula:**

1. Parse the subscript string to identify all index labels
2. For each index label, determine its dimension from operand shapes
3. Base cost = product of ALL index dimensions (contracted + output)
4. Each multiply-add counts as 1 FLOP (configurable via `BudgetContext.flop_multiplier`)

**Symmetry savings (repeated operands):**

When the same ndarray object (checked via Python `is`) is passed as multiple operands, and the einsum expression is symmetric with respect to those operands, the cost is divided by `k!` where `k` is the number of times that operand appears.

Detection: For operands that share the same `id()`, the library checks whether swapping their positions in the subscript produces an equivalent expression. If so, the symmetry saving applies.

**Symmetry savings (explicit `symmetric_dims`):**

`symmetric_dims=[(0,1), (2,3)]` declares that the output tensor has symmetry groups: dims 0 and 1 are interchangeable, and dims 2 and 3 are interchangeable. Cost is divided by the product of factorials of group sizes (`2! * 2! = 4`).

When `symmetric_dims` is provided, the library validates the claim by checking `np.allclose(result, result.transpose(perm), atol=1e-6, rtol=1e-5)` for the relevant permutations. Raises `SymmetryError` if the claim is invalid.

**Diagonal/repeated-index patterns:**

Expressions like `'i->ii'` (diagonal embedding) are not supported by `np.einsum` natively. mechestim implements these via `np.diag` or equivalent, with cost = numel(output).

**Examples:**

```python
# Standard matmul: cost = m * k * n
mechestim.einsum('ij,jk->ik', A, B)  # A is (m,k), B is (k,n)

# Symmetric contraction: x passed twice, cost = (a*b*i*j) / 2!
mechestim.einsum('ai,bj,ab->', x, x, A)  # x is same object

# Batched: cost = a*b*c*i*j*k / 3! (x passed 3 times)
mechestim.einsum('ai,bj,ck,abc->ijk', x, x, x, A)

# Trace: cost = n
mechestim.einsum('ii->', A)

# With explicit symmetric output dims: cost reduced by 2!
mechestim.einsum('ai,bj,ab->ij', x, y, A, symmetric_dims=[(0,1)])
```

### 4.2 Pointwise Operations

Element-wise operations. Cost = `numel(output)` (1 FLOP per output element).

For binary ops, output shape follows NumPy broadcasting rules.

| Function | Signature | NumPy equivalent |
|---|---|---|
| `mechestim.exp(x)` | `(ndarray) -> ndarray` | `np.exp` |
| `mechestim.log(x)` | `(ndarray) -> ndarray` | `np.log` |
| `mechestim.abs(x)` | `(ndarray) -> ndarray` | `np.abs` |
| `mechestim.negative(x)` | `(ndarray) -> ndarray` | `np.negative` |
| `mechestim.sqrt(x)` | `(ndarray) -> ndarray` | `np.sqrt` |
| `mechestim.square(x)` | `(ndarray) -> ndarray` | `np.square` |
| `mechestim.add(x, y)` | `(ndarray, ndarray) -> ndarray` | `np.add` |
| `mechestim.subtract(x, y)` | `(ndarray, ndarray) -> ndarray` | `np.subtract` |
| `mechestim.multiply(x, y)` | `(ndarray, ndarray) -> ndarray` | `np.multiply` |
| `mechestim.divide(x, y)` | `(ndarray, ndarray) -> ndarray` | `np.divide` |
| `mechestim.maximum(x, y)` | `(ndarray, ndarray) -> ndarray` | `np.maximum` |
| `mechestim.minimum(x, y)` | `(ndarray, ndarray) -> ndarray` | `np.minimum` |
| `mechestim.power(x, y)` | `(ndarray, ndarray) -> ndarray` | `np.power` |
| `mechestim.clip(x, a_min, a_max)` | `(ndarray, scalar, scalar) -> ndarray` | `np.clip` |

### 4.3 Reductions

Cost = `numel(input)` (must scan all input elements).

| Function | Signature | NumPy equivalent |
|---|---|---|
| `mechestim.sum(x, axis=None)` | `(ndarray, axis?) -> ndarray` | `np.sum` |
| `mechestim.max(x, axis=None)` | `(ndarray, axis?) -> ndarray` | `np.max` |
| `mechestim.min(x, axis=None)` | `(ndarray, axis?) -> ndarray` | `np.min` |
| `mechestim.mean(x, axis=None)` | `(ndarray, axis?) -> ndarray` | `np.mean` |
| `mechestim.prod(x, axis=None)` | `(ndarray, axis?) -> ndarray` | `np.prod` |

`mechestim.mean` costs `numel(input) + numel(output)` (sum + divide).

### 4.4 Linear Algebra: SVD

```python
mechestim.svd(A: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Truncated singular value decomposition.

**FLOP cost:** `m * n * k` where `A` is `(m, n)`. If `k is None`, `k = min(m, n)`.

**Returns:** `(U, S, Vt)` where:
- `U` is `(m, k)` — left singular vectors
- `S` is `(k,)` — singular values (descending order)
- `Vt` is `(k, n)` — right singular vectors (transposed)

**Backend:** `np.linalg.svd(A, full_matrices=False)` truncated to top-k components.

**Constraints:**
- Input must be 2D
- `k` must satisfy `1 <= k <= min(m, n)`

### 4.5 FLOP Cost Summary

```
OPERATION                       | FLOP COST
--------------------------------|------------------------------------------
einsum(subscripts, *ops)        | product(all index dims) / symmetry_factor
exp, log, abs, neg, sqrt, sq    | numel(input)
add, sub, mul, div, max, min    | numel(output)  [broadcast-aware]
power, clip                     | numel(output)
sum, max, min, prod (reduction) | numel(input)
mean (reduction)                | numel(input) + numel(output)
svd(A, k)                       | m * n * k
Free ops (reshape, etc.)        | 0
```

All costs are multiplied by `BudgetContext.flop_multiplier` (default 1).

### 4.6 FLOP Cost Query API

Each operation has a corresponding cost calculator that can be called independently for planning:

```python
import mechestim.flops as flops

# Query costs without executing
cost = flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
cost = flops.svd_cost(m=256, n=256, k=10)
cost = flops.pointwise_cost(shape=(256, 256))
cost = flops.reduction_cost(input_shape=(256, 256), axis=1)
```

These functions take shapes (not data), return an integer FLOP count, and do not require a `BudgetContext`. They are useful for participants planning their FLOP budget allocation.

## 5. Free Operations (Zero FLOP Cost)

These operations are provided because raw NumPy is disabled in the evaluation sandbox. They wrap NumPy equivalents with no FLOP cost. They do NOT require a `BudgetContext` — they work anywhere.

### 5.1 Tensor Creation

| Function | Signature | NumPy equivalent |
|---|---|---|
| `mechestim.array(data, dtype=None)` | `(array_like, dtype?) -> ndarray` | `np.array` |
| `mechestim.zeros(shape, dtype=float)` | `(shape, dtype?) -> ndarray` | `np.zeros` |
| `mechestim.ones(shape, dtype=float)` | `(shape, dtype?) -> ndarray` | `np.ones` |
| `mechestim.full(shape, fill, dtype=None)` | `(shape, scalar, dtype?) -> ndarray` | `np.full` |
| `mechestim.eye(n, m=None, dtype=float)` | `(int, int?, dtype?) -> ndarray` | `np.eye` |
| `mechestim.diag(v, k=0)` | `(ndarray, int?) -> ndarray` | `np.diag` |
| `mechestim.arange(*args, dtype=None)` | `(start?, stop, step?, dtype?) -> ndarray` | `np.arange` |
| `mechestim.linspace(start, stop, num=50)` | `(scalar, scalar, int?) -> ndarray` | `np.linspace` |
| `mechestim.zeros_like(x)` | `(ndarray) -> ndarray` | `np.zeros_like` |
| `mechestim.ones_like(x)` | `(ndarray) -> ndarray` | `np.ones_like` |
| `mechestim.empty(shape, dtype=float)` | `(shape, dtype?) -> ndarray` | `np.empty` |

### 5.2 Tensor Manipulation

| Function | Signature | NumPy equivalent |
|---|---|---|
| `mechestim.reshape(x, shape)` | `(ndarray, shape) -> ndarray` | `np.reshape` |
| `mechestim.transpose(x, axes=None)` | `(ndarray, axes?) -> ndarray` | `np.transpose` |
| `mechestim.concatenate(arrays, axis=0)` | `(list[ndarray], int?) -> ndarray` | `np.concatenate` |
| `mechestim.stack(arrays, axis=0)` | `(list[ndarray], int?) -> ndarray` | `np.stack` |
| `mechestim.split(x, indices, axis=0)` | `(ndarray, int/list, int?) -> list` | `np.split` |
| `mechestim.squeeze(x, axis=None)` | `(ndarray, axis?) -> ndarray` | `np.squeeze` |
| `mechestim.expand_dims(x, axis)` | `(ndarray, int) -> ndarray` | `np.expand_dims` |
| `mechestim.copy(x)` | `(ndarray) -> ndarray` | `np.copy` |
| `mechestim.where(cond, x, y)` | `(ndarray, ndarray, ndarray) -> ndarray` | `np.where` |
| `mechestim.tile(x, reps)` | `(ndarray, int/tuple) -> ndarray` | `np.tile` |
| `mechestim.repeat(x, repeats, axis=None)` | `(ndarray, int, int?) -> ndarray` | `np.repeat` |
| `mechestim.flip(x, axis=None)` | `(ndarray, int?) -> ndarray` | `np.flip` |
| `mechestim.sort(x, axis=-1)` | `(ndarray, int?) -> ndarray` | `np.sort` |
| `mechestim.argsort(x, axis=-1)` | `(ndarray, int?) -> ndarray` | `np.argsort` |

### 5.3 Type and Info

| Function | Signature | NumPy equivalent |
|---|---|---|
| `mechestim.astype(x, dtype)` | `(ndarray, dtype) -> ndarray` | `x.astype(dtype)` |

**Indexing/slicing:** `x[i, j]`, `x[:, 0]`, `x[mask]`, etc. work natively on ndarrays. No wrapping needed.

**ndarray attributes:** `.shape`, `.dtype`, `.ndim`, `.size`, `.T` all work natively.

## 6. Budget API

### 6.1 BudgetContext

```python
class BudgetContext:
    """Context manager for FLOP budget enforcement.

    All counted mechestim operations must be called within an active
    BudgetContext. Free operations (tensor creation, reshaping) work
    both inside and outside the context.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed. Must be > 0.
    flop_multiplier : float, optional
        Multiplier applied to all FLOP costs. Default 1 (1 FLOP per
        multiply-add). Set to 2 for strict IEEE FLOP counting.

    Raises
    ------
    ValueError
        If flop_budget <= 0.
    RuntimeError
        If a BudgetContext is already active (nesting not allowed).

    Examples
    --------
    >>> with mechestim.BudgetContext(flop_budget=1_000_000) as budget:
    ...     x = mechestim.ones((256, 256))  # free
    ...     y = mechestim.einsum('ij,jk->ik', x, x)  # costs 256^3 FLOPs
    ...     print(budget.flops_used)
    16777216
    """
```

### 6.2 Budget Introspection

| Attribute / Method | Type | Description |
|---|---|---|
| `budget.flop_budget` | `int` | Original budget |
| `budget.flops_used` | `int` | Cumulative FLOPs consumed |
| `budget.flops_remaining` | `int` | `flop_budget - flops_used` |
| `budget.flop_multiplier` | `float` | Cost multiplier |
| `budget.op_log` | `list[OpRecord]` | Every counted op executed |
| `budget.summary()` | `str` | Pretty-printed breakdown |

### 6.3 OpRecord

```python
from typing import NamedTuple

class OpRecord(NamedTuple):
    """Record of a single counted operation."""
    op_name: str          # e.g., "einsum", "exp", "svd"
    subscripts: str | None  # einsum subscript string, None for other ops
    shapes: tuple         # input shapes
    flop_cost: int        # FLOPs charged for this op
    cumulative: int       # cumulative FLOPs after this op
```

### 6.4 Budget Summary Format

```
mechestim FLOP Budget Summary
==============================
  Total budget:      1,000,000
  Used:                750,432  (75.0%)
  Remaining:           249,568  (25.0%)

  By operation:
    einsum           720,000  (96.0%)  [142 calls]
    exp               15,360  ( 2.0%)  [60 calls]
    svd               15,072  ( 2.0%)  [3 calls]
```

## 7. Error Handling

### 7.1 Exception Hierarchy

```python
class MechEstimError(Exception):
    """Base exception for all mechestim errors."""

class BudgetExhaustedError(MechEstimError):
    """Raised when an operation would exceed the FLOP budget.

    Attributes
    ----------
    op_name : str
        Name of the operation that exceeded the budget.
    flop_cost : int
        FLOPs the operation would have cost.
    flops_remaining : int
        FLOPs remaining in the budget.
    """

class NoBudgetContextError(MechEstimError):
    """Raised when a counted operation is called outside a BudgetContext."""

class SymmetryError(MechEstimError):
    """Raised when a claimed tensor symmetry does not hold.

    Attributes
    ----------
    dims : tuple[int, ...]
        The dimension group that failed validation.
    max_deviation : float
        Maximum element-wise deviation from symmetry.
    """
```

### 7.2 Error Behavior

| Situation | Exception | Message pattern |
|---|---|---|
| Counted op called outside BudgetContext | `NoBudgetContextError` | "No active BudgetContext. Wrap your code in `with mechestim.BudgetContext(...):`" |
| Op would exceed budget | `BudgetExhaustedError` | "einsum would cost 16,777,216 FLOPs but only 1,000 remain" |
| Invalid einsum subscripts | `ValueError` | Delegates to NumPy's error message |
| Shape mismatch | `ValueError` | Delegates to NumPy's error message |
| Non-ndarray to counted op | `TypeError` | "Expected numpy.ndarray, got list" |
| Claimed symmetry invalid | `SymmetryError` | "Tensor not symmetric along dims (0, 1): max deviation = 0.5" |
| Nested BudgetContext | `RuntimeError` | "Cannot nest BudgetContexts" |
| Budget <= 0 | `ValueError` | "flop_budget must be > 0, got -5" |

### 7.3 NaN/Inf Warnings

When any counted operation produces NaN or Inf values in its result, mechestim issues a Python warning via `warnings.warn()`:

```
MechEstimWarning: einsum produced 3 NaN and 0 Inf values in output of shape (256, 256)
```

This does NOT raise an exception. Participants can suppress warnings with `warnings.filterwarnings('ignore', category=MechEstimWarning)`.

## 8. Documentation Requirements

### 8.1 Docstring Standard

Every public function and class uses NumPy-style docstrings with these sections:

```python
def einsum(subscripts, *operands, symmetric_dims=None):
    """Evaluate an Einstein summation convention on the operands.

    Wraps ``numpy.einsum`` with analytical FLOP counting and budget
    enforcement. See ``numpy.einsum`` for full subscript syntax.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as a comma-separated
        list of subscript labels. An implicit (classical Einstein
        summation) calculation is performed unless the explicit
        indicator '->' is included.
    *operands : numpy.ndarray
        The arrays for the operation.
    symmetric_dims : list of tuple of int, optional
        Declares symmetry groups in the output tensor. Each tuple
        lists dimensions that are interchangeable. The FLOP cost
        is reduced by the product of factorials of group sizes.
        The claim is validated at runtime.

    Returns
    -------
    numpy.ndarray
        The calculation result.

    Raises
    ------
    BudgetExhaustedError
        If the operation would exceed the remaining FLOP budget.
    NoBudgetContextError
        If called outside a ``BudgetContext``.
    SymmetryError
        If ``symmetric_dims`` is provided but the result is not
        actually symmetric along those dimensions.
    ValueError
        If ``subscripts`` is invalid or shapes are incompatible.
    TypeError
        If any operand is not a ``numpy.ndarray``.

    Notes
    -----
    **FLOP cost:** Product of all index dimensions in the subscript.
    For example, ``'ij,jk->ik'`` with shapes ``(m, k)`` and ``(k, n)``
    costs ``m * k * n`` FLOPs.

    **Symmetry savings from repeated operands:** When the same array
    object is passed multiple times (checked via Python ``is``), and
    the expression is symmetric with respect to those operands, the
    cost is divided by ``k!`` where ``k`` is the repeat count.

    **Symmetry savings from symmetric_dims:** Cost is further divided
    by the product of factorials of each symmetry group size.

    Examples
    --------
    Standard matrix multiplication:

    >>> with mechestim.BudgetContext(flop_budget=10**8) as budget:
    ...     A = mechestim.ones((256, 128))
    ...     B = mechestim.ones((128, 64))
    ...     C = mechestim.einsum('ij,jk->ik', A, B)
    ...     assert budget.flops_used == 256 * 128 * 64

    Symmetric contraction (same operand passed twice):

    >>> with mechestim.BudgetContext(flop_budget=10**8) as budget:
    ...     x = mechestim.ones((10, 256))
    ...     A = mechestim.ones((10, 10))
    ...     result = mechestim.einsum('ai,bi,ab->', x, x, A)
    ...     # Cost = (10 * 10 * 256) / 2! = 12800
    """
```

### 8.2 Documentation Site

**Tool:** mkdocs with mkdocstrings (Python handler) and the Material theme.

**Structure:**
```
docs/
├── index.md                    # Getting Started
├── quickstart.md               # Installation + first example
├── concepts/
│   ├── flop-counting.md        # How FLOP costs work
│   ├── budget-context.md       # Budget enforcement model
│   └── symmetry.md             # Symmetry detection and savings
├── api/                        # Auto-generated from docstrings
│   ├── counted-ops.md          # einsum, pointwise, svd
│   ├── free-ops.md             # tensor creation, manipulation
│   ├── budget.md               # BudgetContext, OpRecord
│   ├── flops.md                # Cost query API
│   └── errors.md               # Exception classes
├── examples/
│   ├── meanprop.md             # Full meanprop walkthrough
│   └── flop-planning.md        # Using cost query API for planning
└── changelog.md
```

**Hosting:** GitHub Pages, auto-deployed from main branch via GitHub Actions.

**mkdocs.yml configuration:**

```yaml
site_name: mechestim
theme:
  name: material
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: true
```

## 9. Module Structure

```
mechestim/
├── pyproject.toml           # Package config (uv/pip compatible)
├── README.md                # Overview + installation + quick example
├── mkdocs.yml               # Documentation site config
├── src/
│   └── mechestim/
│       ├── __init__.py      # Public API: re-exports all public functions
│       ├── _budget.py       # BudgetContext, OpRecord, budget state
│       ├── _flops.py        # FLOP cost calculators (public: mechestim.flops)
│       ├── _einsum.py       # einsum: parsing, symmetry detection, execution
│       ├── _pointwise.py    # Pointwise ops and reductions
│       ├── _linalg.py       # SVD (and future linalg ops)
│       ├── _free_ops.py     # Zero-cost tensor creation/manipulation
│       ├── _validation.py   # Input validation, NaN/Inf checking
│       ├── errors.py        # Exception classes (public)
│       ├── flops.py         # Cost query API (public: mechestim.flops)
│       └── py.typed         # PEP 561 marker for type checkers
├── tests/
│   ├── test_einsum.py       # Einsum ops + symmetry
│   ├── test_pointwise.py    # Pointwise + reductions
│   ├── test_linalg.py       # SVD
│   ├── test_budget.py       # BudgetContext behavior
│   ├── test_flops.py        # FLOP cost model correctness
│   ├── test_free_ops.py     # Free ops pass-through
│   ├── test_errors.py       # Error conditions
│   └── test_integration.py  # End-to-end estimator flows
└── docs/                    # mkdocs source
    └── ...
```

**src layout:** Uses the `src/` layout per Python packaging best practices. Prevents accidental imports from the source directory during testing.

## 10. Testing Strategy

### 10.1 FLOP Cost Model Tests

The most critical test suite. For every counted operation, verify that the FLOP cost matches the analytical prediction exactly.

```python
# test_flops.py examples:

# einsum: product of all index dims
assert einsum_cost('ij,jk->ik', [(3,4), (4,5)]) == 3 * 4 * 5

# einsum: trace
assert einsum_cost('ii->', [(10,10)]) == 10

# einsum: batch matmul
assert einsum_cost('bij,bjk->bik', [(2,3,4), (2,4,5)]) == 2 * 3 * 4 * 5

# einsum: symmetry with repeated operand (same object)
# cost should be halved
assert einsum_cost_with_symmetry('ai,bi,ab->', [(10,256), 'same', (10,10)]) == (10 * 10 * 256) // 2

# svd: m * n * k
assert svd_cost(m=100, n=50, k=10) == 100 * 50 * 10

# pointwise: numel
assert pointwise_cost((256, 256)) == 256 * 256

# reduction: numel of input
assert reduction_cost((256, 256)) == 256 * 256
```

### 10.2 Budget Enforcement Tests

```python
# Budget exactly sufficient
with BudgetContext(flop_budget=256**3) as budget:
    mechestim.einsum('ij,jk->ik', A_256, B_256)  # succeeds

# Budget insufficient
with pytest.raises(BudgetExhaustedError):
    with BudgetContext(flop_budget=100) as budget:
        mechestim.einsum('ij,jk->ik', A_256, B_256)

# Outside context
with pytest.raises(NoBudgetContextError):
    mechestim.einsum('ij,jk->ik', A, B)

# Nested context
with pytest.raises(RuntimeError):
    with BudgetContext(1000):
        with BudgetContext(500):
            pass
```

### 10.3 Numerical Correctness Tests

Every counted operation must produce the same numerical result as its NumPy equivalent:

```python
# For every op, verify: mechestim.op(x) == np.op(x)
with BudgetContext(10**9) as budget:
    assert np.allclose(mechestim.exp(x), np.exp(x))
    assert np.allclose(mechestim.einsum('ij,jk->ik', A, B), np.einsum('ij,jk->ik', A, B))
    U, S, Vt = mechestim.svd(A, k=5)
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    assert np.allclose(S, S_np[:5])
```

### 10.4 Integration Tests

Run ARC's reference algorithms (meanprop at minimum) end-to-end through mechestim:
- Verify the estimator completes within a reasonable FLOP budget
- Verify numerical output matches direct NumPy computation
- Verify the op_log correctly records all operations

## 11. Packaging and Distribution

### 11.1 pyproject.toml

```toml
[project]
name = "mechestim"
version = "0.1.0"
description = "Constrained mathematical primitives with FLOP counting for the Mechanistic Estimation Challenge"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
]
docs = [
    "mkdocs-material",
    "mkdocstrings[python]",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mechestim"]
```

### 11.2 Installation

```bash
# From GitHub (primary distribution)
pip install git+https://github.com/AIcrowd/mechestim.git

# For development
git clone https://github.com/AIcrowd/mechestim.git
cd mechestim
uv sync --all-extras
```

### 11.3 Versioning

- **v0.1.0:** Warm-up round. API may change based on participant feedback.
- **v0.x.y:** Adding new ops is a minor version bump. Changing FLOP cost models is a minor version bump with changelog.
- **v1.0.0:** Stable API for main competition round.

## 12. Not in Scope (v1)

| Item | Rationale |
|------|-----------|
| Sparse matmul | Paul hopes it's not needed. Add if requested during warm-up. |
| PyTorch/JAX backend | NumPy only. CPU-only evaluation. |
| GPU support | CPU-only evaluation environment. |
| Custom tensor class | Plain ndarrays. Sandbox import restrictions handle enforcement. |
| Contraction path optimization | FLOP count is canonical, not runtime-optimized. |
| Docker sandbox enforcement | Evaluation harness concern, not library concern. |
| Cholesky, eigendecomposition, FFT | Not in initial op set. Add if requested. |
| Evaluation harness integration | Separate repo. mechestim is standalone. |

## 13. Open Items for ARC Confirmation

1. **Einsum symmetry detection algorithm:** The exact algorithm for detecting when repeated operands produce symmetric savings needs validation against ARC's reference implementations. Paul and Wilson Wu's discussion about partially symmetric tensors should inform this.

2. **SVD cost constant:** Currently specified as exactly `m * n * k`. Paul suggested potentially calibrating based on optimized implementations. Decision: start with `m * n * k` and adjust if needed.

3. **Additional ops for Edgeworth expansion:** Wilson Wu and Paul were going to discuss what ops Edgeworth expansion needs beyond einsum. This may add to the counted op set.

4. **Competition-specific parameters:** FLOP budget size, depth configurations, and scoring formula are evaluation harness concerns but affect how participants use mechestim.
