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

- **NumPy drop-in API:** `import mechestim as np` should feel natural. Function names, signatures, and submodule structure mirror NumPy. Participants write normal NumPy code; the only change is the import line.
- **NumPy-native tensors:** All tensors are plain `numpy.ndarray`. No custom array class.
- **Analytical FLOP counting:** FLOP cost is computed from shapes, not measured from execution. The cost model is deterministic and hardware-independent.
- **Single context model:** Everything runs inside a `BudgetContext`. Non-compute ops (reshape, index, etc.) have zero FLOP cost. No separate precomputation phase.
- **Documentation-first:** Every public function has a comprehensive NumPy-style docstring. Professional documentation site auto-generated from docstrings.
- **Transparent diagnostics:** Op log and FLOP breakdown always available, same behavior locally and in remote evaluation.

## 3. Architecture

### 3.1 Usage Pattern

```python
import mechestim as np

# Budget context is the only mechestim-specific API
with np.BudgetContext(flop_budget=1_000_000) as budget:
    # Everything below reads like normal NumPy code
    W = np.array(weight_matrix)          # free (0 FLOPs)
    x = np.zeros((256,))                 # free
    h = np.einsum('ij,j->i', W, x)      # counted
    h = np.maximum(h, 0)                 # counted (ReLU)
    U, S, Vt = np.linalg.svd(W, k=10)   # counted

    print(budget.summary())

# Additional mechestim-specific APIs accessed via full import
import mechestim
cost = mechestim.flops.einsum_cost('ij,j->i', shapes=[(256, 256), (256,)])
```

### 3.2 System Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                     PARTICIPANT CODE                           │
│                                                                │
│  import mechestim as np                                        │
│                                                                │
│  with np.BudgetContext(flop_budget=N) as budget:               │
│      # Writes normal NumPy code                                │
│      # Free ops: np.zeros, np.reshape, np.concatenate, ...     │
│      # Counted ops: np.einsum, np.exp, np.linalg.svd, ...     │
│      result = my_estimator(weights, budget)                    │
│      print(budget.summary())                                   │
├────────────────────────────────────────────────────────────────┤
│                  mechestim module                               │
│                                                                │
│  Top-level (mirrors np.*)       Submodules                     │
│  ┌────────────────────────┐     ┌────────────────────────────┐ │
│  │ Counted:               │     │ np.linalg                  │ │
│  │   einsum, exp, log,    │     │   .svd(A, k=None)          │ │
│  │   abs, sqrt, square,   │     │   (future: eigh, cholesky) │ │
│  │   add, subtract,       │     │                            │ │
│  │   multiply, divide,    │     ├────────────────────────────┤ │
│  │   maximum, minimum,    │     │ np.random (passthrough)    │ │
│  │   power, clip,         │     │   .seed, .randn, .rand,   │ │
│  │   sum, max, min,       │     │   .normal, .uniform, ...   │ │
│  │   mean, prod           │     │   (free — 0 FLOPs)        │ │
│  │                        │     └────────────────────────────┘ │
│  │ Free:                  │                                    │
│  │   array, zeros, ones,  │     mechestim-specific             │
│  │   full, eye, diag,     │     ┌────────────────────────────┐ │
│  │   arange, linspace,    │     │ BudgetContext              │ │
│  │   reshape, transpose,  │     │ flops (cost query API)     │ │
│  │   concatenate, stack,  │     │ errors                     │ │
│  │   split, squeeze,      │     │ OpRecord                   │ │
│  │   copy, where, ...     │     └────────────────────────────┘ │
│  └───────────┬────────────┘                                    │
│              │                                                 │
│              ▼                                                 │
│  ┌────────────────────────┐                                    │
│  │ FLOP Counter            │                                    │
│  │ (per-op cost calc)      │                                    │
│  └───────────┬────────────┘                                    │
│              ▼                                                 │
│  ┌────────────────────────┐                                    │
│  │ NumPy backend           │                                    │
│  │ (actual np.* calls)     │                                    │
│  └────────────────────────┘                                    │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow for Counted Operations

```
np.einsum(subscripts, *operands)    # np = mechestim
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
    ├── 4. Execute via real NumPy backend (numpy.einsum)
    │
    ├── 5. Check result for NaN/Inf → warn if present
    │
    ├── 6. Record in op_log (OpRecord)
    │
    └── 7. Deduct cost from budget, return result (ndarray)
```

### 3.4 Data Flow for Free Operations

```
np.reshape(x, shape)    # np = mechestim
    │
    ├── 1. Validate inputs
    │
    ├── 2. Execute via real NumPy backend (numpy.reshape)
    │
    └── 3. Return result (ndarray) — no budget interaction
```

## 4. API Surface: What's Supported

The goal is: if a participant writes valid NumPy code using operations in the supported set, replacing `import numpy as np` with `import mechestim as np` should be the only change needed.

### 4.1 Unsupported Operations

Any attribute or function NOT listed in Sections 4.2-4.4 raises `AttributeError` with a helpful message:

```python
import mechestim as np
np.fft.fft(x)
# AttributeError: mechestim does not provide 'fft.fft'.
# Supported operations: einsum, exp, log, ..., linalg.svd.
# See https://... for the full list.
# If you need this operation, request it at https://...
```

This is critical for the sandbox: participants get a clear error instead of silently falling back to real NumPy.

### 4.2 Counted Operations (FLOP cost > 0)

These are the operations that consume the FLOP budget. Signatures match NumPy exactly (with one extension: `symmetric_dims` on `einsum`).

#### 4.2.1 `np.einsum`

```python
np.einsum(subscripts: str, *operands: ndarray, symmetric_dims: list[tuple[int, ...]] | None = None) -> ndarray
```

The primary workhorse. Wraps `numpy.einsum`.

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

When `symmetric_dims` is provided, the library validates the claim by checking `numpy.allclose(result, result.transpose(perm), atol=1e-6, rtol=1e-5)` for the relevant permutations. Raises `SymmetryError` if the claim is invalid.

**Diagonal/repeated-index patterns:**

Expressions like `'i->ii'` (diagonal embedding) are not supported by `numpy.einsum` natively. mechestim implements these via `numpy.diag` or equivalent, with cost = numel(output).

**Examples:**

```python
import mechestim as np

with np.BudgetContext(flop_budget=10**8) as budget:
    # Standard matmul: cost = m * k * n
    C = np.einsum('ij,jk->ik', A, B)  # A is (m,k), B is (k,n)

    # Symmetric contraction: x passed twice, cost = (a*b*i) / 2!
    result = np.einsum('ai,bi,ab->', x, x, A)  # x is same object

    # Trace: cost = n
    tr = np.einsum('ii->', A)

    # With explicit symmetric output dims: cost reduced by 2!
    S = np.einsum('ai,bj,ab->ij', x, y, A, symmetric_dims=[(0,1)])
```

**Note:** `symmetric_dims` is the ONLY non-standard NumPy parameter. It is optional and keyword-only, so standard NumPy einsum calls work unchanged.

#### 4.2.2 Pointwise Operations

Element-wise operations. Cost = `numel(output)` (1 FLOP per output element). For binary ops, output shape follows NumPy broadcasting rules.

| Function | NumPy equivalent | Cost |
|---|---|---|
| `np.exp(x)` | `numpy.exp` | numel(input) |
| `np.log(x)` | `numpy.log` | numel(input) |
| `np.log2(x)` | `numpy.log2` | numel(input) |
| `np.log10(x)` | `numpy.log10` | numel(input) |
| `np.abs(x)` | `numpy.abs` | numel(input) |
| `np.negative(x)` | `numpy.negative` | numel(input) |
| `np.sqrt(x)` | `numpy.sqrt` | numel(input) |
| `np.square(x)` | `numpy.square` | numel(input) |
| `np.sin(x)` | `numpy.sin` | numel(input) |
| `np.cos(x)` | `numpy.cos` | numel(input) |
| `np.tanh(x)` | `numpy.tanh` | numel(input) |
| `np.sign(x)` | `numpy.sign` | numel(input) |
| `np.ceil(x)` | `numpy.ceil` | numel(input) |
| `np.floor(x)` | `numpy.floor` | numel(input) |
| `np.add(x, y)` | `numpy.add` | numel(output) |
| `np.subtract(x, y)` | `numpy.subtract` | numel(output) |
| `np.multiply(x, y)` | `numpy.multiply` | numel(output) |
| `np.divide(x, y)` | `numpy.divide` | numel(output) |
| `np.maximum(x, y)` | `numpy.maximum` | numel(output) |
| `np.minimum(x, y)` | `numpy.minimum` | numel(output) |
| `np.power(x, y)` | `numpy.power` | numel(output) |
| `np.clip(x, a_min, a_max)` | `numpy.clip` | numel(output) |
| `np.mod(x, y)` | `numpy.mod` | numel(output) |

#### 4.2.3 Reductions

Cost = `numel(input)` (must scan all input elements).

| Function | NumPy equivalent | Cost |
|---|---|---|
| `np.sum(x, axis=None)` | `numpy.sum` | numel(input) |
| `np.max(x, axis=None)` | `numpy.max` | numel(input) |
| `np.min(x, axis=None)` | `numpy.min` | numel(input) |
| `np.mean(x, axis=None)` | `numpy.mean` | numel(input) + numel(output) |
| `np.prod(x, axis=None)` | `numpy.prod` | numel(input) |
| `np.std(x, axis=None)` | `numpy.std` | 2 * numel(input) + numel(output) |
| `np.var(x, axis=None)` | `numpy.var` | 2 * numel(input) + numel(output) |
| `np.argmax(x, axis=None)` | `numpy.argmax` | numel(input) |
| `np.argmin(x, axis=None)` | `numpy.argmin` | numel(input) |
| `np.cumsum(x, axis=None)` | `numpy.cumsum` | numel(input) |
| `np.cumprod(x, axis=None)` | `numpy.cumprod` | numel(input) |

#### 4.2.4 Linear Algebra: `np.linalg`

```python
np.linalg.svd(A: ndarray, k: int | None = None) -> tuple[ndarray, ndarray, ndarray]
```

Truncated singular value decomposition.

**FLOP cost:** `m * n * k` where `A` is `(m, n)`. If `k is None`, `k = min(m, n)`.

**Returns:** `(U, S, Vt)` where:
- `U` is `(m, k)` — left singular vectors
- `S` is `(k,)` — singular values (descending order)
- `Vt` is `(k, n)` — right singular vectors (transposed)

**Note:** The `k` parameter is a mechestim extension. Standard `numpy.linalg.svd` does not have it. When `k is None`, behavior matches NumPy exactly (full SVD with `full_matrices=False`).

**Constraints:**
- Input must be 2D
- `k` must satisfy `1 <= k <= min(m, n)`

**Future linalg ops** (not in v1 but may be added):
- `np.linalg.eigh` — symmetric eigendecomposition
- `np.linalg.cholesky` — Cholesky decomposition
- `np.linalg.norm` — matrix/vector norms
- `np.linalg.solve` — linear system solve

#### 4.2.5 Dot and Matmul

| Function | NumPy equivalent | Cost |
|---|---|---|
| `np.dot(a, b)` | `numpy.dot` | Same as equivalent einsum |
| `np.matmul(a, b)` | `numpy.matmul` | Same as equivalent einsum |

These are convenience wrappers. Internally they delegate to `np.einsum` with the appropriate subscript, so the FLOP cost is identical.

#### 4.2.6 FLOP Cost Summary

```
OPERATION                       | FLOP COST
--------------------------------|------------------------------------------
einsum(subscripts, *ops)        | product(all index dims) / symmetry_factor
dot, matmul                     | equivalent einsum cost
exp, log, abs, neg, sqrt, sq,   | numel(input)
sin, cos, tanh, sign, ceil,     |
floor                           |
add, sub, mul, div, max, min,   | numel(output)  [broadcast-aware]
power, clip, mod                |
sum, max, min, prod (reduction) | numel(input)
mean (reduction)                | numel(input) + numel(output)
std, var (reduction)            | 2 * numel(input) + numel(output)
argmax, argmin                  | numel(input)
cumsum, cumprod                 | numel(input)
linalg.svd(A, k)                | m * n * k
Free ops (reshape, etc.)        | 0
```

All costs are multiplied by `BudgetContext.flop_multiplier` (default 1).

### 4.3 Free Operations (Zero FLOP Cost)

These operations are provided because raw NumPy is disabled in the evaluation sandbox. They wrap NumPy equivalents with no FLOP cost. They do NOT require a `BudgetContext` — they work anywhere.

#### 4.3.1 Tensor Creation

| Function | NumPy equivalent |
|---|---|
| `np.array(data, dtype=None)` | `numpy.array` |
| `np.zeros(shape, dtype=float)` | `numpy.zeros` |
| `np.ones(shape, dtype=float)` | `numpy.ones` |
| `np.full(shape, fill, dtype=None)` | `numpy.full` |
| `np.eye(n, m=None, dtype=float)` | `numpy.eye` |
| `np.diag(v, k=0)` | `numpy.diag` |
| `np.arange(*args, dtype=None)` | `numpy.arange` |
| `np.linspace(start, stop, num=50)` | `numpy.linspace` |
| `np.zeros_like(x)` | `numpy.zeros_like` |
| `np.ones_like(x)` | `numpy.ones_like` |
| `np.full_like(x, fill)` | `numpy.full_like` |
| `np.empty(shape, dtype=float)` | `numpy.empty` |
| `np.empty_like(x)` | `numpy.empty_like` |
| `np.identity(n)` | `numpy.identity` |

#### 4.3.2 Tensor Manipulation

| Function | NumPy equivalent |
|---|---|
| `np.reshape(x, shape)` | `numpy.reshape` |
| `np.transpose(x, axes=None)` | `numpy.transpose` |
| `np.swapaxes(x, a1, a2)` | `numpy.swapaxes` |
| `np.moveaxis(x, src, dst)` | `numpy.moveaxis` |
| `np.concatenate(arrays, axis=0)` | `numpy.concatenate` |
| `np.stack(arrays, axis=0)` | `numpy.stack` |
| `np.vstack(arrays)` | `numpy.vstack` |
| `np.hstack(arrays)` | `numpy.hstack` |
| `np.split(x, indices, axis=0)` | `numpy.split` |
| `np.hsplit(x, indices)` | `numpy.hsplit` |
| `np.vsplit(x, indices)` | `numpy.vsplit` |
| `np.squeeze(x, axis=None)` | `numpy.squeeze` |
| `np.expand_dims(x, axis)` | `numpy.expand_dims` |
| `np.ravel(x)` | `numpy.ravel` |
| `np.flatten(x)` | (via ndarray method) |
| `np.copy(x)` | `numpy.copy` |
| `np.where(cond, x, y)` | `numpy.where` |
| `np.tile(x, reps)` | `numpy.tile` |
| `np.repeat(x, repeats, axis=None)` | `numpy.repeat` |
| `np.flip(x, axis=None)` | `numpy.flip` |
| `np.roll(x, shift, axis=None)` | `numpy.roll` |
| `np.sort(x, axis=-1)` | `numpy.sort` |
| `np.argsort(x, axis=-1)` | `numpy.argsort` |
| `np.searchsorted(a, v)` | `numpy.searchsorted` |
| `np.unique(x)` | `numpy.unique` |
| `np.pad(x, pad_width, mode)` | `numpy.pad` |
| `np.triu(x, k=0)` | `numpy.triu` |
| `np.tril(x, k=0)` | `numpy.tril` |
| `np.diagonal(x, offset=0)` | `numpy.diagonal` |
| `np.trace(x)` | `numpy.trace` |
| `np.broadcast_to(x, shape)` | `numpy.broadcast_to` |
| `np.meshgrid(*xi)` | `numpy.meshgrid` |

#### 4.3.3 Type, Constants, and Info

| Name | NumPy equivalent |
|---|---|
| `np.astype(x, dtype)` | `x.astype(dtype)` |
| `np.asarray(x)` | `numpy.asarray` |
| `np.isnan(x)` | `numpy.isnan` |
| `np.isinf(x)` | `numpy.isinf` |
| `np.isfinite(x)` | `numpy.isfinite` |
| `np.allclose(a, b)` | `numpy.allclose` |
| `np.pi` | `numpy.pi` |
| `np.e` | `numpy.e` |
| `np.inf` | `numpy.inf` |
| `np.nan` | `numpy.nan` |
| `np.float32`, `np.float64`, etc. | `numpy.float32`, etc. |
| `np.int32`, `np.int64`, etc. | `numpy.int32`, etc. |
| `np.bool_` | `numpy.bool_` |
| `np.ndarray` | `numpy.ndarray` |
| `np.newaxis` | `numpy.newaxis` |

**Indexing/slicing:** `x[i, j]`, `x[:, 0]`, `x[mask]`, etc. work natively on ndarrays. No wrapping needed.

**ndarray attributes and methods:** `.shape`, `.dtype`, `.ndim`, `.size`, `.T`, `.reshape()`, `.transpose()`, `.astype()`, `.copy()`, `.flatten()`, `.ravel()` all work natively since tensors are plain ndarrays.

#### 4.3.4 Random: `np.random`

| Function | NumPy equivalent |
|---|---|
| `np.random.seed(s)` | `numpy.random.seed` |
| `np.random.randn(*shape)` | `numpy.random.randn` |
| `np.random.rand(*shape)` | `numpy.random.rand` |
| `np.random.normal(loc, scale, size)` | `numpy.random.normal` |
| `np.random.uniform(low, high, size)` | `numpy.random.uniform` |
| `np.random.choice(a, size, ...)` | `numpy.random.choice` |
| `np.random.permutation(x)` | `numpy.random.permutation` |
| `np.random.shuffle(x)` | `numpy.random.shuffle` |
| `np.random.RandomState(seed)` | `numpy.random.RandomState` |
| `np.random.default_rng(seed)` | `numpy.random.default_rng` |

All random ops are free (0 FLOPs). Sampling random numbers is data generation, not compute. This allows participants to draw Monte Carlo samples within their estimator as part of their algorithm.

### 4.4 FLOP Cost Query API

Each operation has a corresponding cost calculator. These are mechestim-specific (not NumPy) and accessed via `mechestim.flops`:

```python
import mechestim

# Query costs without executing (no BudgetContext needed)
cost = mechestim.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
cost = mechestim.flops.svd_cost(m=256, n=256, k=10)
cost = mechestim.flops.pointwise_cost(shape=(256, 256))
cost = mechestim.flops.reduction_cost(input_shape=(256, 256), axis=1)
```

These functions take shapes (not data), return an integer FLOP count, and do not require a `BudgetContext`. They are useful for participants planning their FLOP budget allocation.

## 5. Budget API

### 5.1 BudgetContext

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
    >>> import mechestim as np
    >>> with np.BudgetContext(flop_budget=1_000_000) as budget:
    ...     x = np.ones((256, 256))  # free
    ...     y = np.einsum('ij,jk->ik', x, x)  # costs 256^3 FLOPs
    ...     print(budget.flops_used)
    16777216
    """
```

### 5.2 Budget Introspection

| Attribute / Method | Type | Description |
|---|---|---|
| `budget.flop_budget` | `int` | Original budget |
| `budget.flops_used` | `int` | Cumulative FLOPs consumed |
| `budget.flops_remaining` | `int` | `flop_budget - flops_used` |
| `budget.flop_multiplier` | `float` | Cost multiplier |
| `budget.op_log` | `list[OpRecord]` | Every counted op executed |
| `budget.summary()` | `str` | Pretty-printed breakdown |

### 5.3 OpRecord

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

### 5.4 Budget Summary Format

```
mechestim FLOP Budget Summary
==============================
  Total budget:      1,000,000
  Used:                750,432  (75.0%)
  Remaining:           249,568  (25.0%)

  By operation:
    einsum           720,000  (96.0%)  [142 calls]
    exp               15,360  ( 2.0%)  [60 calls]
    linalg.svd        15,072  ( 2.0%)  [3 calls]
```

## 6. Error Handling

### 6.1 Exception Hierarchy

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

### 6.2 Error Behavior

| Situation | Exception | Message pattern |
|---|---|---|
| Counted op outside BudgetContext | `NoBudgetContextError` | "No active BudgetContext. Wrap your code in `with mechestim.BudgetContext(...):`" |
| Op would exceed budget | `BudgetExhaustedError` | "einsum would cost 16,777,216 FLOPs but only 1,000 remain" |
| Unsupported function/attribute | `AttributeError` | "mechestim does not provide 'fft.fft'. Supported: ..." |
| Invalid einsum subscripts | `ValueError` | Delegates to NumPy's error message |
| Shape mismatch | `ValueError` | Delegates to NumPy's error message |
| Non-ndarray to counted op | `TypeError` | "Expected numpy.ndarray, got list" |
| Claimed symmetry invalid | `SymmetryError` | "Tensor not symmetric along dims (0, 1): max deviation = 0.5" |
| Nested BudgetContext | `RuntimeError` | "Cannot nest BudgetContexts" |
| Budget <= 0 | `ValueError` | "flop_budget must be > 0, got -5" |

### 6.3 NaN/Inf Warnings

When any counted operation produces NaN or Inf values in its result, mechestim issues a Python warning via `warnings.warn()`:

```
MechEstimWarning: einsum produced 3 NaN and 0 Inf values in output of shape (256, 256)
```

This does NOT raise an exception. Participants can suppress warnings with `warnings.filterwarnings('ignore', category=MechEstimWarning)`.

## 7. Documentation Requirements

### 7.1 Docstring Standard

Every public function and class uses NumPy-style docstrings with these sections:

- **Short summary** — one line
- **Extended summary** — wraps which NumPy function, explains FLOP cost
- **Parameters** — with types and descriptions
- **Returns** — with types
- **Raises** — all possible exceptions
- **Notes** — FLOP cost formula, symmetry behavior
- **Examples** — using `import mechestim as np` pattern

### 7.2 Documentation Site

**Tool:** mkdocs with mkdocstrings (Python handler) and the Material theme.

**Structure:**
```
docs/
├── index.md                    # Getting Started
├── quickstart.md               # Installation + first example
├── concepts/
│   ├── flop-counting.md        # How FLOP costs work
│   ├── budget-context.md       # Budget enforcement model
│   ├── symmetry.md             # Symmetry detection and savings
│   └── numpy-compatibility.md  # What works, what doesn't, why
├── api/                        # Auto-generated from docstrings
│   ├── counted-ops.md          # einsum, pointwise, reductions, linalg
│   ├── free-ops.md             # tensor creation, manipulation, random
│   ├── budget.md               # BudgetContext, OpRecord
│   ├── flops.md                # Cost query API
│   └── errors.md               # Exception classes
├── examples/
│   ├── meanprop.md             # Full meanprop walkthrough
│   └── flop-planning.md        # Using cost query API for planning
└── changelog.md
```

**Hosting:** GitHub Pages, auto-deployed from main branch via GitHub Actions.

## 8. Module Structure

```
mechestim/
├── pyproject.toml           # Package config (uv/pip compatible)
├── README.md                # Overview + installation + quick example
├── mkdocs.yml               # Documentation site config
├── src/
│   └── mechestim/
│       ├── __init__.py      # Public API: mirrors numpy top-level namespace
│       ├── linalg/          # np.linalg submodule
│       │   ├── __init__.py  # svd (and future linalg ops)
│       │   └── _svd.py      # SVD implementation
│       ├── random/          # np.random submodule (passthrough to numpy)
│       │   └── __init__.py  # All random functions
│       ├── _budget.py       # BudgetContext, OpRecord, budget state
│       ├── _flops.py        # FLOP cost calculators (internal)
│       ├── _einsum.py       # einsum: parsing, symmetry detection
│       ├── _pointwise.py    # Pointwise ops and reductions
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
│   ├── test_random.py       # Random submodule pass-through
│   ├── test_errors.py       # Error conditions + AttributeError on unsupported
│   ├── test_numpy_compat.py # Verify `import mechestim as np` pattern works
│   └── test_integration.py  # End-to-end estimator flows
└── docs/                    # mkdocs source
    └── ...
```

### 8.1 `__init__.py` Design

The top-level `__init__.py` exposes all supported functions and raises `AttributeError` for unsupported ones via `__getattr__`:

```python
# mechestim/__init__.py (conceptual structure)

from mechestim._budget import BudgetContext, OpRecord
from mechestim._einsum import einsum
from mechestim._pointwise import exp, log, abs, sqrt, ...
from mechestim._pointwise import add, subtract, multiply, divide, ...
from mechestim._pointwise import sum, max, min, mean, prod, ...
from mechestim._free_ops import array, zeros, ones, full, eye, ...
from mechestim._free_ops import reshape, transpose, concatenate, ...
from mechestim import linalg
from mechestim import random
from mechestim import flops
from mechestim.errors import (
    MechEstimError, BudgetExhaustedError,
    NoBudgetContextError, SymmetryError,
)

# NumPy constants
import numpy as _np
pi = _np.pi
e = _np.e
inf = _np.inf
nan = _np.nan
newaxis = _np.newaxis
float32 = _np.float32
float64 = _np.float64
int32 = _np.int32
int64 = _np.int64
bool_ = _np.bool_
ndarray = _np.ndarray
# ... etc.

_SUPPORTED = {name for name in dir() if not name.startswith('_')}

def __getattr__(name):
    raise AttributeError(
        f"mechestim does not provide '{name}'. "
        f"See https://... for supported operations."
    )
```

## 9. Testing Strategy

### 9.1 FLOP Cost Model Tests (`test_flops.py`)

The most critical test suite. For every counted operation, verify that the FLOP cost matches the analytical prediction exactly.

```python
# einsum: product of all index dims
assert einsum_cost('ij,jk->ik', [(3,4), (4,5)]) == 3 * 4 * 5

# einsum: trace
assert einsum_cost('ii->', [(10,10)]) == 10

# einsum: batch matmul
assert einsum_cost('bij,bjk->bik', [(2,3,4), (2,4,5)]) == 2 * 3 * 4 * 5

# einsum: symmetry with repeated operand
assert einsum_cost_symmetric('ai,bi,ab->', [(10,256), 'same', (10,10)]) == (10*10*256) // 2

# svd: m * n * k
assert svd_cost(m=100, n=50, k=10) == 100 * 50 * 10

# pointwise: numel
assert pointwise_cost((256, 256)) == 256 * 256

# reduction: numel of input
assert reduction_cost((256, 256)) == 256 * 256
```

### 9.2 Budget Enforcement Tests (`test_budget.py`)

```python
import mechestim as np

# Budget exactly sufficient
with np.BudgetContext(flop_budget=256**3) as budget:
    np.einsum('ij,jk->ik', A_256, B_256)  # succeeds

# Budget insufficient
with pytest.raises(np.BudgetExhaustedError):
    with np.BudgetContext(flop_budget=100) as budget:
        np.einsum('ij,jk->ik', A_256, B_256)

# Outside context
with pytest.raises(np.NoBudgetContextError):
    np.einsum('ij,jk->ik', A, B)

# Free ops work without context
x = np.zeros((10, 10))  # no error

# Nested context
with pytest.raises(RuntimeError):
    with np.BudgetContext(1000):
        with np.BudgetContext(500):
            pass
```

### 9.3 Numerical Correctness Tests (`test_numpy_compat.py`)

Every supported operation must produce the same numerical result as NumPy:

```python
import numpy
import mechestim as np

with np.BudgetContext(10**9) as budget:
    assert numpy.allclose(np.exp(x), numpy.exp(x))
    assert numpy.allclose(np.einsum('ij,jk->ik', A, B), numpy.einsum('ij,jk->ik', A, B))
    assert numpy.allclose(np.dot(A, B), numpy.dot(A, B))
    U, S, Vt = np.linalg.svd(A, k=5)
    U_np, S_np, Vt_np = numpy.linalg.svd(A, full_matrices=False)
    assert numpy.allclose(S, S_np[:5])
```

### 9.4 Unsupported Op Tests (`test_errors.py`)

```python
import mechestim as np

# Unsupported top-level
with pytest.raises(AttributeError, match="does not provide"):
    np.fft

# Unsupported linalg
with pytest.raises(AttributeError, match="does not provide"):
    np.linalg.cholesky
```

### 9.5 Integration Tests (`test_integration.py`)

Run ARC's reference algorithms (meanprop at minimum) end-to-end through mechestim:
- Verify the estimator completes within a reasonable FLOP budget
- Verify numerical output matches direct NumPy computation
- Verify the op_log correctly records all operations

## 10. Packaging and Distribution

### 10.1 pyproject.toml

```toml
[project]
name = "mechestim"
version = "0.1.0"
description = "NumPy-compatible math primitives with FLOP counting for the Mechanistic Estimation Challenge"
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

### 10.2 Installation

```bash
# From GitHub (primary distribution)
pip install git+https://github.com/AIcrowd/mechestim.git

# For development
git clone https://github.com/AIcrowd/mechestim.git
cd mechestim
uv sync --all-extras
```

### 10.3 Versioning

- **v0.1.0:** Warm-up round. API may change based on participant feedback.
- **v0.x.y:** Adding new ops is a minor version bump. Changing FLOP cost models is a minor version bump with changelog.
- **v1.0.0:** Stable API for main competition round.

## 11. Not in Scope (v1)

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
| `np.linalg.norm`, `np.linalg.solve` | Not in initial op set. Add if requested. |

## 12. Open Items for ARC Confirmation

1. **Einsum symmetry detection algorithm:** The exact algorithm for detecting when repeated operands produce symmetric savings needs validation against ARC's reference implementations.

2. **SVD cost constant:** Currently specified as exactly `m * n * k`. Paul suggested potentially calibrating based on optimized implementations. Decision: start with `m * n * k` and adjust if needed.

3. **Additional ops for Edgeworth expansion:** Wilson Wu and Paul were going to discuss what ops Edgeworth expansion needs beyond einsum. This may add to the counted op set.

4. **Competition-specific parameters:** FLOP budget size, depth configurations, and scoring formula are evaluation harness concerns but affect how participants use mechestim.
