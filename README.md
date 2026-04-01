<div align="center">

# mechestim

**NumPy-compatible math primitives with analytical FLOP counting**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen.svg)]()

*Built for the [ARC Mechanistic Estimation Challenge](https://aicrowd.com) by [AIcrowd](https://aicrowd.com)*

</div>

---

**mechestim** is a drop-in replacement for a subset of NumPy that counts FLOPs as you compute. It lets researchers focus on **algorithmic innovation** instead of performance engineering &mdash; the competition score depends on your algorithm's analytical FLOP cost, not wall-clock time.

## Key Features

- **NumPy-compatible API** &mdash; `import mechestim as me` and write familiar NumPy code
- **Analytical FLOP counting** &mdash; deterministic, hardware-independent cost tracking
- **Budget enforcement** &mdash; operations are checked before execution; exceeding the budget raises a clear error
- **Symmetry-aware einsum** &mdash; automatic FLOP savings for repeated operands and declared symmetry groups
- **Transparent diagnostics** &mdash; inspect per-operation costs, cumulative budget usage, and detailed summaries at any time
- **Truncated SVD** &mdash; top-k singular value decomposition with `O(m*n*k)` cost

## Quick Start

### Installation

```bash
pip install git+https://github.com/AIcrowd/mechestim.git
```

### Basic Usage

```python
import mechestim as me

with me.BudgetContext(flop_budget=10_000_000) as budget:
    # Free operations (0 FLOPs) -- tensor creation, reshaping, indexing
    W = me.array(weight_matrix)
    x = me.zeros((256,))

    # Counted operations -- each deducts from the FLOP budget
    h = me.einsum('ij,j->i', W, x)      # cost: 256 * 256 FLOPs
    h = me.maximum(h, 0)                 # cost: 256 FLOPs (ReLU)
    U, S, Vt = me.linalg.svd(W, k=10)   # cost: 256 * 256 * 10 FLOPs

    # Inspect your budget at any time
    print(budget.summary())
```

```
mechestim FLOP Budget Summary
==============================
  Total budget:       10,000,000
  Used:                  721,664  ( 7.2%)
  Remaining:           9,278,336  (92.8%)

  By operation:
    einsum                65,536  ( 9.1%)  [1 call]
    linalg.svd           655,360  (90.8%)  [1 call]
    maximum                  256  ( 0.0%)  [1 call]
    mean                     512  ( 0.1%)  [1 call]
```

### Plan Your Budget Before Executing

```python
# Query FLOP costs without running anything (no BudgetContext needed)
cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,}")  # 16,777,216

cost = me.flops.svd_cost(m=256, n=256, k=10)
print(f"SVD cost: {cost:,}")     # 655,360
```

### Symmetry Savings

When you pass the same array object multiple times, mechestim automatically detects the symmetry and reduces the FLOP count:

```python
with me.BudgetContext(flop_budget=10**8) as budget:
    x = me.ones((10, 256))
    A = me.ones((10, 10))

    # x is passed twice (same object) -- cost is divided by 2!
    result = me.einsum('ai,bi,ab->', x, x, A)
    print(f"Cost with symmetry: {budget.flops_used:,}")   # 12,800
    # Without symmetry it would be 25,600
```

## Supported Operations

| Category | Operations | FLOP Cost |
|----------|-----------|-----------|
| **Einsum** | `me.einsum` | product of all index dimensions |
| **Dot / Matmul** | `me.dot`, `me.matmul` | equivalent einsum cost |
| **Unary** | `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tanh`, ... | numel(input) |
| **Binary** | `add`, `multiply`, `maximum`, `divide`, `power`, ... | numel(output) |
| **Reductions** | `sum`, `mean`, `max`, `min`, `std`, `var`, `argmax`, ... | numel(input) |
| **Linear algebra** | `me.linalg.svd(A, k=...)` | m &times; n &times; k |
| **Free (0 FLOPs)** | `zeros`, `ones`, `reshape`, `transpose`, `concatenate`, `where`, ... | 0 |
| **Random (0 FLOPs)** | `me.random.randn`, `me.random.normal`, `me.random.seed`, ... | 0 |

Unsupported operations raise a helpful `AttributeError` with a link to the docs.

## How It Works

1. **All computation runs inside a `BudgetContext`** &mdash; there is no separate precomputation phase. Free ops (tensor creation, reshaping) cost 0 FLOPs.
2. **FLOP costs are analytical** &mdash; computed from tensor shapes, not measured from execution. A matmul of `(m, k) @ (k, n)` always costs `m * k * n` FLOPs regardless of hardware.
3. **Budget is checked before execution** &mdash; if an operation would exceed the budget, `BudgetExhaustedError` is raised and the operation does not run.
4. **All tensors are plain `numpy.ndarray`** &mdash; no custom tensor class, no hidden state.

## Development

```bash
git clone https://github.com/AIcrowd/mechestim.git
cd mechestim
uv sync --all-extras
uv run pytest                  # 140 tests
uv run mkdocs serve            # local docs at http://127.0.0.1:8000
```

## License

MIT
