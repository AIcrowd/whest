<div align="center">
<img src="docs/assets/logo/logo.png" alt="mechestim" height="80">
<h1>mechestim</h1>
<p><strong>NumPy-compatible math primitives with analytical FLOP counting</strong></p>
</div>

<div align="center">

[![CI](https://github.com/AIcrowd/mechestim/actions/workflows/ci.yml/badge.svg)](https://github.com/AIcrowd/mechestim/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://aicrowd.github.io/mechestim/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

</div>

*Built for the [ARC Mechanistic Estimation Challenge](https://aicrowd.com) by [AIcrowd](https://aicrowd.com)*

---

**mechestim** is a drop-in replacement for a subset of NumPy that counts floating-point operations as you compute. Algorithms submitted to the ARC Mechanistic Estimation Challenge are scored by their analytical FLOP cost, not wall-clock time, so researchers can focus on **algorithmic innovation** rather than hardware tuning. Every arithmetic call deducts from a fixed budget; exceed it and execution stops immediately.

## Why mechestim?

<table>
<tr><th>NumPy</th><th>mechestim</th></tr>
<tr>
<td>

```python
import numpy as np

depth, width = 5, 256

# Weight init
scale = np.sqrt(2 / width)
weights = [
    np.random.randn(width, width) * scale
    for _ in range(depth)
]

# Forward pass
x = np.random.randn(width)
h = x
for i, W in enumerate(weights):
    h = np.einsum('ij,j->i', W, h)
    if i < depth - 1:
        h = np.maximum(h, 0)
# Total FLOPs? No idea.
```

</td>
<td>

```python
import mechestim as me

depth, width = 5, 256

# Weight init
scale = me.sqrt(2 / width)
weights = [
    me.random.randn(width, width) * scale
    for _ in range(depth)
]

# Forward pass
x = me.random.randn(width)
h = x
for i, W in enumerate(weights):
    h = me.einsum('ij,j->i', W, h)
    if i < depth - 1:
        h = me.maximum(h, 0)
me.budget_summary()  # 984,321 FLOPs
```

</td>
</tr>
</table>

## Key Features

- **NumPy-compatible API** -- `import mechestim as me` and write familiar NumPy code
- **Analytical FLOP counting** -- deterministic, hardware-independent cost tracking
- **Budget enforcement** -- operations are checked before execution; exceeding the budget raises a clear error
- **Symmetry-aware einsum** -- automatic FLOP savings for repeated operands and declared symmetry groups
- **Transparent diagnostics** -- inspect per-operation costs, cumulative budget usage, and detailed summaries at any time
- **Truncated SVD** -- top-k singular value decomposition with `O(m * n * k)` cost

## What's Supported

| Module | Operations | Cost Model | Status |
|--------|-----------|------------|--------|
| Core (`me.*`) | 367 | Varies by category (unary, binary, reduction, free) | Supported |
| `me.linalg` | 31 | Per-operation formulas | Supported |
| `me.fft` | 18 | `5n * ceil(log2(n))` for transforms | Supported |
| `me.random` | 51 | `numel(output)` per sample; shuffle: `n*ceil(log2(n))` | Supported |
| `me.polynomial` | 10 | Per-operation formulas | Supported |
| `me.window` | 5 | `n` to `3n` per window | Supported |
| **Total** | **450 supported** | | **32 blocked** |

Blocked operations (I/O, config, and system calls) raise a helpful `AttributeError` with a link to the docs.

## Quick Start

### Installation

```bash
pip install git+https://github.com/AIcrowd/mechestim.git
# or
uv add git+https://github.com/AIcrowd/mechestim.git
```

### Basic Usage

```python
import mechestim as me

depth, width = 5, 256

with me.BudgetContext(flop_budget=10**8) as budget:
    # Weight init
    scale = me.sqrt(2 / width)
    weights = [me.random.randn(width, width) * scale
               for _ in range(depth)]

    # Forward pass
    x = me.random.randn(width)
    h = x
    for i, W in enumerate(weights):
        h = me.einsum('ij,j->i', W, h)
        if i < depth - 1:
            h = me.maximum(h, 0)

    print(budget.summary())
```

```
mechestim FLOP Budget Summary
=============================
  Total budget:     100,000,000
  Used:                 984,321  (1.0%)
  Remaining:         99,015,679  (99.0%)

  By operation:
    einsum                655,360  ( 66.6%)  [5 calls]
    random.randn          327,936  ( 33.3%)  [6 calls]
    maximum                 1,024  (  0.1%)  [4 calls]
    sqrt                        1  (  0.0%)  [1 call]
```

### Plan Your Budget Before Executing

```python
# Query FLOP costs without running anything (no BudgetContext needed)
cost = me.flops.einsum_cost('ij,jk->ik', shapes=[(256, 256), (256, 256)])
print(f"Matmul cost: {cost:,}")  # 33,554,432

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

## How It Works

1. **FLOPs are tracked automatically.** A global default budget activates on first use, or you can wrap code in an explicit `BudgetContext` for a custom limit. Free ops (tensor creation, reshaping) cost 0 FLOPs.
2. **FLOP costs are analytical.** Costs are computed from tensor shapes, not measured from execution. A matmul of `(m, k) @ (k, n)` always costs `2 * m * k * n` FLOPs regardless of hardware.
3. **Budget is checked before execution.** If an operation would exceed the remaining budget, `BudgetExhaustedError` is raised and the operation does not run.
4. **All tensors are plain `numpy.ndarray`.** There is no custom tensor class and no hidden state. You can pass mechestim arrays to any NumPy-compatible library.

## Sharp Edges

**Budget is always active.** A global default budget (1e15 FLOPs, configurable via `MECHESTIM_DEFAULT_BUDGET` env var) activates automatically. Use an explicit `BudgetContext` to set a custom limit.

**32 operations are blocked.** I/O, config, and system-level functions (`save`, `load`, `set_printoptions`, etc.) raise `AttributeError` by design. These have no meaningful FLOP cost and are not part of the competition API.

**sort, argsort, trace, and random sampling all have analytical FLOP costs** based on their algorithmic complexity.

**Nested explicit BudgetContexts are not allowed.** Opening an explicit `BudgetContext` while another explicit `BudgetContext` is already active raises `RuntimeError`. However, opening an explicit context while only the global default is active is fine — the explicit context temporarily replaces the default.

**Cost is analytical, not wall-clock.** Two operations with the same shapes always report the same FLOP cost, regardless of data values, cache effects, or hardware. This is intentional -- it makes scores reproducible across machines.

**SymmetricTensor propagation rules.** Symmetry metadata (used by einsum for FLOP savings) propagates through reshaping, slicing, and unary pointwise operations (e.g., `exp`, `log`). Binary pointwise operations (e.g., `add`, `multiply`) intersect the symmetry groups of both operands. Reductions may drop symmetry on the reduced axis. If the result doesn't carry the symmetry you expect, declare symmetry groups explicitly with `me.as_symmetric()`.

## Documentation

**Getting Started**

- [Installation & Setup](docs/getting-started/installation.md)
- [Your First Budget](docs/getting-started/first-budget.md)

**How-To Guides**

- [Use Einsum](docs/how-to/use-einsum.md)
- [Exploit Symmetry](docs/how-to/exploit-symmetry.md)
- [Use Linear Algebra](docs/how-to/use-linalg.md)
- [Plan Your Budget](docs/how-to/plan-your-budget.md)
- [Debug Budget Overruns](docs/how-to/debug-budget-overruns.md)
- [Migrate from NumPy](docs/how-to/migrate-from-numpy.md)

**Concepts**

- [FLOP Counting Model](docs/concepts/flop-counting-model.md)
- [Operation Categories](docs/concepts/operation-categories.md)

**API Reference**

- [Full API Reference](docs/api/)

**For AI Agents:**

- [Cheat Sheet](docs/reference/cheat-sheet.md) -- compact reference for fast lookup

## Development

```bash
git clone https://github.com/AIcrowd/mechestim.git
cd mechestim
uv sync --all-extras
uv run pytest                  # run tests
uv run mkdocs serve            # local docs at http://127.0.0.1:8000
```

## Citation

```bibtex
@misc{mechestim2026,
  title  = {mechestim: NumPy-compatible math primitives with FLOP counting},
  author = {AIcrowd},
  year   = {2026},
  url    = {https://github.com/AIcrowd/mechestim},
  note   = {Built for the ARC Mechanistic Estimation Challenge}
}
```

## License

[MIT](https://opensource.org/licenses/MIT)
