<img src="assets/logo/logo.png" alt="mechestim logo" style="height: 80px;">

# mechestim

**NumPy-compatible math primitives with analytical FLOP counting.**

!!! warning "mechestim is not a drop-in NumPy replacement"
    Operations have analytical FLOP costs and 32 operations are blocked.
    A `BudgetContext` is optional — a global default activates automatically — but
    using one explicitly gives you budget limits, namespacing, and summaries.
    See [Operation Categories](concepts/operation-categories.md).

Pick the path that matches what you need right now.

## 🚀 I want to get started

- [Installation](./getting-started/installation.md)
- [Your First Budget](./getting-started/first-budget.md)

## 🛠 Something isn't working

- [Common Errors](./troubleshooting/common-errors.md)
- [Error Reference](./api/errors.md)

## 📈 I want to write efficient code with mechestim

- [Migrate from NumPy](./how-to/migrate-from-numpy.md)
- [Use Einsum](./how-to/use-einsum.md)
- [Exploit Symmetry](./how-to/exploit-symmetry.md)
- [Use Linear Algebra](./how-to/use-linalg.md)
- [Plan Your Budget](./how-to/plan-your-budget.md)
- [Debug Budget Overruns](./how-to/debug-budget-overruns.md)

## 🧠 I want to understand how it works

- [FLOP Counting Model](./concepts/flop-counting-model.md) — how costs are computed, why FLOPs
- [Operation Categories](./concepts/operation-categories.md) — free vs counted vs unsupported

## 🏗 I want to understand the sandboxed architecture

- [Client-Server Model](./architecture/client-server.md) — why it exists, how it works
- [Running with Docker](./architecture/docker.md) — local setup with Docker

## Quick example

Operations run freely without any setup — the global default budget tracks FLOPs automatically:

```python
import mechestim as me

# No BudgetContext needed — the global default is active
scale = me.sqrt(me.array(2 / 256))
W = me.multiply(me.random.randn(256, 256), scale)
x = me.einsum('ij,j->i', W, me.random.randn(256))

print(me.budget_summary())
```

For budget limits and namespacing, use an explicit `BudgetContext`:

```python
import mechestim as me

depth, width = 5, 256

with me.BudgetContext(flop_budget=10**8, namespace="mlp-forward") as budget:
    # Weight init — randn and multiply are both counted
    scale = me.sqrt(me.array(2 / width))
    weights = [me.multiply(me.random.randn(width, width), scale)
               for _ in range(depth)]

    # Forward pass
    x = me.random.randn(width)
    h = x
    for i, W in enumerate(weights):
        h = me.einsum('ij,j->i', W, h)    # linear layer
        if i < depth - 1:
            h = me.maximum(h, 0)           # ReLU

print(me.budget_summary())
```

```
mechestim FLOP Budget Summary
==============================
  Namespace:        mlp-forward
  Total budget:     100,000,000
  Used:                 656,385  (0.7%)
  Remaining:         99,343,615  (99.3%)

  By operation:
    multiply              327,680  ( 49.9%)  [5 calls]
    einsum                327,680  ( 49.9%)  [5 calls]
    maximum                 1,024  (  0.2%)  [4 calls]
    sqrt                        1  (  0.0%)  [1 call]
```

## Installation

```bash
uv add git+https://github.com/AIcrowd/mechestim.git
```

## Full Taxonomy

- **Getting Started:** [Installation](./getting-started/installation.md), [Your First Budget](./getting-started/first-budget.md)
- **How-To:** [Migrate from NumPy](./how-to/migrate-from-numpy.md), [Use Einsum](./how-to/use-einsum.md), [Exploit Symmetry](./how-to/exploit-symmetry.md), [Use Linear Algebra](./how-to/use-linalg.md), [Use FFT](./how-to/use-fft.md), [Plan Your Budget](./how-to/plan-your-budget.md), [Debug Budget Overruns](./how-to/debug-budget-overruns.md)
- **Concepts:** [FLOP Counting Model](./concepts/flop-counting-model.md), [Operation Categories](./concepts/operation-categories.md), [NumPy Compatibility Testing](./concepts/numpy-compatibility-testing.md)
- **Architecture:** [Client-Server Model](./architecture/client-server.md), [Running with Docker](./architecture/docker.md)
- **API Reference:** [Counted Ops](./api/counted-ops.md), [Free Ops](./api/free-ops.md), [Symmetric Tensors](./api/symmetric.md), [Linear Algebra](./api/linalg.md), [FFT](./api/fft.md), [Random](./api/random.md), [Polynomial](./api/polynomial.md), [Window Functions](./api/window.md), [Budget](./api/budget.md), [FLOP Cost Query](./api/flops.md), [Errors](./api/errors.md)
- **Reference:** [For AI Agents](./reference/for-agents.md), [Operation Audit](./reference/operation-audit.md), [FLOP Cost Cheat Sheet](./reference/cheat-sheet.md)
- **Troubleshooting:** [Common Errors](./troubleshooting/common-errors.md)
- **Changelog:** [Changelog](./changelog.md)
