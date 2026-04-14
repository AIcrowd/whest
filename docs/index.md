<img src="assets/logo/logo.png" alt="whest logo" style="height: 80px;">

# whest

**NumPy-compatible math primitives with analytical FLOP counting.**

!!! warning "whest is not a drop-in NumPy replacement"
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

## 📈 I want to write efficient code with whest

- [Migrate from NumPy](./how-to/migrate-from-numpy.md)
- [Use Einsum](./how-to/use-einsum.md)
- [Exploit Symmetry](./how-to/exploit-symmetry.md)
- [Use Linear Algebra](./how-to/use-linalg.md)
- [Plan Your Budget](./how-to/plan-your-budget.md)
- [Calibrate Weights](./how-to/calibrate-weights.md)
- [Debug Budget Overruns](./how-to/debug-budget-overruns.md)

## 🧠 I want to understand how it works

- [FLOP Counting Model](./concepts/flop-counting-model.md) — how costs are computed, why FLOPs
- [Operation Categories](./concepts/operation-categories.md) — free vs counted vs unsupported

## 🏗 I want to understand the sandboxed architecture

- [Client-Server Model](./architecture/client-server.md) — why it exists, how it works
- [Running with Docker](./architecture/docker.md) — local setup with Docker

## 🧪 I want to work on the repository

- [Contributor Guide](./development/contributing.md) — repo layout, test commands, generated docs

## Quick example

Operations run freely without any setup — the global default budget tracks FLOPs automatically:

```python
import whest as we

# No BudgetContext needed — the global default is active
scale = we.sqrt(we.array(2 / 256))
W = we.multiply(we.random.randn(256, 256), scale)
x = we.einsum('ij,j->i', W, we.random.randn(256))

print(we.budget_summary())
```

For budget limits and namespacing, use an explicit `BudgetContext`:

```python
import whest as we

depth, width = 5, 256

with we.BudgetContext(flop_budget=10**8, namespace="mlp-forward") as budget:
    # Weight init — randn and multiply are both counted
    scale = we.sqrt(we.array(2 / width))
    weights = [we.multiply(we.random.randn(width, width), scale)
               for _ in range(depth)]

    # Forward pass
    x = we.random.randn(width)
    h = x
    for i, W in enumerate(weights):
        h = we.einsum('ij,j->i', W, h)    # linear layer
        if i < depth - 1:
            h = we.maximum(h, 0)           # ReLU

print(we.budget_summary())
```

```
whest FLOP Budget Summary
==============================
  Namespace:        mlp-forward
  Total budget:     100,000,000
  Used:               1,312,001  (1.3%)
  Remaining:         98,687,999  (98.7%)

  By operation:
    einsum                655,360  ( 50.0%)  [5 calls]
    random.randn          327,936  ( 25.0%)  [6 calls]
    multiply              327,680  ( 25.0%)  [5 calls]
    maximum                 1,024  (  0.1%)  [4 calls]
    sqrt                        1  (  0.0%)  [1 call]
```

## Installation

```bash
uv add git+https://github.com/AIcrowd/whest.git
```

## Full Taxonomy

- **Getting Started:** [Installation](./getting-started/installation.md), [Your First Budget](./getting-started/first-budget.md)
- **How-To:** [Migrate from NumPy](./how-to/migrate-from-numpy.md), [Use Einsum](./how-to/use-einsum.md), [Exploit Symmetry](./how-to/exploit-symmetry.md), [Use Linear Algebra](./how-to/use-linalg.md), [Use FFT](./how-to/use-fft.md), [Plan Your Budget](./how-to/plan-your-budget.md), [Calibrate Weights](./how-to/calibrate-weights.md), [Debug Budget Overruns](./how-to/debug-budget-overruns.md)
- **Concepts:** [FLOP Counting Model](./concepts/flop-counting-model.md), [Operation Categories](./concepts/operation-categories.md), [NumPy Compatibility Testing](./concepts/numpy-compatibility-testing.md)
- **Architecture:** [Client-Server Model](./architecture/client-server.md), [Running with Docker](./architecture/docker.md)
- **Development:** [Contributor Guide](./development/contributing.md)
- **API Reference:** [Counted Ops](./api/counted-ops.md), [Free Ops](./api/free-ops.md), [Symmetric Tensors](./api/symmetric.md), [Linear Algebra](./api/linalg.md), [FFT](./api/fft.md), [Random](./api/random.md), [Polynomial](./api/polynomial.md), [Window Functions](./api/window.md), [Budget](./api/budget.md), [FLOP Cost Query](./api/flops.md), [Errors](./api/errors.md)
- **Reference:** [For AI Agents](./reference/for-agents.md), [Operation Audit](./reference/operation-audit.md), [FLOP Cost Cheat Sheet](./reference/cheat-sheet.md)
- **Troubleshooting:** [Common Errors](./troubleshooting/common-errors.md)
- **Changelog:** [Changelog](./changelog.md)
