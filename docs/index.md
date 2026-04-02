<img src="assets/logo/logo.png" alt="mechestim logo" style="height: 80px;">

# mechestim

**NumPy-compatible math primitives with analytical FLOP counting.**

!!! warning "mechestim is not a drop-in NumPy replacement"
    All computation requires a `BudgetContext`. Operations have analytical FLOP
    costs. 32 operations are blocked. See [Operation Categories](concepts/operation-categories.md).

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

```python
import mechestim as me

depth, width = 5, 256

with me.BudgetContext(flop_budget=10**8) as budget:
    # Build weight matrices (like nn.Linear layers)
    weights = []
    for _ in range(depth):
        fan_in = width
        W = me.random.randn(width, width)                   # free
        W = me.multiply(W, me.sqrt(me.array(2 / fan_in)))   # counted
        weights.append(W)

    # Forward pass
    x = me.random.randn(width)
    h = x
    for i, W in enumerate(weights):
        h = me.einsum('ij,j->i', W, h)    # linear layer
        if i < depth - 1:
            h = me.maximum(h, 0)           # ReLU
    print(budget.summary())
```

## Installation

```bash
uv add git+https://github.com/AIcrowd/mechestim.git
```

## Full Taxonomy

- **Getting Started:** [Installation](./getting-started/installation.md), [Your First Budget](./getting-started/first-budget.md)
- **How-To:** [Migrate from NumPy](./how-to/migrate-from-numpy.md), [Use Einsum](./how-to/use-einsum.md), [Exploit Symmetry](./how-to/exploit-symmetry.md), [Use Linear Algebra](./how-to/use-linalg.md), [Plan Your Budget](./how-to/plan-your-budget.md), [Debug Budget Overruns](./how-to/debug-budget-overruns.md)
- **Concepts:** [FLOP Counting Model](./concepts/flop-counting-model.md), [Operation Categories](./concepts/operation-categories.md)
- **Architecture:** [Client-Server Model](./architecture/client-server.md), [Running with Docker](./architecture/docker.md)
- **API Reference:** [Counted Ops](./api/counted-ops.md), [Free Ops](./api/free-ops.md), [Budget](./api/budget.md), [FLOP Cost Query](./api/flops.md), [Errors](./api/errors.md)
- **Reference:** [Operation Audit](./reference/operation-audit.md)
- **Troubleshooting:** [Common Errors](./troubleshooting/common-errors.md)
