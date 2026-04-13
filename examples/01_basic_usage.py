"""Basic whest usage — BudgetContext, free ops, counted ops, summary.

Run: uv run python examples/01_basic_usage.py
"""

import whest as we

with we.BudgetContext(flop_budget=10_000_000) as budget:
    # Free operations (0 FLOPs)
    W = we.ones((256, 256))
    x = we.ones((256,))

    # Counted operations
    h = we.einsum("ij,j->i", W, x)  # matrix-vector multiply
    h = we.maximum(h, 0)  # ReLU
    h = we.exp(h)  # exponential
    total = we.sum(h)  # reduction

    print(budget.summary())
    print(f"\nResult: {total}")
