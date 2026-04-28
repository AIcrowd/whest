"""Basic flopscope usage — BudgetContext, free ops, counted ops, summary.

Run: uv run python examples/01_basic_usage.py
"""

import flopscope as flops
import flopscope.numpy as fnp

with flops.BudgetContext(flop_budget=10_000_000) as budget:
    # Free operations (0 FLOPs)
    W = fnp.ones((256, 256))
    x = fnp.ones((256,))

    # Counted operations
    h = fnp.einsum("ij,j->i", W, x)  # matrix-vector multiply
    h = fnp.maximum(h, 0)  # ReLU
    h = fnp.exp(h)  # exponential
    total = fnp.sum(h)  # reduction

    print(budget.summary())
    print(f"\nResult: {total}")
