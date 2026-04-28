"""Basic smoke test — verifies flopscope works under lockdown."""

import flopscope as flops
import flopscope.numpy as fnp

with flops.BudgetContext(flop_budget=10_000_000) as budget:
    W = fnp.ones((256, 256))
    x = fnp.ones((256,))
    h = fnp.einsum("ij,j->i", W, x)
    h = fnp.maximum(h, 0)
    total = fnp.sum(h)
    print(budget.summary())
    print(f"Result: {total}")
