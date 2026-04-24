"""Truncated SVD usage and cost.

Run: uv run python examples/04_svd_usage.py
"""

import flopscope as flops
import flopscope.numpy as fnp
with flops.BudgetContext(flop_budget=10**8) as budget:
    A = fnp.random.randn(256, 256)

    U, S, Vt = fnp.linalg.svd(A, k=10)

    print(f"Input shape:  {A.shape}")
    print(f"U shape:      {U.shape}")
    print(f"S shape:      {S.shape}")
    print(f"Vt shape:     {Vt.shape}")
    print(f"FLOP cost:    {budget.flops_used:,}")
    print(f"\nPredicted cost: {flops.accounting.svd_cost(m=256, n=256, k=10):,}")
