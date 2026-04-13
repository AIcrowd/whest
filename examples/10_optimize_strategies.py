"""Comparing einsum optimize strategies.

``me.einsum_path`` accepts an ``optimize=`` parameter to choose how the
contraction path is found.  Different strategies trade planning time for
path quality.

Run: uv run python examples/10_optimize_strategies.py
"""

import time

import mechestim as me

# ---------------------------------------------------------------------------
# 1. Set up a 5-operand contraction (big enough for strategies to differ)
# ---------------------------------------------------------------------------
n = 50
A = me.random.randn(n, n)
B = me.random.randn(n, n)
C = me.random.randn(n, n)
D = me.random.randn(n, n)
E = me.random.randn(n, n)

subscripts = "ab,bc,cd,de,ef->af"
operands = (A, B, C, D, E)

# ---------------------------------------------------------------------------
# 2. Compare strategies
# ---------------------------------------------------------------------------
strategies = ["auto", "greedy", "optimal", "dp"]

print(f"=== Optimize strategies for: {subscripts} (n={n}) ===\n")
print(f"{'Strategy':<12} {'Cost':>14} {'Plan time':>12} {'Path'}")
print("-" * 65)

for name in strategies:
    t0 = time.perf_counter()
    path, info = me.einsum_path(subscripts, *operands, optimize=name)
    dt = time.perf_counter() - t0

    extra = f"  (resolved to {info.optimizer_used})" if name == "auto" else ""
    print(f"{name:<12} {info.optimized_cost:>14,} {dt * 1000:>10.2f}ms  {path}{extra}")

print(f"\nNaive cost (no optimization): {info.naive_cost:,}")

# ---------------------------------------------------------------------------
# 3. Plan once, execute many
# ---------------------------------------------------------------------------
print("\n=== Plan once, execute many ===\n")

path, info = me.einsum_path(subscripts, *operands, optimize="optimal")

with me.BudgetContext(flop_budget=10**9, quiet=True) as budget:
    for _ in range(10):
        result = me.einsum(subscripts, *operands, optimize=path)

print("10 executions with pre-planned path:")
print(f"  FLOPs per call: {budget.flops_used // 10:,}")
print(f"  Total FLOPs:    {budget.flops_used:,}")
print(f"  Result shape:   {result.shape}")
