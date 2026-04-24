"""Contraction path analysis with einsum_path.

``fnp.einsum_path`` computes the optimal contraction order *without*
executing or spending any budget.  It returns a ``(path, PathInfo)``
tuple that you can inspect and later feed back to ``fnp.einsum``.

Run: uv run python examples/07_einsum_path.py
"""

import flopscope as flops
import flopscope.numpy as fnp
# ---------------------------------------------------------------------------
# 1. Path analysis for a three-matrix chain multiply
# ---------------------------------------------------------------------------
n = 200
A = fnp.random.randn(n, n)
B = fnp.random.randn(n, n)
C = fnp.random.randn(n, n)

path, info = fnp.einsum_path("ij,jk,kl->il", A, B, C)

print("=== Three-matrix chain: ij,jk,kl->il ===\n")
print(info.format_table(verbose=True))
print(f"Contraction path: {path}")
print(f"Naive cost:       {info.naive_cost:>14,} FLOPs")
print(f"Optimized cost:   {info.optimized_cost:>14,} FLOPs")
print(f"Speedup:          {info.speedup:.1f}x")

# ---------------------------------------------------------------------------
# 2. Automatic same-object symmetry detection
# ---------------------------------------------------------------------------
# When the same Python object appears multiple times, einsum_path detects
# that the operands are identical and exploits resulting symmetries.

print("\n=== Same-object detection: ij,ik->jk ===\n")

X = fnp.random.randn(100, 100)
Y = fnp.random.randn(100, 100)  # different object, same shape

_, info_same = fnp.einsum_path("ij,ik->jk", X, X)  # same object twice
_, info_diff = fnp.einsum_path("ij,ik->jk", X, Y)  # two different objects

print(f"Same object (X, X):  {info_same.optimized_cost:>10,} FLOPs")
print(f"Diff objects (X, Y): {info_diff.optimized_cost:>10,} FLOPs")
savings = 1 - info_same.optimized_cost / info_diff.optimized_cost
print(f"Symmetry savings:    {savings:.0%}")

# ---------------------------------------------------------------------------
# 3. Re-use a pre-computed path
# ---------------------------------------------------------------------------
# Compute the path once, then execute many times with that exact order.

print("\n=== Execute with pre-computed path ===\n")

with flops.BudgetContext(flop_budget=10**9) as budget:
    result = fnp.einsum("ij,jk,kl->il", A, B, C, optimize=path)

print(f"Result shape:  {result.shape}")
print(f"FLOPs used:    {budget.flops_used:,}")
