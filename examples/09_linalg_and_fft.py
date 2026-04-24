"""Linear algebra and FFT cost tracking.

flopscope tracks FLOPs for linalg decompositions, solvers, and FFT
operations — not just einsum.  Each function also has a ``_cost``
companion for pre-flight estimation.

Run: uv run python examples/09_linalg_and_fft.py
"""

import flopscope as flops
import flopscope.numpy as fnp
# ---------------------------------------------------------------------------
# 1. Decomposition cost comparison
# ---------------------------------------------------------------------------
print("=== Decomposition costs (256x256 matrix) ===\n")

n = 256

decomps = [
    ("SVD", flops.accounting.svd_cost(n, n)),
    ("SVD (k=16)", flops.accounting.svd_cost(n, n, k=16)),
    ("QR", flops.accounting.qr_cost(n, n)),
    ("Cholesky", flops.accounting.cholesky_cost(n)),
    ("Eigh", flops.accounting.eigh_cost(n)),
]

print(f"{'Decomposition':<16} {'FLOPs':>14}")
print("-" * 32)
for name, cost in decomps:
    print(f"{name:<16} {cost:>14,}")

# ---------------------------------------------------------------------------
# 2. Solve vs. inverse — solve is cheaper
# ---------------------------------------------------------------------------
print("\n=== Solve vs. inverse (256x256) ===\n")

solve_cost = flops.accounting.solve_cost(n, n)
inv_cost = flops.accounting.inv_cost(n)
matmul_cost = flops.accounting.einsum_cost("ij,jk->ik", shapes=[(n, n), (n, n)])

print(f"solve(A, B):                {solve_cost:>14,} FLOPs")
print(f"inv(A) + matmul(A_inv, B):  {inv_cost + matmul_cost:>14,} FLOPs")

# Verify with actual execution
with flops.BudgetContext(flop_budget=10**10, quiet=True) as budget:
    A = fnp.random.randn(n, n)
    A = fnp.einsum("ij,kj->ik", A, A)  # make positive-definite-ish
    b = fnp.random.randn(n, n)
    x = fnp.linalg.solve(A, b)

print(f"Actual solve:               {budget.flops_used:>14,} FLOPs")

# ---------------------------------------------------------------------------
# 3. FFT operations
# ---------------------------------------------------------------------------
print("\n=== FFT costs ===\n")

signal_len = 4096
fft_cost = flops.accounting.fft_cost(signal_len)
rfft_cost = flops.accounting.rfft_cost(signal_len)

print(f"FFT  (n={signal_len}): {fft_cost:>10,} FLOPs")
print(f"RFFT (n={signal_len}): {rfft_cost:>10,} FLOPs  (real-valued input)")

with flops.BudgetContext(flop_budget=10**8, quiet=True) as budget:
    signal = fnp.random.randn(signal_len)
    spectrum = fnp.fft.rfft(signal)

print(f"Actual RFFT:      {budget.flops_used:>10,} FLOPs")
print(f"Spectrum shape:   {spectrum.shape}")

# ---------------------------------------------------------------------------
# 4. Multi-dot optimal ordering
# ---------------------------------------------------------------------------
print("\n=== multi_dot — optimal chain multiplication ===\n")

# A(10x200) @ B(200x5) @ C(5x300) — order matters for cost
shapes = [(10, 200), (200, 5), (5, 300)]
optimal_cost = flops.accounting.multi_dot_cost(shapes)

# Naive left-to-right: (A@B)@C
left_right = flops.accounting.einsum_cost(
    "ij,jk->ik", shapes=[shapes[0], shapes[1]]
) + flops.accounting.einsum_cost("ij,jk->ik", shapes=[(10, 5), shapes[2]])

print(f"Optimal (multi_dot): {optimal_cost:>10,} FLOPs")
print(f"Left-to-right:       {left_right:>10,} FLOPs")

with flops.BudgetContext(flop_budget=10**8, quiet=True) as budget:
    A = fnp.random.randn(*shapes[0])
    B = fnp.random.randn(*shapes[1])
    C = fnp.random.randn(*shapes[2])
    result = fnp.linalg.multi_dot([A, B, C])

print(f"Actual multi_dot:    {budget.flops_used:>10,} FLOPs")
print(f"Result shape:        {result.shape}")
