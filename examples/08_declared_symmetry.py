"""Declared tensor symmetry — as_symmetric, PermutationGroup, and Cycle.

Tensors with known symmetries (e.g. a symmetric matrix where A[i,j] = A[j,i])
store fewer unique elements.  flopscope uses this to reduce FLOP counts in
einsum contractions automatically.

Run: uv run python examples/08_declared_symmetry.py
"""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from flopscope import Cycle, Permutation, PermutationGroup

# ---------------------------------------------------------------------------
# 1. Symmetric matrix — simple axis declaration
# ---------------------------------------------------------------------------
print("=== Symmetric matrix (S2 on axes 0,1) ===\n")

n = 100
data = np.random.randn(n, n)
data = (data + data.T) / 2  # make it actually symmetric

A_sym = flops.as_symmetric(data, symmetric_axes=(0, 1))
print(f"Shape:            {A_sym.shape}")
print(f"Symmetry factor:  {A_sym.symmetry_info.symmetry_factor}")
print(f"Unique elements:  {A_sym.symmetry_info.unique_elements:,} / {n * n:,}")

# einsum_path sees the symmetry and reduces cost
v = fnp.random.randn(n)
_, info = fnp.einsum_path("ij,j->i", A_sym, v)
_, info_dense = fnp.einsum_path("ij,j->i", fnp.array(data), v)
print(f"Matvec cost (symmetric): {info.optimized_cost:>10,} FLOPs")
print(f"Matvec cost (dense):     {info_dense.optimized_cost:>10,} FLOPs")

# ---------------------------------------------------------------------------
# 2. Fully-symmetric rank-3 tensor + same-object detection
# ---------------------------------------------------------------------------
print("\n=== Rank-3 symmetric tensor + repeated operands ===\n")

n = 50
raw = np.random.randn(n, n, n)
# Symmetrize over all 6 permutations of (i,j,k)
T_data = (
    raw
    + raw.transpose(0, 2, 1)
    + raw.transpose(1, 0, 2)
    + raw.transpose(1, 2, 0)
    + raw.transpose(2, 0, 1)
    + raw.transpose(2, 1, 0)
) / 6

T = flops.as_symmetric(T_data, symmetric_axes=(0, 1, 2))
M = fnp.random.randn(n, n)

# Pass M three times as the same object — einsum_path detects both
# the declared S3 symmetry on T and the identity-based savings from M=M=M
_, info_same = fnp.einsum_path("ijk,ai,bj,ck->abc", T, M, M, M)

# Compare: three DIFFERENT matrices
M2 = fnp.random.randn(n, n)
M3 = fnp.random.randn(n, n)
_, info_diff = fnp.einsum_path("ijk,ai,bj,ck->abc", T, M, M2, M3)

print(f"Same M (declared + identity): {info_same.optimized_cost:>12,} FLOPs")
print(f"Diff M (declared only):       {info_diff.optimized_cost:>12,} FLOPs")
print(info_same.format_table(verbose=True))

# ---------------------------------------------------------------------------
# 3. PermutationGroup constructors and introspection
# ---------------------------------------------------------------------------
print("=== PermutationGroup types ===\n")

S3 = PermutationGroup.symmetric(3, axes=(0, 1, 2))
C3 = PermutationGroup.cyclic(3, axes=(0, 1, 2))
D3 = PermutationGroup.dihedral(3, axes=(0, 1, 2))

print(
    f"S3 (symmetric):  order={S3.order():<3}  abelian={S3.is_abelian}  transitive={S3.is_transitive}"
)
print(
    f"C3 (cyclic):     order={C3.order():<3}  abelian={C3.is_abelian}  transitive={C3.is_transitive}"
)
print(
    f"D3 (dihedral):   order={D3.order():<3}  abelian={D3.is_abelian}  transitive={D3.is_transitive}"
)

# Burnside's lemma: count unique elements under group action
size_dict = {0: n, 1: n, 2: n}
print(f"\nUnique elements in {n}x{n}x{n} tensor:")
print(f"  Under S3: {S3.burnside_unique_count(size_dict):>8,}")
print(f"  Under C3: {C3.burnside_unique_count(size_dict):>8,}")
print(f"  Under D3: {D3.burnside_unique_count(size_dict):>8,}")
print(f"  No sym:   {n**3:>8,}")

# Cyclic symmetry gives less savings than full symmetric
T_cyclic = flops.as_symmetric(T_data, symmetry=C3)
_, info_cyclic = fnp.einsum_path("ijk,ai,bj,ck->abc", T_cyclic, M, M, M)
print(f"\nWith S3 symmetry: {info_same.optimized_cost:>12,} FLOPs")
print(f"With C3 symmetry: {info_cyclic.optimized_cost:>12,} FLOPs")

# ---------------------------------------------------------------------------
# 4. Cycle and Permutation — building permutations from cycles
# ---------------------------------------------------------------------------
print("\n=== Cycle and Permutation ===\n")

# Build a block-swap permutation: (0 2)(1 3) — swaps blocks (0,1) and (2,3)
block_swap = Permutation(Cycle(0, 2)(1, 3))
print(f"Block swap (0 2)(1 3): {block_swap.array_form}")
print(f"  Cyclic form:   {block_swap.cyclic_form}")
print(f"  Order:         {block_swap.order}")
print(f"  Parity:        {block_swap.parity()}")

# Build S2 from the block swap and use as output symmetry in einsum
# This declares that result[a,b,c,d] == result[c,d,a,b]
print("\n=== Output symmetry via einsum symmetry= parameter ===\n")

X = fnp.random.randn(20, 20)
block_group = PermutationGroup(block_swap, axes=(0, 1, 2, 3))

with flops.BudgetContext(flop_budget=10**9, quiet=True) as budget:
    # Outer product X[a,b]*X[c,d] is symmetric under block swap (a,b)↔(c,d)
    result = fnp.einsum("ab,cd->abcd", X, X, symmetry=block_group)

print(f"Result type: {type(result).__name__}")
print(f"Result shape: {result.shape}")
print(f"Symmetry factor: {result.symmetry_info.symmetry_factor}")
print(
    f"Verify: result[a,b,c,d] == result[c,d,a,b]? "
    f"{np.allclose(np.array(result), np.array(result).transpose(2, 3, 0, 1))}"
)
