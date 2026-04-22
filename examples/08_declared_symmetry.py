"""Declared tensor symmetry with the final exact-group API.

Run: uv run python examples/08_declared_symmetry.py
"""

import math

import numpy as np

import whest as we


print("=== Symmetric matrix (S2 on axes 0,1) ===\n")

n = 100
data = np.random.randn(n, n)
data = (data + data.T) / 2

matrix_group = we.SymmetryGroup.symmetric(axes=(0, 1))
A_sym = we.as_symmetric(data, symmetry=matrix_group)
print(f"Shape:            {A_sym.shape}")
print(f"Group order:      {A_sym.symmetry.order()}")
print(f"Unique elements:  {n * (n + 1) // 2:,} / {n * n:,}")

v = we.random.randn(n)
_, info = we.einsum_path("ij,j->i", A_sym, v)
_, info_dense = we.einsum_path("ij,j->i", we.array(data), v)
print(f"Matvec cost (symmetric): {info.optimized_cost:>10,} FLOPs")
print(f"Matvec cost (dense):     {info_dense.optimized_cost:>10,} FLOPs")

print("\n=== Rank-3 symmetric tensor + repeated operands ===\n")

n = 50
raw = np.random.randn(n, n, n)
T_data = (
    raw
    + raw.transpose(0, 2, 1)
    + raw.transpose(1, 0, 2)
    + raw.transpose(1, 2, 0)
    + raw.transpose(2, 0, 1)
    + raw.transpose(2, 1, 0)
) / 6

s3_group = we.SymmetryGroup.symmetric(axes=(0, 1, 2))
T = we.as_symmetric(T_data, symmetry=s3_group)
M = we.random.randn(n, n)
_, info_same = we.einsum_path("ijk,ai,bj,ck->abc", T, M, M, M)
M2 = we.random.randn(n, n)
M3 = we.random.randn(n, n)
_, info_diff = we.einsum_path("ijk,ai,bj,ck->abc", T, M, M2, M3)

print(f"Same M (declared + identity): {info_same.optimized_cost:>12,} FLOPs")
print(f"Diff M (declared only):       {info_diff.optimized_cost:>12,} FLOPs")
print(info_same.format_table(verbose=True))

print("=== SymmetryGroup types ===\n")

C3 = we.SymmetryGroup.cyclic(axes=(0, 1, 2))
D3 = we.SymmetryGroup.dihedral(axes=(0, 1, 2))

print(
    f"S3 (symmetric):  order={s3_group.order():<3}  abelian={s3_group.is_abelian}  transitive={s3_group.is_transitive}"
)
print(
    f"C3 (cyclic):     order={C3.order():<3}  abelian={C3.is_abelian}  transitive={C3.is_transitive}"
)
print(
    f"D3 (dihedral):   order={D3.order():<3}  abelian={D3.is_abelian}  transitive={D3.is_transitive}"
)

size_dict = {0: n, 1: n, 2: n}
print(f"\nUnique elements in {n}x{n}x{n} tensor:")
print(f"  Under S3: {s3_group.burnside_unique_count(size_dict):>8,}")
print(f"  Under C3: {C3.burnside_unique_count(size_dict):>8,}")
print(f"  Under D3: {D3.burnside_unique_count(size_dict):>8,}")
print(f"  No sym:   {n**3:>8,}")

T_cyclic = we.as_symmetric(T_data, symmetry=C3)
_, info_cyclic = we.einsum_path("ijk,ai,bj,ck->abc", T_cyclic, M, M, M)
print(f"\nWith S3 symmetry: {info_same.optimized_cost:>12,} FLOPs")
print(f"With C3 symmetry: {info_cyclic.optimized_cost:>12,} FLOPs")

print("\n=== Exact generators via SymmetryGroup.from_generators ===\n")

block_group = we.SymmetryGroup.from_generators(
    [[2, 3, 0, 1]],
    axes=(0, 1, 2, 3),
)
print(f"Block-swap generators: {block_group.generator_literals}")
print(f"Block-swap order:      {block_group.order()}")
print(f"Unique elements (20^4 with block swap): {block_group.burnside_unique_count({0: 20, 1: 20, 2: 20, 3: 20}):,}")

print("\n=== Output symmetry via einsum(symmetry=...) ===\n")

X = we.random.randn(20, 20)
with we.BudgetContext(flop_budget=10**9, quiet=True):
    result = we.einsum("ab,cd->abcd", X, X, symmetry=block_group)

print(f"Result type:    {type(result).__name__}")
print(f"Result shape:   {result.shape}")
print(f"Group order:    {result.symmetry.order()}")
print(
    "Verify block swap: "
    f"{np.allclose(np.array(result), np.array(result).transpose(2, 3, 0, 1))}"
)
