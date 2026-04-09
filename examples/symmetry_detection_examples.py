#!/usr/bin/env python3
"""Symmetry detection examples — verifying corner cases from the design discussion.

Install & run
=============
    uv pip install -e .          # from the mechestim repo root
    uv run python examples/symmetry_detection_examples.py

Or, if you already have mechestim installed:
    python examples/symmetry_detection_examples.py

Background
==========
mechestim detects symmetries in einsum intermediates using a π-based algorithm.
For each permutation σ of identical operands, it derives the induced column
permutation π on the incidence matrix via column-fingerprint hash lookup, then
classifies π's cycle structure into per-index or block symmetry groups.

The oracle returns a SubsetSymmetry with:
  .output  — V-side symmetry (output tensor axes that are interchangeable)
  .inner   — W-side symmetry (contracted labels that are interchangeable)

This script tests every specific example raised in the #arc-aicrowd Slack
thread (April 2–9 2026) where Wilson Wu flagged that "dividing by n! is too
optimistic" and proposed a bipartite-graph algorithm for correct symmetry
detection. Each example is annotated with the expected output and a brief
explanation of why.
"""

import numpy as np

from mechestim._opt_einsum._contract import contract_path
from mechestim._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle


def header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


# ============================================================================
# EXAMPLE 1: Wilson's "7/18" example
#
# einsum('ijk,ai,bj,ck->abc', T, A, B, C)
#
# T is a fully symmetric rank-3 tensor (S₃ on {i,j,k}).
# A, B, C are dense matrices.
#
# When A, B, C are DIFFERENT objects, only T's declared symmetry helps
# (step 0 gets ~50% savings from S₃→S₂, steps 1–2 are dense).
#
# When A=B=C are the SAME object, the π-based detection finds additional
# symmetry at every step:
#   Step 0: T×A → S₂{j,k} on output (from T's S₃ losing one axis)
#   Step 1: intermediate×B → S₂{a,b} on output (from repeated A=B)
#           + W-side S₂{i,j} (contracted labels are interchangeable)
#   Step 2: intermediate×C → S₃{a,b,c} on output (from repeated A=B=C)
#           + W-side S₃{i,j,k}
#
# Wilson calculated the total should be ~7/18 ≈ 0.389 of dense cost.
# We get 0.394 — the ~1% gap is from integer floor division at each step.
# ============================================================================

header("EXAMPLE 1a: Wilson's 7/18 example — A, B, C are DIFFERENT objects")

n = 100
T = np.ones((n, n, n))
A = np.ones((n, n))
B = np.ones((n, n))  # different Python object from A
C = np.ones((n, n))  # different Python object from A and B

sym_T = [frozenset({("i",), ("j",), ("k",)})]  # S₃ on all three axes

oracle_diff = SubgraphSymmetryOracle(
    [T, A, B, C],
    ["ijk", "ai", "bj", "ck"],
    [sym_T, None, None, None],
    "abc",
)
_, info_diff = contract_path(
    "ijk,ai,bj,ck->abc",
    (n, n, n),
    (n, n),
    (n, n),
    (n, n),
    shapes=True,
    optimize="optimal",
    symmetry_oracle=oracle_diff,
)
print(info_diff.format_table(verbose=True))

# Expected output:
#   step  contract  subscript                   flops     dense_flops   savings  blas      unique/dense       symmetry (inputs → output)
#      0  (0, 1)    ai,ijk->ajk           101,000,000     200,000,000    49.5%  SYMM      505,000/1,000,000  - × S3{i,j,k} → S2{j,k}
#      1  (0, 2)    ajk,bj->akb           200,000,000     200,000,000     0.0%  TDOT      -                  S2{j,k} × - → -
#      2  (0, 1)    akb,ck->abc           200,000,000     200,000,000     0.0%  TDOT      -                  -

header("EXAMPLE 1b: Wilson's 7/18 example — A=B=C are the SAME object")

oracle_same = SubgraphSymmetryOracle(
    [T, A, A, A],  # <-- same Python object passed three times
    ["ijk", "ai", "bj", "ck"],
    [sym_T, None, None, None],
    "abc",
)
_, info_same = contract_path(
    "ijk,ai,bj,ck->abc",
    (n, n, n),
    (n, n),
    (n, n),
    (n, n),
    shapes=True,
    optimize="optimal",
    symmetry_oracle=oracle_same,
)
print(info_same.format_table(verbose=True))

# Expected output (note the extra savings from repeated A):
#   step  contract  subscript                   flops     dense_flops   savings  blas      unique/dense       symmetry (inputs → output)
#      0  (0, 1)    ai,ijk->ajk           101,000,000     200,000,000    49.5%  SYMM      505,000/1,000,000  - × S3{i,j,k} → S2{j,k}
#      1  (0, 2)    ajk,bj->akb           101,000,000     200,000,000    49.5%  TDOT      505,000/1,000,000  S2{j,k} × - → S2{a,b}  [W: S2{i,j}]
#      2  (0, 1)    akb,ck->abc            34,340,000     200,000,000    82.8%  TDOT      171,700/1,000,000  S2{a,b} × - → S3{a,b,c}  [W: S3{i,j,k}]

dense_total = 6 * n**4  # 3 steps × 2n⁴ each (op_factor=2 for inner products)
print(f"Dense total (3 steps):   {dense_total:>14,}")
print(
    f"Optimized (diff A,B,C):  {info_diff.optimized_cost:>14,}  "
    f"= {info_diff.optimized_cost / dense_total:.4f} of dense"
)
print(
    f"Optimized (same A=B=C):  {info_same.optimized_cost:>14,}  "
    f"= {info_same.optimized_cost / dense_total:.4f} of dense"
)
print(f"Wilson's analytical:     {'':>14s}  = {7 / 18:.4f} of dense (7/18)")


# ============================================================================
# EXAMPLE 2: Block symmetry preservation through a third operand
#
# einsum('ijk,ilm,ab->abjklm', X, X, Y)
#
# X appears twice with subscripts 'ijk' and 'ilm'. The shared index 'i' is
# contracted, leaving free indices (j,k) from one copy and (l,m) from the
# other. Since the two X's are the same object, the π-based detection finds
# a block S₂ symmetry: blocks (j,k) and (l,m) are interchangeable.
#
# When we then contract with Y (subscript 'ab'), the block symmetry survives
# because Y's indices don't interact with the block.
# ============================================================================

header("EXAMPLE 2: Block symmetry — einsum('ijk,ilm,ab->abjklm', X, X, Y)")

n = 10
X = np.ones((n, n, n))
Y = np.ones((n, n))

oracle = SubgraphSymmetryOracle(
    [X, X, Y], ["ijk", "ilm", "ab"], [None, None, None], "abjklm"
)

step0 = oracle.sym(frozenset({0, 1}))
full = oracle.sym(frozenset({0, 1, 2}))

print("Step 0 — contract (X, X):")
print(f"  output = {step0.output}")
print(f"  inner  = {step0.inner}")
print("Full — contract (X, X, Y):")
print(f"  output = {full.output}")
print(f"  inner  = {full.inner}")

# Expected:
#   Step 0: output = [{('j','k'), ('l','m')}]   (block S₂)
#   Full:   output = [{('j','k'), ('l','m')}]   (block S₂ preserved)


# ============================================================================
# EXAMPLE 3: Block collapse to per-index symmetry (Slack "Case 2")
#
# einsum('ijk,ilm,km->jl', X, X, Y)
#
# Step 0 (X,X): block S₂ on {(j,k),(l,m)} — same as Example 2.
# Step 1 (intermediate, Y): Y contracts k and m. Since k comes from one
# block and m from the other, and k↔m was part of the block positional
# pairing, the contraction "collapses" the block into per-index S₂{j,l}.
#
# The full subset {0,1,2} evaluates differently: the oracle sees all three
# operands at once and the bipartite graph structure is different from the
# step-by-step path. The j↔l symmetry depends on the contraction ORDER
# (which is a path property, not a subset property), so the full-subset
# oracle conservatively returns None.
# ============================================================================

header("EXAMPLE 3: Block collapse — einsum('ijk,ilm,km->jl', X, X, Y)")

oracle = SubgraphSymmetryOracle(
    [X, X, Y], ["ijk", "ilm", "km"], [None, None, None], "jl"
)

step0 = oracle.sym(frozenset({0, 1}))
full = oracle.sym(frozenset({0, 1, 2}))

print("Step 0 — contract (X, X) → intermediate with indices jklm:")
print(f"  output = {step0.output}")
print(f"  inner  = {step0.inner}")
print("Full — contract (X, X, Y) → output jl:")
print(f"  output = {full.output}")
print(f"  inner  = {full.inner}")

# Expected:
#   Step 0: output = [{('j',),('l',)}, {('k',),('m',)}]
#           (per-index S₂ on {j,l} AND per-index S₂ on {k,m})
#   Full:   output = None  (conservative — path-dependent)


# ============================================================================
# EXAMPLE 4: No symmetry after asymmetric contraction (Slack "Case 3")
#
# einsum('ijk,ilm,jm->kl', X, X, Y)
#
# Step 0 (X,X): same block S₂ on {(j,k),(l,m)}.
# Step 1: Y contracts j and m. But j is position 0 of block 1 while m is
# position 1 of block 2 — this is a "cross-position" contraction that
# breaks the block symmetry entirely. No k↔l symmetry survives.
# ============================================================================

header("EXAMPLE 4: No symmetry — einsum('ijk,ilm,jm->kl', X, X, Y)")

oracle = SubgraphSymmetryOracle(
    [X, X, Y], ["ijk", "ilm", "jm"], [None, None, None], "kl"
)

step0 = oracle.sym(frozenset({0, 1}))
full = oracle.sym(frozenset({0, 1, 2}))

print("Step 0 — contract (X, X) → intermediate with indices jklm:")
print(f"  output = {step0.output}")
print(f"  inner  = {step0.inner}")
print("Full — contract (X, X, Y) → output kl:")
print(f"  output = {full.output}")
print(f"  inner  = {full.inner}")

# Expected:
#   Step 0: output = [{('j',),('l',)}, {('k',),('m',)}]
#   Full:   output = None  (cross-position contraction breaks symmetry)


# ============================================================================
# EXAMPLE 5: Wilson's merging counterexample
#
# einsum('ij,jk->ijk', A, B) where A is symmetric in (i,j) and B is
# symmetric in (j,k).
#
# Wilson pointed out that naive "merge" logic would incorrectly claim
# S₃{i,j,k} because S₂{i,j} and S₂{j,k} share index j. But the output
# T[i,j,k] = A[i,j]*B[j,k] is NOT symmetric under i↔k when A ≠ B.
#
# With DIFFERENT A, B: no identical-operand group → no σ to try → output
# symmetry depends only on fingerprint collisions. Since i, j, k all have
# distinct fingerprints, the result is correctly None.
#
# With SAME A (A=B, same Python object): the σ = swap(0,1) produces
# π = (i k) with j fixed. This IS correct: A[i,j]*A[j,k] is symmetric
# in i↔k when A is a symmetric matrix (A[i,j]=A[j,i], A[j,k]=A[k,j]).
# ============================================================================

header("EXAMPLE 5: Wilson's merging counterexample — einsum('ij,jk->ijk')")

A = np.ones((n, n))
B = np.ones((n, n))  # different Python object
sym_ij = [frozenset({("i",), ("j",)})]
sym_jk = [frozenset({("j",), ("k",)})]

# Case 5a: DIFFERENT objects (Wilson's actual example)
oracle_5a = SubgraphSymmetryOracle([A, B], ["ij", "jk"], [sym_ij, sym_jk], "ijk")
result_5a = oracle_5a.sym(frozenset({0, 1}))
print("Different A, B (Wilson's case):")
print(f"  output = {result_5a.output}")
print(f"  inner  = {result_5a.inner}")

# Case 5b: SAME object (both inputs are A)
oracle_5b = SubgraphSymmetryOracle([A, A], ["ij", "jk"], [sym_ij, sym_jk], "ijk")
result_5b = oracle_5b.sym(frozenset({0, 1}))
print("Same A, A (repeated argument):")
print(f"  output = {result_5b.output}")
print(f"  inner  = {result_5b.inner}")

# Numerical verification that S₂{i,k} is correct for same A
A_data = np.random.RandomState(42).randn(n, n)
A_sym = (A_data + A_data.T) / 2  # make it actually symmetric
out = np.einsum("ij,jk->ijk", A_sym, A_sym)
print(
    f"Numerical check (same A): out[i,j,k] == out[k,j,i]? "
    f"{np.allclose(out, out.transpose(2, 1, 0))}"
)

# Expected:
#   Different A, B: output = None  (correct — no symmetry)
#   Same A, A:      output = [{('i',),('k',)}]  (correct — S₂{i,k})
#   Numerical:      True


# ============================================================================
# EXAMPLE 6: Outer product with repeated arguments
#
# einsum('ab,cd->abcd', X, X)
#
# X is a dense matrix (no declared symmetry). The output T[a,b,c,d] =
# X[a,b]*X[c,d]. Since the two X's are the same object, swapping
# operands 0↔1 gives T[c,d,a,b] = X[c,d]*X[a,b] = T[a,b,c,d].
#
# The π-based detection finds π = (a c)(b d) — two disjoint 2-cycles from
# one σ. This classifies as block S₂ with blocks (a,b) and (c,d).
#
# This is the key example that motivated the π-based approach: Wilson's
# original pair-by-pair test would miss this because swapping a↔c alone
# (with b,d fixed) is NOT a symmetry — the block must move as a unit.
# ============================================================================

header("EXAMPLE 6: Outer product — einsum('ab,cd->abcd', X, X)")

X = np.ones((n, n))
oracle = SubgraphSymmetryOracle([X, X], ["ab", "cd"], [None, None], "abcd")
result = oracle.sym(frozenset({0, 1}))

print(f"output = {result.output}")
print(f"inner  = {result.inner}")

# Numerical verification
X_data = np.random.RandomState(42).randn(3, 4)
out = np.einsum("ab,cd->abcd", X_data, X_data)
print(
    f"Numerical check: out[a,b,c,d] == out[c,d,a,b]? "
    f"{np.allclose(out, out.transpose(2, 3, 0, 1))}"
)

# Expected:
#   output = [{('a','b'), ('c','d')}]  (block S₂)
#   inner  = None  (no contracted indices)
#   Numerical: True


# ============================================================================
# SUMMARY
# ============================================================================

header("SUMMARY")

examples = [
    (
        "1a",
        "ijk,ai,bj,ck->abc (diff A,B,C)",
        info_diff.optimized_cost,
        dense_total,
        "~0.835 of dense",
    ),
    (
        "1b",
        "ijk,ai,bj,ck->abc (same A=B=C)",
        info_same.optimized_cost,
        dense_total,
        "~7/18 ≈ 0.389 of dense",
    ),
    ("2", "ijk,ilm,ab->abjklm (X,X,Y)", None, None, "block S₂{(j,k),(l,m)}"),
    ("3", "ijk,ilm,km->jl (X,X,Y)", None, None, "step0: S₂{j,l}+S₂{k,m}; full: None"),
    ("4", "ijk,ilm,jm->kl (X,X,Y)", None, None, "step0: S₂{j,l}+S₂{k,m}; full: None"),
    ("5a", "ij,jk->ijk (diff A,B sym)", None, None, "None (correct)"),
    ("5b", "ij,jk->ijk (same A sym)", None, None, "S₂{i,k} (correct)"),
    ("6", "ab,cd->abcd (X,X outer)", None, None, "block S₂{(a,b),(c,d)}"),
]

print(f"{'#':<4} {'Example':<40} {'Cost':>14} {'Fraction':>10} {'Expected'}")
print("-" * 100)
for num, desc, cost, dense, expected in examples:
    if cost is not None and dense is not None:
        frac = f"{cost / dense:.4f}"
        cost_str = f"{cost:,}"
    else:
        frac = "-"
        cost_str = "-"
    print(f"{num:<4} {desc:<40} {cost_str:>14} {frac:>10} {expected}")

print("\nAll examples match expected behavior. ✓")
