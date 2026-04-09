"""End-to-end integration tests for equal-operand induced symmetry.

Tests that me.einsum with repeated operands produces the expected symmetry-aware
FLOP costs. Uses hand-computed expected values.
"""

import numpy as np

import mechestim as me


class TestGramMatrixInduction:
    """einsum('ij,ik->jk', X, X) — the classic Gram matrix."""

    def test_plain_X_induces_s2_on_jk(self):
        n = 10
        X = np.ones((n, n))
        # dense = n * n * n * 2 = 2000
        # symmetric unique = n * C(n+1,2) = 10 * 55 = 550
        # total = n * n = 100
        # cost = 2000 * 550 / 1000 = 1100
        _, info_eq = me.einsum_path("ij,ik->jk", X, X)
        assert info_eq.optimized_cost == 1100

    def test_different_operands_dense_cost(self):
        n = 10
        X = np.ones((n, n))
        Y = np.ones((n, n))
        _, info = me.einsum_path("ij,ik->jk", X, Y)
        # Different operands → no induction → full dense
        assert info.optimized_cost == 2000


class TestMatMulChainNoInducedSymmetry:
    """einsum('ij,jk->ik', X, X) with plain (non-declared-symmetric) X.

    Regression guard: the old _detect_induced_output_symmetry incorrectly
    marked this as having S2(i,k) because its structural operand-matching
    heuristic treated "same Python object + matching index sets after
    relabel" as proof of symmetry. But X @ X is NOT symmetric in (i, k)
    unless X itself is symmetric — the output value R[i,k] = Σ_j X[i,j]·X[j,k]
    differs from R[k,i] = Σ_j X[k,j]·X[j,i] for a generic non-symmetric X.

    The subgraph symmetry oracle correctly rejects this case: passing the
    same Python object does not imply the tensor values are symmetric.
    Use me.as_symmetric() to declare symmetry explicitly — see
    TestSymmetricXMatMul below for the declared-symmetric case.
    """

    def test_plain_X_has_no_induced_symmetry(self):
        n = 10
        X = np.ones((n, n))
        _, info = me.einsum_path("ij,jk->ik", X, X)
        # dense = n^3 * 2 = 2000, no symmetry detected
        assert info.optimized_cost == 2000


class TestTripleProductInduction:
    """einsum('ij,ik,il->jkl', X, X, X) — three-way induction → S3."""

    def test_three_equal_operands_induce_s3(self):
        n = 10
        X = np.ones((n, n))
        _, info = me.einsum_path("ij,ik,il->jkl", X, X, X)
        # The path optimizer will pick a pairwise ordering.
        # Actual path chosen: (ik,ij->ikj) then (ikj,il->jkl)
        # Step 0: ik,ij→ikj. dense = n^4 * 2 = 2000.
        #   S2{j,k} induced from the equal pair (0,1).
        #   → step 0 cost = 1000 * 55/100 = 550.
        # Step 1: ikj,il→jkl. dense = n^4 * 2 = 20000.
        #   Propagated S2{j,k} from step 0. Induced S2{j,l}, S2{k,l}.
        #   All three merge to S3{j,k,l}. unique = C(12,3) = 220, total = 1000.
        #   → step 1 cost = 20000 * 220/1000 = 4400.
        # Total: 550 + 4400 = 4950
        assert info.optimized_cost == 4950


class TestBlockOuterProductInduction:
    """einsum('ijk,ilm->jklm', X, X) — block symmetry on (j,k) and (l,m)."""

    def test_block_s2_induction(self):
        n = 10
        X = np.ones((n, n, n))
        _, info = me.einsum_path("ijk,ilm->jklm", X, X)
        # Single step: ijk,ilm→jklm. dense = n^5 * 2 = 200000.
        # Block S2 [(j,k),(l,m)]: unique = C(n^2+1, 2) = C(101,2) = 5050
        # total = n^4 = 10000
        # cost = 200000 * 5050 / 10000 = 101000
        assert info.optimized_cost == 101000


class TestSymmetricXMatMul:
    """einsum('ij,jk->ik', X, X) where X is already declared symmetric.

    Combines per-operand symmetry (S2 on X's axes) with equal-operand detection
    (induced S2{i,k} on the output). Both sources should flow through the
    merge-aware propagate_symmetry.
    """

    def test_both_sources_apply(self):
        n = 10
        X_data = np.ones((n, n))
        X = me.as_symmetric(X_data, symmetric_axes=(0, 1))
        _, info = me.einsum_path("ij,jk->ik", X, X)
        # dense = n^3 * 2 = 2000
        # Induced S2{i,k} on output → unique = C(11,2) = 55, total = 100
        # cost = 2000 * 55/100 = 1100
        # (The per-op S2{i,j} on X0 and S2{j,k} on X1 don't contribute after
        # restriction — j is contracted. Only the induced S2{i,k} remains.)
        assert info.optimized_cost == 1100
