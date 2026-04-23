"""End-to-end integration tests for equal-operand induced symmetry.

Tests that we.einsum with repeated operands produces the expected symmetry-aware
FLOP costs. Uses hand-computed expected values.
"""

import numpy as np

import whest as we
from whest._budget import BudgetContext
from whest._perm_group import SymmetryGroup
from whest._symmetric import SymmetricTensor


class TestGramMatrixInduction:
    """einsum('ij,ik->jk', X, X) — the classic Gram matrix."""

    def test_plain_X_induces_s2_on_jk(self):
        n = 10
        X = np.ones((n, n))
        # dense = n * n * n = 1000 (FMA=1)
        # symmetric unique = n * C(n+1,2) = 10 * 55 = 550
        # total = n * n = 100
        # cost = 1000 * 55/100 = 550
        _, info_eq = we.einsum_path("ij,ik->jk", X, X)
        assert info_eq.optimized_cost == 550

    def test_different_operands_dense_cost(self):
        n = 10
        X = np.ones((n, n))
        Y = np.ones((n, n))
        _, info = we.einsum_path("ij,ik->jk", X, Y)
        # Different operands → no induction → full dense (FMA=1)
        assert info.optimized_cost == 1000

    def test_einsum_infers_output_symmetry_from_path_info(self):
        X = np.arange(1.0, 17.0).reshape(4, 4)

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.einsum("ij,ik->jk", X, X)

        expected = np.einsum("ij,ik->jk", X, X)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry.axes == (0, 1)
        assert result.is_symmetric(symmetry=SymmetryGroup.symmetric(axes=(0, 1)))

    def test_einsum_with_plain_out_preserves_output_identity(self):
        X = np.arange(1.0, 17.0).reshape(4, 4)
        out = np.empty((4, 4))

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.einsum("ij,ik->jk", X, X, out=out)

        expected = np.einsum("ij,ik->jk", X, X)
        assert result is out
        assert not isinstance(result, SymmetricTensor)
        np.testing.assert_allclose(out, expected, rtol=1e-10)


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
    Use we.as_symmetric() to declare symmetry explicitly — see
    TestSymmetricXMatMul below for the declared-symmetric case.
    """

    def test_plain_X_has_no_induced_symmetry(self):
        n = 10
        X = np.ones((n, n))
        _, info = we.einsum_path("ij,jk->ik", X, X)
        # dense = n^3 = 1000 (FMA=1), no symmetry detected
        assert info.optimized_cost == 1000


class TestTripleProductInduction:
    """einsum('ij,ik,il->jkl', X, X, X) — three-way induction → S3."""

    def test_three_equal_operands_induce_s3(self):
        n = 10
        X = np.ones((n, n))
        _, info = we.einsum_path("ij,ik,il->jkl", X, X, X)
        # Path: (ik,ij->ikj) then (ikj,il->jkl), FMA=1.
        # Step 0: dense = n^3 = 1000, S2 savings: 1000 * 55/100 = 550.
        # Step 1: dense = n^4 = 10000, S3 savings: 10000 * 220/1000 = 2200.
        # Total: 550 + 2200 = 2750
        assert info.optimized_cost == 2750


class TestBlockOuterProductInduction:
    """einsum('ijk,ilm->jklm', X, X) — block symmetry on (j,k) and (l,m)."""

    def test_block_s2_induction(self):
        n = 10
        X = np.ones((n, n, n))
        _, info = we.einsum_path("ijk,ilm->jklm", X, X)
        # Single step: ijk,ilm→jklm. dense = n^5 = 100000 (FMA=1).
        # Direct evaluation: output has G(2){j,k,l,m} from same-object
        # detection (block S2 on {j,k} and {l,m}).
        # unique_output / total_output = 5050 / 10000 → cost = 50500.
        assert info.optimized_cost == 50500


class TestSymmetricXMatMul:
    """einsum('ij,jk->ik', X, X) where X is already declared symmetric.

    Combines per-operand symmetry (S2 on X's axes) with equal-operand detection
    (induced S2{i,k} on the output). Both sources should flow through the
    merge-aware propagate_symmetry.
    """

    def test_both_sources_apply(self):
        n = 10
        X_data = np.ones((n, n))
        X = we.as_symmetric(X_data, symmetry=(0, 1))
        _, info = we.einsum_path("ij,jk->ik", X, X)
        # dense = n^3 = 1000 (FMA=1)
        # Induced S2{i,k} on output → unique = C(11,2) = 55, total = 100
        # cost = 1000 * 55/100 = 550
        assert info.optimized_cost == 550
