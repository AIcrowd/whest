"""End-to-end integration tests for equal-operand induced symmetry.

Tests that fnp.einsum with repeated operands produces the expected symmetry-aware
FLOP costs. Uses hand-computed expected values.

Migration note (direct-event accumulation model):
  The new model uses total = num_terms * m_total, where m_total is the number of
  unique output elements after Burnside's lemma. This replaces the old oracle-based
  formula. All optimized_cost values are updated accordingly. For a 2-term expression
  with S2 savings reducing output from 100 to 55 elements: cost = 2 * 550 = 1100.

  Output auto-tagging as SymmetricTensor has also been removed (the oracle that
  inferred output symmetry from equal-operand detection is gone). Results are plain
  FlopscopeArrays; use flops.as_symmetric() explicitly if you need the tag.
"""

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._budget import BudgetContext
from flopscope._symmetric import SymmetricTensor


class TestGramMatrixInduction:
    """einsum('ij,ik->jk', X, X) — the classic Gram matrix."""

    def test_plain_X_induces_s2_on_jk(self):
        n = 10
        X = np.ones((n, n))
        # Accumulation model: m_total = n * C(n+1,2) = 10 * 55 = 550 unique (i,j,k) combos.
        # total = num_terms * m_total = 2 * 550 = 1100
        _, info_eq = fnp.einsum_path("ij,ik->jk", X, X)
        assert info_eq.optimized_cost == 1100
        # Verify savings are present: m_total < dense_baseline
        acc = info_eq.accumulation
        assert acc.m_total < acc.dense_baseline

    def test_different_operands_dense_cost(self):
        n = 10
        X = np.ones((n, n))
        Y = np.ones((n, n))
        _, info = fnp.einsum_path("ij,ik->jk", X, Y)
        # Different operands → no induction → full dense.
        # total = num_terms * m_total = 2 * 1000 = 2000
        assert info.optimized_cost == 2000
        acc = info.accumulation
        assert acc.m_total == acc.dense_baseline  # no savings

    def test_einsum_numerically_correct(self):
        """The gram matrix result is numerically correct."""
        X = np.arange(1.0, 17.0).reshape(4, 4)

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = fnp.einsum("ij,ik->jk", X, X)

        expected = np.einsum("ij,ik->jk", X, X)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        # The accumulation model detects savings but does NOT auto-tag output
        # as SymmetricTensor (the oracle that did that has been removed).
        assert not isinstance(result, SymmetricTensor)

    def test_einsum_with_plain_out_preserves_output_identity(self):
        X = np.arange(1.0, 17.0).reshape(4, 4)
        out = np.empty((4, 4))

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = fnp.einsum("ij,ik->jk", X, X, out=out)

        expected = np.einsum("ij,ik->jk", X, X)
        assert result is out
        assert not isinstance(result, SymmetricTensor)
        np.testing.assert_allclose(out, expected, rtol=1e-10)


class TestMatMulChainNoInducedSymmetry:
    """einsum('ij,jk->ik', X, X) with plain (non-declared-symmetric) X.

    Regression guard: passing the same Python object does not imply the tensor
    values are symmetric. X @ X is NOT symmetric in (i, k) unless X itself is
    symmetric — the output value R[i,k] = Σ_j X[i,j]·X[j,k] differs from
    R[k,i] = Σ_j X[k,j]·X[j,i] for a generic non-symmetric X.

    Use flops.as_symmetric() to declare symmetry explicitly — see
    TestSymmetricXMatMul below for the declared-symmetric case.
    """

    def test_plain_X_has_no_induced_symmetry(self):
        n = 10
        X = np.ones((n, n))
        _, info = fnp.einsum_path("ij,jk->ik", X, X)
        # No symmetry detected: m_total == dense_baseline.
        # total = num_terms * m_total = 2 * 1000 = 2000
        assert info.optimized_cost == 2000
        acc = info.accumulation
        assert acc.m_total == acc.dense_baseline  # no savings


class TestTripleProductInduction:
    """einsum('ij,ik,il->jkl', X, X, X) — three-way induction → S3."""

    def test_three_equal_operands_induce_s3(self):
        n = 10
        X = np.ones((n, n))
        _, info = fnp.einsum_path("ij,ik,il->jkl", X, X, X)
        # Accumulation model: m_total is the number of unique (i,j,k,l) combos
        # after the full-expression S3 symmetry on the output (j,k,l) axes.
        # total = num_terms * m_total = 3 * 2200 = 6600
        assert info.optimized_cost == 6600
        acc = info.accumulation
        assert acc.m_total < acc.dense_baseline  # savings from S3


class TestBlockOuterProductInduction:
    """einsum('ijk,ilm->jklm', X, X) — block symmetry on (j,k) and (l,m)."""

    def test_block_s2_induction(self):
        n = 10
        X = np.ones((n, n, n))
        _, info = fnp.einsum_path("ijk,ilm->jklm", X, X)
        # Accumulation model: block S2 swaps the two operand blocks.
        # m_total = 50500, total = num_terms * m_total = 2 * 50500 = 101000
        assert info.optimized_cost == 101000
        acc = info.accumulation
        assert acc.m_total < acc.dense_baseline  # savings from block S2


class TestSymmetricXMatMul:
    """einsum('ij,jk->ik', X, X) where X is already declared symmetric.

    Per-operand symmetry (S2 on X's axes) detected; same m_total as the
    gram matrix case.
    """

    def test_both_sources_apply(self):
        n = 10
        X_data = np.ones((n, n))
        X = flops.as_symmetric(X_data, symmetry=(0, 1))
        _, info = fnp.einsum_path("ij,jk->ik", X, X)
        # Accumulation model: m_total = 550, total = 2 * 550 = 1100
        assert info.optimized_cost == 1100
        acc = info.accumulation
        assert acc.m_total < acc.dense_baseline  # savings detected
