"""Comprehensive corner-case tests for subgraph symmetry detection.

Tests grouped by category, each with mathematical justification for the
expected result. These tests verify the algorithm against hand-worked
expectations and numerical verification.
"""

from __future__ import annotations

import numpy as np
import pytest

import whest as we
from whest._perm_group import PermutationGroup


def _symmetrize(shape, group):
    """Reynolds-average random data to enforce symmetry."""
    data = np.random.randn(*shape)
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    total = np.zeros_like(data)
    for g in group.elements():
        af = g.array_form
        perm = list(range(len(shape)))
        for i in range(len(axes)):
            perm[axes[i]] = axes[af[i]]
        total = total + np.transpose(data, perm)
    return we.as_symmetric(total / group.order(), symmetry=group)


def _detect_order(expr, *operands):
    """Run einsum_path and return the max detected group order across all steps.

    Multi-operand contractions are decomposed into pairwise steps; the full
    symmetry may only appear in the final step that combines all operands.
    """
    _, info = we.einsum_path(expr, *operands)
    max_order = 1
    for step in info.steps:
        if step.output_group:
            max_order = max(max_order, step.output_group.order())
        if step.inner_group:
            max_order = max(max_order, step.inner_group.order())
    return max_order


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


N = 4


class TestIdenticalOperands:
    """Symmetry from identical Python objects at multiple operand positions."""

    def test_matmul_no_symmetry(self):
        """A·A = einsum('ij,jk->ik', A, A): different subscript structure, trivial."""
        A = we.random.randn(N, N)
        assert _detect_order("ij,jk->ik", A, A) == 1

    def test_gram_matrix_s2(self):
        """X^T X = einsum('ia,ib->ab', X, X): swapping operands swaps a↔b → S2."""
        X = we.random.randn(N, N)
        assert _detect_order("ia,ib->ab", X, X) == 2
        result = we.einsum("ia,ib->ab", X, X)
        assert np.allclose(result, result.T)

    def test_vector_outer_s2(self):
        """v⊗v = einsum('i,j->ij', v, v): S2{i,j}."""
        v = we.random.randn(N)
        assert _detect_order("i,j->ij", v, v) == 2
        result = we.einsum("i,j->ij", v, v)
        assert np.allclose(result, result.T)

    def test_triple_outer_s3(self):
        """v⊗v⊗v = einsum('i,j,k->ijk', v, v, v): S3{i,j,k}, order 6."""
        v = we.random.randn(N)
        assert _detect_order("i,j,k->ijk", v, v, v) == 6

    def test_quad_outer_s4(self):
        """v⊗v⊗v⊗v: S4{i,j,k,l}, order 24."""
        v = we.random.randn(N)
        assert _detect_order("i,j,k,l->ijkl", v, v, v, v) == 24

    def test_directed_triangle_c3(self):
        """einsum('ij,jk,ki->ijk', A, A, A): cyclic chain → C3, not S3.

        The cyclic permutation of operands (ij,jk,ki)→(jk,ki,ij) is valid,
        but reflections are not (would require reversing the chain direction).
        """
        A = we.random.randn(N, N)
        assert _detect_order("ij,jk,ki->ijk", A, A, A) == 3

    def test_directed_4_cycle_c4(self):
        """einsum('ij,jk,kl,li->ijkl', A, A, A, A): C4, order 4."""
        A = we.random.randn(N, N)
        assert _detect_order("ij,jk,kl,li->ijkl", A, A, A, A) == 4

    def test_block_outer_product(self):
        """einsum('ab,cd->abcd', X, X): block swap (a,b)↔(c,d), order 2."""
        X = we.random.randn(N, N)
        assert _detect_order("ab,cd->abcd", X, X) == 2

    def test_hadamard_no_symmetry(self):
        """einsum('ij,ij->ij', A, A): elementwise square.

        Swapping operands gives identity π (same subscripts), so trivial.
        A²[i,j] ≠ A²[j,i] in general.
        """
        A = we.random.randn(N, N)
        assert _detect_order("ij,ij->ij", A, A) == 1
        result = we.einsum("ij,ij->ij", A, A)
        assert not np.allclose(result, result.T)


class TestDeclaredSymmetryNonIdentical:
    """Per-operand declared symmetry on non-identical operands."""

    def test_symmetric_times_dense_trivial(self):
        """S·W where S has S2: no output symmetry (non-identical operands, no
        identical-operand source)."""
        S = _symmetrize((N, N), PermutationGroup.symmetric(2, axes=(0, 1)))
        W = we.random.randn(N, N)
        assert _detect_order("ij,jk->ik", S, W) == 1

    def test_c3_contraction(self):
        """einsum('aijk,ab->ijkb', T, W) with C3 on T axes (1,2,3).

        T has C3, W is dense, non-identical operands. Source A produces the
        C3 generator on T's axes, which induces C3{i,j,k} on the output.
        """
        T = _symmetrize((N, N, N, N), PermutationGroup.cyclic(3, axes=(1, 2, 3)))
        W = we.random.randn(N, N)
        assert _detect_order("aijk,ab->ijkb", T, W) == 3

    def test_d4_contraction(self):
        """einsum('aijkl,ab->ijklb', T, W) with D4 on T axes (1,2,3,4).

        D4 has order 8 (4 rotations + 4 reflections).
        """
        T = _symmetrize(
            (N, N, N, N, N), PermutationGroup.dihedral(4, axes=(1, 2, 3, 4))
        )
        W = we.random.randn(N, N)
        assert _detect_order("aijkl,ab->ijklb", T, W) == 8


class TestDeclaredPlusIdentical:
    """Interaction of per-operand symmetry with identical-operand swaps."""

    def test_symmetric_matmul_s2(self):
        """S·S = einsum('ij,jk->ik', S, S) where S is symmetric.

        S[i,j]=S[j,i]. Result R[i,k] = sum_j S[i,j]*S[j,k].
        Using S=S^T: R[k,i] = sum_j S[k,j]*S[j,i] = sum_j S[j,k]*S[i,j] = R[i,k].
        So the result IS symmetric → S2{i,k} is correct.
        """
        S = _symmetrize((N, N), PermutationGroup.symmetric(2, axes=(0, 1)))
        assert _detect_order("ij,jk->ik", S, S) == 2
        result = we.einsum("ij,jk->ik", S, S)
        assert np.allclose(result, result.T)

    def test_undirected_4_cycle_d4(self):
        """einsum('ij,jk,kl,li->ijkl', S, S, S, S) where S is symmetric.

        S symmetric enables reflections: C4 + reflections = D4, order 8.
        """
        S = _symmetrize((N, N), PermutationGroup.symmetric(2, axes=(0, 1)))
        assert _detect_order("ij,jk,kl,li->ijkl", S, S, S, S) == 8

    def test_c3_self_contraction_trivial(self):
        """einsum('ijk,jki->ik', T, T) with C3 on T: MUST be trivial.

        REGRESSION TEST: old code falsely detected S2{i,k}.
        C3 has no transpositions, so orbit-based merging was wrong.
        Numerically: Result[i,k] ≠ Result[k,i].
        """
        T = _symmetrize((N, N, N), PermutationGroup.cyclic(3, axes=(0, 1, 2)))
        assert _detect_order("ijk,jki->ik", T, T) == 1
        result = we.einsum("ijk,jki->ik", T, T)
        assert not np.allclose(result, result.T)

    def test_symmetric_triangle_s3(self):
        """einsum('ij,jk,ki->ijk', S, S, S) where S is symmetric.

        Directed triangle with symmetric matrices: S2 on each operand enables
        reflections, promoting C3 → S3 (order 6).
        """
        S = _symmetrize((N, N), PermutationGroup.symmetric(2, axes=(0, 1)))
        assert _detect_order("ij,jk,ki->ijk", S, S, S) == 6


class TestWSideSymmetry:
    """Symmetry on summed (contracted) indices."""

    def test_trace_aa_s2(self):
        """Tr(A·A) = einsum('ij,ji->', A, A): W-side S2{i,j}, order 2."""
        A = we.random.randn(N, N)
        assert _detect_order("ij,ji->", A, A) == 2

    def test_trace_aaa_c3(self):
        """Tr(A·A·A) = einsum('ij,jk,ki->', A, A, A): W-side C3{i,j,k}."""
        A = we.random.randn(N, N)
        assert _detect_order("ij,jk,ki->", A, A, A) == 3

    def test_trace_aaaa_c4(self):
        """Tr(A^4) = einsum('ij,jk,kl,li->', A, A, A, A): W-side C4."""
        A = we.random.randn(N, N)
        assert _detect_order("ij,jk,kl,li->", A, A, A, A) == 4

    def test_partial_trace_trivial(self):
        """einsum('ij,jk,ki->i', A, A, A): i is free, j,k summed.

        C3 rotates i→j→k→i, mapping free label to summed — invalid.
        No V→V and W→W respecting permutation exists.
        """
        A = we.random.randn(N, N)
        assert _detect_order("ij,jk,ki->i", A, A, A) == 1

    @pytest.mark.xfail(
        reason="Frobenius inner product requires coordinated cross-operand "
        "axis relabeling (Source C), which is not yet implemented. "
        "The old fingerprint fast path covered this; the new "
        "generator-based approach does not.",
        strict=True,
    )
    def test_frobenius_inner_product_w_s2(self):
        """einsum('ij,ij->', A, A): Frobenius inner product.

        sum_{i,j} A[i,j]*A[i,j] = sum_{j,i} A[j,i]*A[j,i] — relabeling
        dummy indices i↔j gives the same sum. So W-side S2{i,j} should hold.

        However, swapping operands gives identity π (same subscripts), and
        no per-operand symmetry is declared. Detecting this requires
        coordinated axis relabeling across all operands simultaneously
        (a "Source C" generator), which is not yet implemented.
        """
        A = we.random.randn(N, N)
        assert _detect_order("ij,ij->", A, A) == 2


class TestMixedOperands:
    """Expressions with non-identical operands (no symmetry expected)."""

    def test_aba_chain(self):
        """A·B·A: A appears twice but B breaks the chain."""
        A = we.random.randn(N, N)
        B = we.random.randn(N, N)
        assert _detect_order("ij,jk,kl->il", A, B, A) == 1

    def test_abab_alternating(self):
        """A·B·A·B: two pairs but interleaved — no swap is valid."""
        A = we.random.randn(N, N)
        B = we.random.randn(N, N)
        assert _detect_order("ij,jk,kl,lm->im", A, B, A, B) == 1

    def test_diagonal_extraction(self):
        """einsum('iij->ij', D): repeated index in subscript, no symmetry."""
        D = np.random.randn(N, N, N)
        assert _detect_order("iij->ij", D) == 1
