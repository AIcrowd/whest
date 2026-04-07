"""Unit tests for equal-operand detection in _einsum.py.

Tests:
- _is_valid_symmetry: the core validity check with self-mapping guard
- _enumerate_per_index_candidates / _enumerate_block_candidates: candidate sigmas
- _detect_induced_output_symmetry: top-level detection
"""

# ruff: noqa: F401
import numpy as np

from mechestim._einsum import (
    _detect_induced_output_symmetry,
    _enumerate_block_candidates,
    _enumerate_per_index_candidates,
    _is_valid_symmetry,
)


class TestIsValidSymmetry:
    """_is_valid_symmetry — the core validity check with self-mapping guard."""

    def test_gram_matrix_swap_is_valid(self):
        # einsum('ij,ik->jk', X, X) — swap j↔k
        X = np.ones((3, 3))
        sigma = {"j": "k", "k": "j"}
        subscript_parts = ["ij", "ik"]
        operands = [X, X]  # same object
        per_op_syms = [None, None]
        assert _is_valid_symmetry(sigma, subscript_parts, operands, per_op_syms)

    def test_matmul_chain_swap_is_valid_both_same(self):
        # einsum('ij,jk->ik', X, X) — swap i↔k
        X = np.ones((3, 3))
        sigma = {"i": "k", "k": "i"}
        subscript_parts = ["ij", "jk"]
        operands = [X, X]
        per_op_syms = [None, None]
        assert _is_valid_symmetry(sigma, subscript_parts, operands, per_op_syms)

    def test_matmul_chain_different_objects_invalid(self):
        # einsum('ij,jk->ik', X, Y) — swap i↔k is NOT a symmetry
        X = np.ones((3, 3))
        Y = np.ones((3, 3))  # different object, same values
        sigma = {"i": "k", "k": "i"}
        subscript_parts = ["ij", "jk"]
        operands = [X, Y]
        per_op_syms = [None, None]
        assert not _is_valid_symmetry(sigma, subscript_parts, operands, per_op_syms)

    def test_self_map_without_per_op_sym_is_invalid(self):
        # einsum('ij,kl->ijkl', X, X) — swap i↔j requires X to be symmetric
        X = np.ones((3, 3))
        sigma = {"i": "j", "j": "i"}
        subscript_parts = ["ij", "kl"]
        operands = [X, X]
        per_op_syms = [None, None]  # X is not declared symmetric
        # Op 0's 'ij' → 'ji' (same set), would self-map with non-trivial relabel
        assert not _is_valid_symmetry(sigma, subscript_parts, operands, per_op_syms)

    def test_self_map_with_per_op_sym_is_valid(self):
        # Same as above but X is declared symmetric
        X = np.ones((3, 3))
        sigma = {"i": "j", "j": "i"}
        subscript_parts = ["ij", "kl"]
        operands = [X, X]
        per_op_syms = [
            [frozenset({('i',), ('j',)})],  # S2{i,j} on op0
            [frozenset({('k',), ('l',)})],  # S2{k,l} on op1 (not strictly needed here)
        ]
        assert _is_valid_symmetry(sigma, subscript_parts, operands, per_op_syms)

    def test_triple_product_one_swap(self):
        # einsum('ij,ik,il->jkl', X, X, X) — swap j↔k
        X = np.ones((3, 3))
        sigma = {"j": "k", "k": "j"}
        subscript_parts = ["ij", "ik", "il"]
        operands = [X, X, X]
        per_op_syms = [None, None, None]
        # op0 'ij' → 'ik' (matches op1), op1 'ik' → 'ij' (matches op0),
        # op2 'il' unchanged → matches itself. All three X. Valid.
        assert _is_valid_symmetry(sigma, subscript_parts, operands, per_op_syms)
