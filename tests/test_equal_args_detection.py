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
            [frozenset({("i",), ("j",)})],  # S2{i,j} on op0
            [frozenset({("k",), ("l",)})],  # S2{k,l} on op1 (not strictly needed here)
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


class TestEnumerateCandidates:
    """Candidate enumerators for per-index and block sigmas."""

    def test_per_index_candidates_two_outputs(self):
        # Output 'ij' → one candidate: swap i↔j
        candidates = _enumerate_per_index_candidates("ij")
        assert len(candidates) == 1
        assert candidates[0] == {"i": "j", "j": "i"}

    def test_per_index_candidates_three_outputs(self):
        # Output 'ijk' → three candidates: swap any pair
        candidates = _enumerate_per_index_candidates("ijk")
        assert len(candidates) == 3
        expected = [
            {"i": "j", "j": "i"},
            {"i": "k", "k": "i"},
            {"j": "k", "k": "j"},
        ]

        # Sort dicts as sets of items for comparison
        def key(d):
            return tuple(sorted(d.items()))

        assert sorted(candidates, key=key) == sorted(expected, key=key)

    def test_per_index_candidates_single_output(self):
        # Output 'i' → no pairs → no candidates
        assert _enumerate_per_index_candidates("i") == []

    def test_per_index_candidates_empty_output(self):
        assert _enumerate_per_index_candidates("") == []

    def test_block_candidates_gram_matrix(self):
        # einsum('ij,ik->jk', X, X) — pair (op0, op1), free_0={j}, free_1={k}
        X = np.ones((3, 3))
        output_chars = frozenset("jk")
        subscript_parts = ["ij", "ik"]
        operands = [X, X]
        candidates = _enumerate_block_candidates(
            operands, subscript_parts, output_chars
        )
        # One candidate: block swap (j,)↔(k,) which is really a per-index swap
        # since the block size is 1. But _enumerate_block_candidates returns
        # ALL pairwise block candidates regardless of size.
        assert len(candidates) == 1
        # The sigma should map j↔k
        sigma = candidates[0]
        assert sigma == {"j": "k", "k": "j"}

    def test_block_candidates_outer_product(self):
        # einsum('ijk,ilm->jklm', X, X) — pair (op0, op1)
        X = np.ones((3, 3, 3))
        output_chars = frozenset("jklm")
        subscript_parts = ["ijk", "ilm"]
        operands = [X, X]
        candidates = _enumerate_block_candidates(
            operands, subscript_parts, output_chars
        )
        # One candidate: block swap (j,k) ↔ (l,m) via positional pairing
        assert len(candidates) == 1
        sigma = candidates[0]
        # j↔l, k↔m (positions 1, 2 of the operand subscripts)
        assert sigma == {"j": "l", "l": "j", "k": "m", "m": "k"}

    def test_block_candidates_no_equal_operands(self):
        X = np.ones((3, 3))
        Y = np.ones((3, 3))
        output_chars = frozenset("jk")
        subscript_parts = ["ij", "ik"]
        operands = [X, Y]
        candidates = _enumerate_block_candidates(
            operands, subscript_parts, output_chars
        )
        # No equal pairs → no block candidates
        assert candidates == []


class TestDetectInducedOutputSymmetry:
    """Top-level detection integrating candidate enumeration + validity check."""

    def test_gram_matrix(self):
        X = np.ones((3, 3))
        # einsum('ij,ik->jk', X, X)
        result = _detect_induced_output_symmetry(
            operands=[X, X],
            subscript_parts=["ij", "ik"],
            output_chars="jk",
            per_op_syms=[None, None],
        )
        assert result == [frozenset({("j",), ("k",)})]

    def test_matmul_chain_plain_X(self):
        X = np.ones((3, 3))
        # einsum('ij,jk->ik', X, X) — plain X, no declared sym
        # Per-index swap i↔k: op0 'ij'→'kj' (sorted 'jk'=op1), op1 'jk'→'ji' (='ij'=op0)
        # Op0↔op1, both X. Valid.
        result = _detect_induced_output_symmetry(
            operands=[X, X],
            subscript_parts=["ij", "jk"],
            output_chars="ik",
            per_op_syms=[None, None],
        )
        assert result == [frozenset({("i",), ("k",)})]

    def test_different_operands_no_induction(self):
        X = np.ones((3, 3))
        Y = np.ones((3, 3))
        result = _detect_induced_output_symmetry(
            operands=[X, Y],
            subscript_parts=["ij", "jk"],
            output_chars="ik",
            per_op_syms=[None, None],
        )
        assert result is None

    def test_triple_product(self):
        X = np.ones((3, 3))
        # einsum('ij,ik,il->jkl', X, X, X)
        result = _detect_induced_output_symmetry(
            operands=[X, X, X],
            subscript_parts=["ij", "ik", "il"],
            output_chars="jkl",
            per_op_syms=[None, None, None],
        )
        # Per-index candidates: j↔k, j↔l, k↔l — all valid
        # After merge: S3{j,k,l}
        assert result is not None
        assert len(result) == 1
        assert result[0] == frozenset({("j",), ("k",), ("l",)})

    def test_single_operand_no_induction(self):
        X = np.ones((3, 3, 3))
        result = _detect_induced_output_symmetry(
            operands=[X],
            subscript_parts=["ijk"],
            output_chars="ikj",
            per_op_syms=[None],
        )
        # Only one operand → no pairs → no induction
        assert result is None

    def test_empty_output(self):
        X = np.ones((3, 3))
        result = _detect_induced_output_symmetry(
            operands=[X, X],
            subscript_parts=["ij", "ij"],
            output_chars="",
            per_op_syms=[None, None],
        )
        # Empty output → no induction possible
        assert result is None

    def test_block_outer_product(self):
        X = np.ones((3, 3, 3))
        # einsum('ijk,ilm->jklm', X, X)
        result = _detect_induced_output_symmetry(
            operands=[X, X],
            subscript_parts=["ijk", "ilm"],
            output_chars="jklm",
            per_op_syms=[None, None],
        )
        # Block candidate: (j,k)↔(l,m). Valid. Induce block S2.
        assert result == [frozenset({("j", "k"), ("l", "m")})]
