"""Tests for _opt_einsum symmetry propagation and cost functions."""

import pytest
from mechestim._opt_einsum._symmetry import (
    IndexSymmetry,
    propagate_symmetry,
    unique_elements,
    symmetry_factor,
    symmetric_flop_count,
    compute_unique_size,
)


class TestPropagateSymmetry:
    def test_s3_contracts_one_index_to_s2(self):
        """ijk (S3) contracted with ai -> ajk has S2 on (j,k)."""
        sym1 = [frozenset("ijk")]
        sym2 = None
        k1 = frozenset("ijk")
        k2 = frozenset("ai")
        k12 = frozenset("ajk")
        result = propagate_symmetry(sym1, k1, sym2, k2, k12)
        assert result == [frozenset("jk")]

    def test_s3_contracts_two_indices_to_none(self):
        """ijk (S3) contracted with ij -> k has no symmetry."""
        sym1 = [frozenset("ijk")]
        sym2 = None
        k1 = frozenset("ijk")
        k2 = frozenset("ij")
        k12 = frozenset("k")
        result = propagate_symmetry(sym1, k1, sym2, k2, k12)
        assert result is None or result == []

    def test_partial_symmetry_preserved(self):
        """ijkl (S2 on ij, S2 on kl) contracted with ak -> aijl has S2 on (i,j) only."""
        sym1 = [frozenset("ij"), frozenset("kl")]
        sym2 = None
        k1 = frozenset("ijkl")
        k2 = frozenset("ak")
        k12 = frozenset("aijl")
        result = propagate_symmetry(sym1, k1, sym2, k2, k12)
        assert frozenset("ij") in result
        assert frozenset("kl") not in result
        assert len(result) == 1

    def test_no_symmetry_inputs(self):
        """Two tensors with no symmetry -> no symmetry."""
        result = propagate_symmetry(None, frozenset("ij"), None, frozenset("jk"), frozenset("ik"))
        assert result is None or result == []

    def test_both_inputs_symmetric(self):
        """Both inputs have symmetry, both propagate."""
        sym1 = [frozenset("ij")]
        sym2 = [frozenset("kl")]
        k1 = frozenset("ijm")
        k2 = frozenset("mkl")
        k12 = frozenset("ijkl")
        result = propagate_symmetry(sym1, k1, sym2, k2, k12)
        assert frozenset("ij") in result
        assert frozenset("kl") in result
        assert len(result) == 2


class TestUniqueElements:
    def test_s2_symmetry(self):
        """C(n+1, 2) for S2 on two indices of size n."""
        size_dict = {"i": 10, "j": 10}
        sym = [frozenset("ij")]
        assert unique_elements(frozenset("ij"), size_dict, sym) == 55

    def test_s3_symmetry(self):
        """C(n+2, 3) for S3."""
        size_dict = {"i": 10, "j": 10, "k": 10}
        sym = [frozenset("ijk")]
        assert unique_elements(frozenset("ijk"), size_dict, sym) == 220

    def test_mixed_symmetric_and_free(self):
        """S2 on (j,k) plus free index a."""
        size_dict = {"a": 5, "j": 10, "k": 10}
        sym = [frozenset("jk")]
        assert unique_elements(frozenset("ajk"), size_dict, sym) == 5 * 55

    def test_no_symmetry(self):
        size_dict = {"i": 3, "j": 4}
        assert unique_elements(frozenset("ij"), size_dict, None) == 12

    def test_empty(self):
        assert unique_elements(frozenset(), {}, None) == 1


class TestSymmetryFactor:
    def test_s2(self):
        assert symmetry_factor([frozenset("ij")]) == 2

    def test_s3(self):
        assert symmetry_factor([frozenset("ijk")]) == 6

    def test_multiple_groups(self):
        assert symmetry_factor([frozenset("ij"), frozenset("kl")]) == 4

    def test_none(self):
        assert symmetry_factor(None) == 1


class TestSymmetricFlopCount:
    def test_s3_contraction_reduces_cost(self):
        """ijk,ai->ajk with S3 on ijk should cost less than dense."""
        size_dict = {"i": 100, "j": 100, "k": 100, "a": 100}
        sym1 = [frozenset("ijk")]
        sym2 = None
        idx_contract = frozenset("aijk")
        cost = symmetric_flop_count(
            idx_contract, True, 2, size_dict,
            input_symmetries=[sym1, sym2],
            output_symmetry=[frozenset("jk")],
        )
        dense_cost = 100**4 * 2
        assert cost < dense_cost
        assert cost > 0

    def test_no_symmetry_matches_dense(self):
        """Without symmetry, symmetric_flop_count should equal flop_count."""
        from mechestim._opt_einsum._helpers import flop_count
        size_dict = {"i": 10, "j": 10, "k": 10}
        idx = frozenset("ijk")
        dense = flop_count(idx, True, 2, size_dict)
        sym = symmetric_flop_count(idx, True, 2, size_dict, input_symmetries=[None, None], output_symmetry=None)
        assert sym == dense


class TestSymmetryAwarePaths:
    def test_greedy_with_symmetry(self):
        """Greedy path optimizer accepts and uses symmetry info."""
        from mechestim._opt_einsum._contract import contract_path

        # ijk,ai,bj,ck->abc where ijk has S3
        symmetries = [[frozenset("ijk")], None, None, None]
        path, info = contract_path(
            "ijk,ai,bj,ck->abc",
            *[(10,)*3, (10,10), (10,10), (10,10)],
            shapes=True,
            input_symmetries=symmetries,
        )
        assert len(path) == 3

    def test_symmetry_aware_cost_less_than_dense(self):
        """Symmetry-aware total cost should be less than dense for symmetric inputs."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ij,jk,ki->", (10,10), (10,10), (10,10))
        kwargs = dict(shapes=True, optimize="greedy")

        _, info_dense = contract_path(*args, **kwargs)
        _, info_sym = contract_path(
            *args, **kwargs,
            input_symmetries=[[frozenset("ij")], None, None],
        )
        assert info_sym.opt_cost <= info_dense.opt_cost

    def test_no_symmetry_matches_upstream(self):
        """With no symmetry info, results should match upstream dense path."""
        from mechestim._opt_einsum._contract import contract_path

        path_no_sym, info_no_sym = contract_path(
            "ij,jk,kl->il", (2,3), (3,4), (4,5),
            shapes=True, optimize="greedy",
        )
        path_sym, info_sym = contract_path(
            "ij,jk,kl->il", (2,3), (3,4), (4,5),
            shapes=True, optimize="greedy",
            input_symmetries=[None, None, None],
        )
        assert path_no_sym == path_sym
        assert info_no_sym.opt_cost == info_sym.opt_cost

    def test_optimal_with_symmetry(self):
        """Optimal algorithm works with symmetry."""
        from mechestim._opt_einsum._contract import contract_path
        symmetries = [[frozenset("ij")], None, None]
        path, info = contract_path(
            "ij,jk,ki->", (5,5), (5,5), (5,5),
            shapes=True, optimize="optimal",
            input_symmetries=symmetries,
        )
        assert len(path) == 2
        assert info.opt_cost > 0

    def test_dp_with_symmetry(self):
        """DP algorithm works with symmetry."""
        from mechestim._opt_einsum._contract import contract_path
        symmetries = [[frozenset("ij")], None, None]
        path, info = contract_path(
            "ij,jk,ki->", (5,5), (5,5), (5,5),
            shapes=True, optimize="dp",
            input_symmetries=symmetries,
        )
        assert len(path) == 2
        assert info.opt_cost > 0

    def test_branch_with_symmetry(self):
        """Branch algorithm works with symmetry."""
        from mechestim._opt_einsum._contract import contract_path
        symmetries = [[frozenset("ij")], None, None]
        path, info = contract_path(
            "ij,jk,ki->", (5,5), (5,5), (5,5),
            shapes=True, optimize="branch-all",
            input_symmetries=symmetries,
        )
        assert len(path) == 2
        assert info.opt_cost > 0
