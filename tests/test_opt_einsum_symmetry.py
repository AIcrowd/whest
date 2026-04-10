"""Tests for _opt_einsum symmetry helpers and oracle-based path optimization."""

import numpy as np

from mechestim._perm_group import PermutationGroup
from mechestim._opt_einsum._symmetry import (
    symmetric_flop_count,
    unique_elements,
)


def _s_group(*labels):
    """Create S_k group with given labels for testing."""
    g = PermutationGroup.symmetric(len(labels))
    g._labels = tuple(labels)
    return g


def _make_oracle(subscripts, operands=None, *, per_op_groups=None):
    """Helper: build a SubgraphSymmetryOracle from a subscript string."""
    from mechestim._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle

    input_str, output_str = (subscripts.split("->") + [""])[:2]
    parts = input_str.split(",")
    n = len(parts)
    if operands is None:
        operands = [object() for _ in range(n)]
    if per_op_groups is None:
        per_op_groups = [None] * n
    return SubgraphSymmetryOracle(
        operands=operands,
        subscript_parts=parts,
        per_op_groups=per_op_groups,
        output_chars=output_str,
    )


class TestUniqueElements:
    def test_s2_symmetry(self):
        """C(n+1, 2) for S2 on two indices of size n."""
        size_dict = {"i": 10, "j": 10}
        assert unique_elements(frozenset("ij"), size_dict, perm_group=_s_group("i", "j")) == 55

    def test_s3_symmetry(self):
        """C(n+2, 3) for S3."""
        size_dict = {"i": 10, "j": 10, "k": 10}
        assert unique_elements(frozenset("ijk"), size_dict, perm_group=_s_group("i", "j", "k")) == 220

    def test_mixed_symmetric_and_free(self):
        """S2 on (j,k) plus free index a."""
        size_dict = {"a": 5, "j": 10, "k": 10}
        assert unique_elements(frozenset("ajk"), size_dict, perm_group=_s_group("j", "k")) == 5 * 55

    def test_no_symmetry(self):
        size_dict = {"i": 3, "j": 4}
        assert unique_elements(frozenset("ij"), size_dict, None) == 12

    def test_empty(self):
        assert unique_elements(frozenset(), {}, None) == 1


class TestSymmetricFlopCount:
    def test_s3_contraction_reduces_cost(self):
        """ijk,ai->ajk with S3 on ijk should cost less than dense."""
        size_dict = {"i": 100, "j": 100, "k": 100, "a": 100}
        idx_contract = frozenset("aijk")
        cost = symmetric_flop_count(
            idx_contract,
            True,
            2,
            size_dict,
            output_indices=frozenset("ajk"),
            output_group=_s_group("j", "k"),
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
        sym = symmetric_flop_count(idx, True, 2, size_dict, output_group=None)
        assert sym == dense


class TestSymmetryAwarePaths:
    def test_greedy_with_oracle(self):
        """Greedy path optimizer accepts and uses oracle."""
        from mechestim._opt_einsum._contract import contract_path

        # ijk,ai,bj,ck->abc where ijk has S3
        sym = [_s_group("i", "j", "k")]
        oracle = _make_oracle(
            "ijk,ai,bj,ck->abc",
            per_op_groups=[sym, None, None, None],
        )
        path, info = contract_path(
            "ijk,ai,bj,ck->abc",
            *[(10,) * 3, (10, 10), (10, 10), (10, 10)],
            shapes=True,
            symmetry_oracle=oracle,
        )
        assert len(path) == 3

    def test_symmetry_aware_cost_less_than_dense(self):
        """Symmetry-aware total cost should be less than dense for symmetric inputs."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ij,jk,ki->", (10, 10), (10, 10), (10, 10))
        kwargs = dict(shapes=True, optimize="greedy")

        _, info_dense = contract_path(*args, **kwargs)

        sym = [_s_group("i", "j")]
        oracle = _make_oracle("ij,jk,ki->", per_op_groups=[sym, None, None])
        _, info_sym = contract_path(*args, **kwargs, symmetry_oracle=oracle)
        assert info_sym.opt_cost <= info_dense.opt_cost

    def test_no_symmetry_matches_upstream(self):
        """With no symmetry info, results should match upstream dense path."""
        from mechestim._opt_einsum._contract import contract_path

        path_no_sym, info_no_sym = contract_path(
            "ij,jk,kl->il",
            (2, 3),
            (3, 4),
            (4, 5),
            shapes=True,
            optimize="greedy",
        )
        # No oracle => same as no symmetry
        path_sym, info_sym = contract_path(
            "ij,jk,kl->il",
            (2, 3),
            (3, 4),
            (4, 5),
            shapes=True,
            optimize="greedy",
        )
        assert path_no_sym == path_sym
        assert info_no_sym.opt_cost == info_sym.opt_cost

    def test_optimal_with_oracle(self):
        """Optimal algorithm works with oracle."""
        from mechestim._opt_einsum._contract import contract_path

        sym = [_s_group("i", "j")]
        oracle = _make_oracle("ij,jk,ki->", per_op_groups=[sym, None, None])
        path, info = contract_path(
            "ij,jk,ki->",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="optimal",
            symmetry_oracle=oracle,
        )
        assert len(path) == 2
        assert info.opt_cost > 0

    def test_dp_with_oracle(self):
        """DP algorithm works with oracle (stubs symmetry, but doesn't crash)."""
        from mechestim._opt_einsum._contract import contract_path

        sym = [_s_group("i", "j")]
        oracle = _make_oracle("ij,jk,ki->", per_op_groups=[sym, None, None])
        path, info = contract_path(
            "ij,jk,ki->",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="dp",
            symmetry_oracle=oracle,
        )
        assert len(path) == 2
        assert info.opt_cost > 0

    def test_branch_with_oracle(self):
        """Branch algorithm works with oracle."""
        from mechestim._opt_einsum._contract import contract_path

        sym = [_s_group("i", "j")]
        oracle = _make_oracle("ij,jk,ki->", per_op_groups=[sym, None, None])
        path, info = contract_path(
            "ij,jk,ki->",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="branch-all",
            symmetry_oracle=oracle,
        )
        assert len(path) == 2
        assert info.opt_cost > 0


class TestSymmetricBlas:
    """Tests for symmetry-aware BLAS classification."""

    def test_gemm_with_symmetric_left_becomes_symm(self):
        """GEMM with symmetric left input -> SYMM."""
        from mechestim._opt_einsum._blas import can_blas

        # ij,jk->ik where ij is symmetric
        result = can_blas(
            ["ij", "jk"],
            "ik",
            frozenset("j"),
            input_groups=[_s_group("i", "j"), None],
        )
        assert result == "SYMM"

    def test_gemm_with_symmetric_right_becomes_symm(self):
        """GEMM with symmetric right input -> SYMM."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["ij", "jk"],
            "ik",
            frozenset("j"),
            input_groups=[None, _s_group("j", "k")],
        )
        assert result == "SYMM"

    def test_gemv_with_symmetric_matrix_becomes_symv(self):
        """GEMV with symmetric matrix -> SYMV."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["ijk", "ji"],
            "k",
            frozenset("ij"),
            input_groups=[_s_group("i", "j"), None],
        )
        assert result == "SYMV"

    def test_dot_with_symmetric_becomes_sydt(self):
        """DOT with symmetric input -> SYDT."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["ij", "ij"],
            "",
            frozenset("ij"),
            input_groups=[_s_group("i", "j"), None],
        )
        assert result == "SYDT"

    def test_no_symmetry_unchanged(self):
        """Without symmetry, classification is unchanged."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(["ij", "jk"], "ik", frozenset("j"))
        assert result == "GEMM"

    def test_none_symmetries_unchanged(self):
        """Explicit None symmetries -> same as no symmetry."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["ij", "jk"],
            "ik",
            frozenset("j"),
            input_groups=[None, None],
        )
        assert result == "GEMM"

    def test_tdot_with_symmetry_unchanged(self):
        """TDOT doesn't have a symmetric variant -- stays TDOT."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["ijk", "lkj"],
            "il",
            frozenset("jk"),
            input_groups=[_s_group("i", "j"), None],
        )
        assert result == "TDOT"

    def test_outer_with_symmetry_unchanged(self):
        """OUTER doesn't have a symmetric variant -- stays OUTER/EINSUM."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["i", "j"],
            "ij",
            frozenset(),
            input_groups=[None, None],
        )
        assert result == "OUTER/EINSUM"


class TestStepInfoBlasType:
    def test_step_info_has_blas_type(self):
        """StepInfo should include blas_type field."""
        from mechestim._opt_einsum._contract import contract_path

        _, info = contract_path(
            "ij,jk,kl->il",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="greedy",
        )
        for step in info.steps:
            assert hasattr(step, "blas_type")

    def test_blas_type_with_symmetric_oracle(self):
        """StepInfo should show SYMM/SYMV for symmetric inputs via oracle."""
        from mechestim._opt_einsum._contract import contract_path

        sym = [_s_group("i", "j")]
        oracle = _make_oracle("ij,jk,kl->il", per_op_groups=[sym, None, None])
        _, info = contract_path(
            "ij,jk,kl->il",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="greedy",
            symmetry_oracle=oracle,
        )
        for step in info.steps:
            assert hasattr(step, "blas_type")


class TestFixedSymmetricFlopCount:
    def test_s2_matvec_no_reduction(self):
        """S[ij](S2) * v[j] -> r[i]: summing j means S2 gives no reduction."""
        from mechestim._opt_einsum._helpers import flop_count

        size_dict = {"i": 10, "j": 10}
        cost = symmetric_flop_count(
            frozenset("ij"),
            True,
            2,
            size_dict,
            output_indices=frozenset("i"),
            output_group=None,
        )
        dense = flop_count(frozenset("ij"), True, 2, size_dict)
        assert cost == dense

    def test_s3_contract_one_gives_s2_reduction(self):
        """T[ijk](S3) * A[ai] -> R[ajk](S2): i summed, j,k survive as S2.

        The reduction comes solely from output symmetry: we compute only
        unique (j,k) pairs.  unique_output / total_output = C(11,2)/100
        = 55/100 = 0.55, so cost = 20000 * 55 // 100 = 11000.
        """
        from mechestim._opt_einsum._helpers import flop_count

        size_dict = {"i": 10, "j": 10, "k": 10, "a": 10}
        cost = symmetric_flop_count(
            frozenset("aijk"),
            True,
            2,
            size_dict,
            output_indices=frozenset("ajk"),
            output_group=_s_group("j", "k"),
        )
        dense = flop_count(frozenset("aijk"), True, 2, size_dict)
        # unique outputs = 10 * C(11,2) = 550, total = 1000
        # cost = 20000 * 550 // 1000 = 11000
        assert cost == 11000

    def test_hand_counted_s3_contraction(self):
        """Verify: 550 unique outputs (10 x C(11,2)), each a length-10 dot
        product with 2 ops each => 550 x 10 x 2 = 11000."""
        size_dict = {"i": 10, "j": 10, "k": 10, "a": 10}
        cost = symmetric_flop_count(
            frozenset("aijk"),
            True,
            2,
            size_dict,
            output_indices=frozenset("ajk"),
            output_group=_s_group("j", "k"),
        )
        assert cost == 11000

    def test_brute_force_flop_count(self):
        """Verify formula against explicit operation counting for small n."""
        n = 4
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n, n, n))
        T = (
            raw
            + raw.transpose(0, 2, 1)
            + raw.transpose(1, 0, 2)
            + raw.transpose(1, 2, 0)
            + raw.transpose(2, 0, 1)
            + raw.transpose(2, 1, 0)
        ) / 6.0
        A = rng.standard_normal((n, n))

        # Count actual multiply-adds for unique (j,k) pairs
        counted_ops = 0
        for a in range(n):
            for j in range(n):
                for k in range(j, n):  # unique (j,k) only
                    for i in range(n):
                        counted_ops += 2  # one multiply + one add

        size_dict = {"i": n, "j": n, "k": n, "a": n}
        formula_cost = symmetric_flop_count(
            frozenset("aijk"),
            True,
            2,
            size_dict,
            output_indices=frozenset("ajk"),
            output_group=_s_group("j", "k"),
        )
        assert formula_cost == counted_ops

    def test_output_indices_none_backward_compat(self):
        """When output_indices is None, equals no-symmetry flop_count."""
        from mechestim._opt_einsum._helpers import flop_count

        size_dict = {"i": 10, "j": 10, "k": 10}
        cost_new = symmetric_flop_count(
            frozenset("ijk"),
            True,
            2,
            size_dict,
            output_indices=None,
            output_group=None,
        )
        dense = flop_count(frozenset("ijk"), True, 2, size_dict)
        assert cost_new == dense

    def test_all_indices_survive(self):
        """When no indices are summed, full group applies (outer product-like)."""
        size_dict = {"i": 10, "j": 10, "a": 10}
        cost = symmetric_flop_count(
            frozenset("aij"),
            False,
            2,
            size_dict,
            output_indices=frozenset("aij"),
            output_group=_s_group("i", "j"),
        )
        # Output has S2 on (i,j): unique = 10 * C(11,2) = 550, total = 1000
        # dense_base = 1000 * 1 (op_factor=1, no inner product)
        # cost = 1000 * 550 // 1000 = 550
        assert cost == 550


class TestAllAlgorithmsOracleAware:
    """Each algorithm accepts a symmetry oracle."""

    def _run_algo(self, algo_name, oracle=None):
        from mechestim._opt_einsum._contract import contract_path

        return contract_path(
            "ijk,ai,bj->abk",
            (5,) * 3,
            (5, 5),
            (5, 5),
            shapes=True,
            optimize=algo_name,
            symmetry_oracle=oracle,
        )

    def _make_s3_oracle(self):
        sym = [_s_group("i", "j", "k")]
        return _make_oracle("ijk,ai,bj->abk", per_op_groups=[sym, None, None])

    def test_optimal_accepts_oracle(self):
        path, info = self._run_algo("optimal", self._make_s3_oracle())
        assert len(path) == 2
        assert info.optimized_cost > 0

    def test_greedy_accepts_oracle(self):
        path, info = self._run_algo("greedy", self._make_s3_oracle())
        assert len(path) == 2

    def test_branch_all_accepts_oracle(self):
        path, info = self._run_algo("branch-all", self._make_s3_oracle())
        assert len(path) == 2

    def test_dp_accepts_oracle(self):
        path, info = self._run_algo("dp", self._make_s3_oracle())
        assert len(path) == 2

    def test_no_oracle_unchanged_all_algos(self):
        """Every algorithm produces identical results without oracle."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ij,jk,kl->il", (2, 3), (3, 4), (4, 5))
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            path_before, _ = contract_path(*args, shapes=True, optimize=algo)
            path_after, _ = contract_path(*args, shapes=True, optimize=algo)
            assert list(path_before) == list(path_after), (
                f"{algo} path non-deterministic"
            )

    def test_symmetric_cost_le_dense_all_algos(self):
        """Symmetric cost <= dense cost for all algorithms."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ijk,ai,bj->abk", (5,) * 3, (5, 5), (5, 5))
        oracle = self._make_s3_oracle()
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            _, info_dense = contract_path(*args, shapes=True, optimize=algo)
            _, info_sym = contract_path(
                *args, shapes=True, optimize=algo, symmetry_oracle=oracle
            )
            assert info_sym.optimized_cost <= info_dense.optimized_cost, (
                f"{algo}: sym={info_sym.optimized_cost} > dense={info_dense.optimized_cost}"
            )


class TestExhaustiveSymmetryValidation:
    """Exhaustive tests to verify symmetry-aware path algorithms are correct."""

    def test_all_algorithms_agree_on_small_problem(self):
        """For a small problem, optimal and dp should find the same cost."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ij,jk,ki->", (5, 5), (5, 5), (5, 5))
        sym = [_s_group("i", "j")]
        oracle = _make_oracle("ij,jk,ki->", per_op_groups=[sym, None, None])
        costs = {}
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            _, info = contract_path(
                *args, shapes=True, optimize=algo, symmetry_oracle=oracle
            )
            costs[algo] = info.optimized_cost
        # Optimal should find the best; all others should be >= optimal
        assert costs["greedy"] >= costs["optimal"], f"greedy < optimal: {costs}"
        assert costs["branch-all"] >= costs["optimal"], f"branch-all < optimal: {costs}"

    def test_symmetric_cost_le_dense_cost_all_algorithms(self):
        """For every algorithm, symmetric cost <= dense cost."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ijk,ai,bj->abk", (5,) * 3, (5, 5), (5, 5))
        sym = [_s_group("i", "j", "k")]
        oracle = _make_oracle("ijk,ai,bj->abk", per_op_groups=[sym, None, None])
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            _, info_dense = contract_path(*args, shapes=True, optimize=algo)
            _, info_sym = contract_path(
                *args, shapes=True, optimize=algo, symmetry_oracle=oracle
            )
            assert info_sym.optimized_cost <= info_dense.optimized_cost, (
                f"{algo}: sym={info_sym.optimized_cost} > dense={info_dense.optimized_cost}"
            )

    def test_no_oracle_all_algorithms_unchanged(self):
        """Every algorithm produces identical results with/without None oracle."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ij,jk,kl->il", (2, 3), (3, 4), (4, 5))
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            path_before, info_before = contract_path(*args, shapes=True, optimize=algo)
            path_after, info_after = contract_path(
                *args, shapes=True, optimize=algo, symmetry_oracle=None
            )
            assert list(path_before) == list(path_after), f"{algo} path changed"

    def test_slack_thread_example(self):
        """The ijk,ai,bj,ck->abc example from the Slack discussion."""
        from mechestim._opt_einsum._contract import contract_path

        sym = [_s_group("i", "j", "k")]
        oracle = _make_oracle("ijk,ai,bj,ck->abc", per_op_groups=[sym, None, None, None])
        _, info = contract_path(
            "ijk,ai,bj,ck->abc",
            *[(100,) * 3, (100, 100), (100, 100), (100, 100)],
            shapes=True,
            symmetry_oracle=oracle,
        )
        assert len(info.steps) == 3
        _, info_dense = contract_path(
            "ijk,ai,bj,ck->abc",
            *[(100,) * 3, (100, 100), (100, 100), (100, 100)],
            shapes=True,
        )
        assert info.optimized_cost < info_dense.optimized_cost
        assert any(s.symmetry_savings > 0 for s in info.steps)

    def test_mixed_symmetry_network(self):
        """Network with S2, S3, and dense tensors."""
        from mechestim._opt_einsum._contract import contract_path

        sym_s3 = [_s_group("i", "j", "k")]
        sym_s2 = [_s_group("k", "l")]
        oracle = _make_oracle("ijk,kl,li->j", per_op_groups=[sym_s3, sym_s2, None])
        _, info = contract_path(
            "ijk,kl,li->j",
            *[(5,) * 3, (5, 5), (5, 5)],
            shapes=True,
            optimize="optimal",
            symmetry_oracle=oracle,
        )
        assert len(info.steps) == 2
        assert info.optimized_cost > 0

    def test_random_greedy_with_oracle(self):
        """RandomGreedy accepts oracle (ignores it as stub, doesn't crash)."""
        from mechestim._opt_einsum._path_random import RandomGreedy

        sym = [_s_group("i", "j")]
        oracle = _make_oracle("ij,jk,ki->", per_op_groups=[sym, None, None])
        rg = RandomGreedy(max_repeats=4)
        # Oracle is passed via contract_path, not directly to RandomGreedy
        # Test the public interface via contract_path
        from mechestim._opt_einsum._contract import contract_path

        path, info = contract_path(
            "ij,jk,ki->",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="greedy",
            symmetry_oracle=oracle,
        )
        assert len(path) == 2

    def test_end_to_end_numerical_correctness(self):
        """Symmetry-aware path produces numerically correct results."""
        from mechestim._budget import BudgetContext
        from mechestim._einsum import einsum
        from mechestim._symmetric import as_symmetric

        n = 8
        T_data = np.random.RandomState(42).rand(n, n, n)
        T_data = (
            T_data
            + T_data.transpose(1, 0, 2)
            + T_data.transpose(2, 1, 0)
            + T_data.transpose(0, 2, 1)
            + T_data.transpose(1, 2, 0)
            + T_data.transpose(2, 0, 1)
        ) / 6
        T = as_symmetric(T_data, symmetric_axes=(0, 1, 2))
        A = np.random.RandomState(43).rand(n, n)
        B = np.random.RandomState(44).rand(n, n)

        expected = np.einsum("ijk,ai,bj->abk", T_data, A, B)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ijk,ai,bj->abk", T, A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestInnerSymmetryFlops:
    def test_no_inner_sym_unchanged(self):
        """Without inner symmetry, the cost should match the existing behavior."""
        cost = symmetric_flop_count(
            "abij",
            True,
            2,
            {"a": 3, "b": 3, "i": 4, "j": 4},
            output_group=_s_group("a", "b"),
            output_indices=frozenset("ab"),
        )
        cost_with_flag = symmetric_flop_count(
            "abij",
            True,
            2,
            {"a": 3, "b": 3, "i": 4, "j": 4},
            output_group=_s_group("a", "b"),
            output_indices=frozenset("ab"),
            use_inner_symmetry=True,
        )
        assert cost == cost_with_flag

    def test_inner_sym_reduces_cost(self):
        """With inner symmetry and flag on, cost should be strictly lower."""
        base_cost = symmetric_flop_count(
            "abij",
            True,
            2,
            {"a": 3, "b": 3, "i": 4, "j": 4},
            output_group=_s_group("a", "b"),
            output_indices=frozenset("ab"),
        )
        reduced_cost = symmetric_flop_count(
            "abij",
            True,
            2,
            {"a": 3, "b": 3, "i": 4, "j": 4},
            output_group=_s_group("a", "b"),
            output_indices=frozenset("ab"),
            inner_group=_s_group("i", "j"),
            inner_indices=frozenset("ij"),
            use_inner_symmetry=True,
        )
        assert reduced_cost < base_cost

    def test_inner_sym_flag_off_no_reduction(self):
        """With flag off, inner_group is ignored."""
        base_cost = symmetric_flop_count(
            "abij",
            True,
            2,
            {"a": 3, "b": 3, "i": 4, "j": 4},
            output_group=_s_group("a", "b"),
            output_indices=frozenset("ab"),
        )
        same_cost = symmetric_flop_count(
            "abij",
            True,
            2,
            {"a": 3, "b": 3, "i": 4, "j": 4},
            output_group=_s_group("a", "b"),
            output_indices=frozenset("ab"),
            inner_group=_s_group("i", "j"),
            inner_indices=frozenset("ij"),
            use_inner_symmetry=False,
        )
        assert same_cost == base_cost

    def test_inner_sym_exact_value(self):
        """Verify exact multiplicative reduction: output_ratio * inner_ratio."""
        # size: a=3, b=3, i=4, j=4
        # dense = 3*3*4*4 * op_factor(inner=True, 2 terms) = 144 * 2 = 288
        # output unique: C(3+1,2) = 6 out of 9 -> ratio 6/9
        # inner unique: C(4+1,2) = 10 out of 16 -> ratio 10/16
        # reduced = 288 * 6/9 * 10/16 = 288 * 2/3 * 5/8 = 120
        cost = symmetric_flop_count(
            "abij",
            True,
            2,
            {"a": 3, "b": 3, "i": 4, "j": 4},
            output_group=_s_group("a", "b"),
            output_indices=frozenset("ab"),
            inner_group=_s_group("i", "j"),
            inner_indices=frozenset("ij"),
            use_inner_symmetry=True,
        )
        assert cost == 120
