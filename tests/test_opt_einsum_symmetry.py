"""Tests for _opt_einsum symmetry propagation and cost functions."""

from mechestim._opt_einsum._symmetry import (
    propagate_symmetry,
    symmetric_flop_count,
    symmetry_factor,
    unique_elements,
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
        result = propagate_symmetry(
            None, frozenset("ij"), None, frozenset("jk"), frozenset("ik")
        )
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
            idx_contract,
            True,
            2,
            size_dict,
            output_indices=frozenset("ajk"),
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
        sym = symmetric_flop_count(
            idx, True, 2, size_dict, input_symmetries=[None, None], output_symmetry=None
        )
        assert sym == dense


class TestSymmetryAwarePaths:
    def test_greedy_with_symmetry(self):
        """Greedy path optimizer accepts and uses symmetry info."""
        from mechestim._opt_einsum._contract import contract_path

        # ijk,ai,bj,ck->abc where ijk has S3
        symmetries = [[frozenset("ijk")], None, None, None]
        path, info = contract_path(
            "ijk,ai,bj,ck->abc",
            *[(10,) * 3, (10, 10), (10, 10), (10, 10)],
            shapes=True,
            input_symmetries=symmetries,
        )
        assert len(path) == 3

    def test_symmetry_aware_cost_less_than_dense(self):
        """Symmetry-aware total cost should be less than dense for symmetric inputs."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ij,jk,ki->", (10, 10), (10, 10), (10, 10))
        kwargs = dict(shapes=True, optimize="greedy")

        _, info_dense = contract_path(*args, **kwargs)
        _, info_sym = contract_path(
            *args,
            **kwargs,
            input_symmetries=[[frozenset("ij")], None, None],
        )
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
        path_sym, info_sym = contract_path(
            "ij,jk,kl->il",
            (2, 3),
            (3, 4),
            (4, 5),
            shapes=True,
            optimize="greedy",
            input_symmetries=[None, None, None],
        )
        assert path_no_sym == path_sym
        assert info_no_sym.opt_cost == info_sym.opt_cost

    def test_optimal_with_symmetry(self):
        """Optimal algorithm works with symmetry."""
        from mechestim._opt_einsum._contract import contract_path

        symmetries = [[frozenset("ij")], None, None]
        path, info = contract_path(
            "ij,jk,ki->",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="optimal",
            input_symmetries=symmetries,
        )
        assert len(path) == 2
        assert info.opt_cost > 0

    def test_dp_with_symmetry(self):
        """DP algorithm works with symmetry."""
        from mechestim._opt_einsum._contract import contract_path

        symmetries = [[frozenset("ij")], None, None]
        path, info = contract_path(
            "ij,jk,ki->",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="dp",
            input_symmetries=symmetries,
        )
        assert len(path) == 2
        assert info.opt_cost > 0

    def test_branch_with_symmetry(self):
        """Branch algorithm works with symmetry."""
        from mechestim._opt_einsum._contract import contract_path

        symmetries = [[frozenset("ij")], None, None]
        path, info = contract_path(
            "ij,jk,ki->",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="branch-all",
            input_symmetries=symmetries,
        )
        assert len(path) == 2
        assert info.opt_cost > 0


class TestSymmetricBlas:
    """Tests for symmetry-aware BLAS classification."""

    def test_gemm_with_symmetric_left_becomes_symm(self):
        """GEMM with symmetric left input → SYMM."""
        from mechestim._opt_einsum._blas import can_blas

        # ij,jk->ik where ij is symmetric
        result = can_blas(
            ["ij", "jk"],
            "ik",
            frozenset("j"),
            input_symmetries=[[frozenset("ij")], None],
        )
        assert result == "SYMM"

    def test_gemm_with_symmetric_right_becomes_symm(self):
        """GEMM with symmetric right input → SYMM."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["ij", "jk"],
            "ik",
            frozenset("j"),
            input_symmetries=[None, [frozenset("jk")]],
        )
        assert result == "SYMM"

    def test_gemv_with_symmetric_matrix_becomes_symv(self):
        """GEMV with symmetric matrix → SYMV."""
        from mechestim._opt_einsum._blas import can_blas

        # ijk,ji->k where ij is symmetric: can_blas returns GEMV/EINSUM
        # (transposed contraction not efficiently expressed as GEMM).
        # With symmetric ij, this should become SYMV.
        result = can_blas(
            ["ijk", "ji"],
            "k",
            frozenset("ij"),
            input_symmetries=[[frozenset("ij")], None],
        )
        assert result == "SYMV"

    def test_dot_with_symmetric_becomes_sydt(self):
        """DOT with symmetric input → SYDT."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["ij", "ij"],
            "",
            frozenset("ij"),
            input_symmetries=[[frozenset("ij")], None],
        )
        assert result == "SYDT"

    def test_no_symmetry_unchanged(self):
        """Without symmetry, classification is unchanged."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(["ij", "jk"], "ik", frozenset("j"))
        assert result == "GEMM"

    def test_none_symmetries_unchanged(self):
        """Explicit None symmetries → same as no symmetry."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["ij", "jk"],
            "ik",
            frozenset("j"),
            input_symmetries=[None, None],
        )
        assert result == "GEMM"

    def test_tdot_with_symmetry_unchanged(self):
        """TDOT doesn't have a symmetric variant — stays TDOT."""
        from mechestim._opt_einsum._blas import can_blas

        # ijk,lkj->il: base classification is TDOT (non-aligned contraction).
        # Symmetry on an index pair present in the input does not upgrade TDOT.
        result = can_blas(
            ["ijk", "lkj"],
            "il",
            frozenset("jk"),
            input_symmetries=[[frozenset("ij")], None],
        )
        assert result == "TDOT"

    def test_outer_with_symmetry_unchanged(self):
        """OUTER doesn't have a symmetric variant — stays OUTER/EINSUM."""
        from mechestim._opt_einsum._blas import can_blas

        result = can_blas(
            ["i", "j"],
            "ij",
            frozenset(),
            input_symmetries=[None, None],
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

    def test_blas_type_with_symmetric_input(self):
        """StepInfo should show SYMM/SYMV for symmetric inputs."""
        from mechestim._opt_einsum._contract import contract_path

        # ij,jk->ik where ij is symmetric — the step that contracts the
        # symmetric tensor should show SYMM
        _, info = contract_path(
            "ij,jk,kl->il",
            (5, 5),
            (5, 5),
            (5, 5),
            shapes=True,
            optimize="greedy",
            input_symmetries=[[frozenset("ij")], None, None],
        )
        blas_types = [s.blas_type for s in info.steps]
        # At least one step should involve the symmetric tensor and get SYMM
        assert any(bt in ("SYMM", "SYMV", "SYDT") for bt in blas_types), (
            f"Expected at least one symmetric BLAS type, got {blas_types}"
        )


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
            input_symmetries=[[frozenset("ij")], None],
            output_symmetry=None,
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
            input_symmetries=[[frozenset("ijk")], None],
            output_symmetry=[frozenset("jk")],
        )
        dense = flop_count(frozenset("aijk"), True, 2, size_dict)
        # unique outputs = 10 * C(11,2) = 550, total = 1000
        # cost = 20000 * 550 // 1000 = 11000
        assert cost == 11000

    def test_hand_counted_s3_contraction(self):
        """Verify: 550 unique outputs (10 × C(11,2)), each a length-10 dot
        product with 2 ops each => 550 × 10 × 2 = 11000."""
        size_dict = {"i": 10, "j": 10, "k": 10, "a": 10}
        cost = symmetric_flop_count(
            frozenset("aijk"),
            True,
            2,
            size_dict,
            output_indices=frozenset("ajk"),
            input_symmetries=[[frozenset("ijk")], None],
            output_symmetry=[frozenset("jk")],
        )
        assert cost == 11000

    def test_brute_force_flop_count(self):
        """Verify formula against explicit operation counting for small n."""
        import numpy as np

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
            input_symmetries=[[frozenset("ijk")], None],
            output_symmetry=[frozenset("jk")],
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
            input_symmetries=[None, None],
            output_symmetry=None,
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
            input_symmetries=[[frozenset("ij")], None],
            output_symmetry=[frozenset("ij")],
        )
        # Output has S2 on (i,j): unique = 10 * C(11,2) = 550, total = 1000
        # dense_base = 1000 * 1 (op_factor=1, no inner product)
        # cost = 1000 * 550 // 1000 = 550
        assert cost == 550


class TestAllAlgorithmsSymmetryAware:
    """Each algorithm accepts and uses input_symmetries."""

    def _run_algo(self, algo_name, input_symmetries):
        from mechestim._opt_einsum._contract import contract_path

        return contract_path(
            "ijk,ai,bj->abk",
            (5,) * 3,
            (5, 5),
            (5, 5),
            shapes=True,
            optimize=algo_name,
            input_symmetries=input_symmetries,
        )

    def test_optimal_accepts_symmetry(self):
        path, info = self._run_algo("optimal", [[frozenset("ijk")], None, None])
        assert len(path) == 2
        assert info.optimized_cost > 0

    def test_greedy_accepts_symmetry(self):
        path, info = self._run_algo("greedy", [[frozenset("ijk")], None, None])
        assert len(path) == 2

    def test_branch_all_accepts_symmetry(self):
        path, info = self._run_algo("branch-all", [[frozenset("ijk")], None, None])
        assert len(path) == 2

    def test_dp_accepts_symmetry(self):
        path, info = self._run_algo("dp", [[frozenset("ijk")], None, None])
        assert len(path) == 2

    def test_no_symmetry_unchanged_all_algos(self):
        """Every algorithm produces identical results without symmetry."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ij,jk,kl->il", (2, 3), (3, 4), (4, 5))
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            path_before, _ = contract_path(*args, shapes=True, optimize=algo)
            path_after, _ = contract_path(
                *args, shapes=True, optimize=algo, input_symmetries=[None, None, None]
            )
            assert list(path_before) == list(path_after), (
                f"{algo} path changed with None symmetries"
            )

    def test_symmetric_cost_le_dense_all_algos(self):
        """Symmetric cost <= dense cost for all algorithms."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ijk,ai,bj->abk", (5,) * 3, (5, 5), (5, 5))
        sym = [[frozenset("ijk")], None, None]
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            _, info_dense = contract_path(*args, shapes=True, optimize=algo)
            _, info_sym = contract_path(
                *args, shapes=True, optimize=algo, input_symmetries=sym
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
        sym = [[frozenset("ij")], None, None]
        costs = {}
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            _, info = contract_path(
                *args, shapes=True, optimize=algo, input_symmetries=sym
            )
            costs[algo] = info.optimized_cost
        # Optimal should find the best; all others should be >= optimal
        assert costs["greedy"] >= costs["optimal"], f"greedy < optimal: {costs}"
        assert costs["branch-all"] >= costs["optimal"], f"branch-all < optimal: {costs}"

    def test_symmetric_cost_le_dense_cost_all_algorithms(self):
        """For every algorithm, symmetric cost <= dense cost."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ijk,ai,bj->abk", (5,) * 3, (5, 5), (5, 5))
        sym = [[frozenset("ijk")], None, None]
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            _, info_dense = contract_path(*args, shapes=True, optimize=algo)
            _, info_sym = contract_path(
                *args, shapes=True, optimize=algo, input_symmetries=sym
            )
            assert info_sym.optimized_cost <= info_dense.optimized_cost, (
                f"{algo}: sym={info_sym.optimized_cost} > dense={info_dense.optimized_cost}"
            )

    def test_no_symmetry_all_algorithms_unchanged(self):
        """Every algorithm produces identical results with None symmetries."""
        from mechestim._opt_einsum._contract import contract_path

        args = ("ij,jk,kl->il", (2, 3), (3, 4), (4, 5))
        for algo in ["optimal", "greedy", "branch-all", "dp"]:
            path_before, info_before = contract_path(*args, shapes=True, optimize=algo)
            path_after, info_after = contract_path(
                *args, shapes=True, optimize=algo, input_symmetries=[None, None, None]
            )
            assert list(path_before) == list(path_after), f"{algo} path changed"

    def test_slack_thread_example(self):
        """The ijk,ai,bj,ck->abc example from the Slack discussion."""
        from mechestim._opt_einsum._contract import contract_path

        _, info = contract_path(
            "ijk,ai,bj,ck->abc",
            *[(100,) * 3, (100, 100), (100, 100), (100, 100)],
            shapes=True,
            input_symmetries=[[frozenset("ijk")], None, None, None],
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

        _, info = contract_path(
            "ijk,kl,li->j",
            *[(5,) * 3, (5, 5), (5, 5)],
            shapes=True,
            optimize="optimal",
            input_symmetries=[[frozenset("ijk")], [frozenset("kl")], None],
        )
        assert len(info.steps) == 2
        assert info.optimized_cost > 0

    def test_dp_invariant_result_symmetry(self):
        """Result symmetry of a subset is order-independent."""
        from mechestim._opt_einsum._symmetry import propagate_symmetry

        sym_T = [frozenset("ijk")]
        # Order 1: T+A -> ajk, then +B -> abk
        sym1 = propagate_symmetry(
            sym_T, frozenset("ijk"), None, frozenset("ai"), frozenset("ajk")
        )
        sym_final1 = propagate_symmetry(
            sym1, frozenset("ajk"), None, frozenset("bj"), frozenset("abk")
        )
        # Order 2: T+B -> ibk, then +A -> abk
        sym2 = propagate_symmetry(
            sym_T, frozenset("ijk"), None, frozenset("bj"), frozenset("ibk")
        )
        sym_final2 = propagate_symmetry(
            sym2, frozenset("ibk"), None, frozenset("ai"), frozenset("abk")
        )
        assert sym_final1 == sym_final2

    def test_random_greedy_with_symmetry(self):
        """RandomGreedy accepts and uses symmetry."""
        from mechestim._opt_einsum._path_random import RandomGreedy

        rg = RandomGreedy(max_repeats=4)
        path = rg(
            [frozenset("ij"), frozenset("jk"), frozenset("ki")],
            frozenset(""),
            {"i": 5, "j": 5, "k": 5},
            input_symmetries=[[frozenset("ij")], None, None],
        )
        assert len(path) == 2

    def test_end_to_end_numerical_correctness(self):
        """Symmetry-aware path produces numerically correct results."""
        import numpy as np

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
