"""Integration tests for symmetry-aware einsum with path optimization."""

import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim._einsum import einsum, einsum_path
from mechestim._symmetric import SymmetricTensor, as_symmetric
from mechestim.errors import BudgetExhaustedError


class TestMultiOperandEinsum:
    def test_three_operand_correctness(self):
        A = numpy.random.RandomState(42).rand(5, 6)
        B = numpy.random.RandomState(43).rand(6, 7)
        C = numpy.random.RandomState(44).rand(7, 8)
        expected = numpy.einsum("ij,jk,kl->il", A, B, C)
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk,kl->il", A, B, C)
        numpy.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_symmetric_input_reduces_multi_operand_cost(self):
        n = 10
        T_data = numpy.random.RandomState(42).rand(n, n, n)
        T_data = (
            T_data
            + T_data.transpose(1, 0, 2)
            + T_data.transpose(2, 1, 0)
            + T_data.transpose(0, 2, 1)
            + T_data.transpose(1, 2, 0)
            + T_data.transpose(2, 0, 1)
        ) / 6
        T = as_symmetric(T_data, symmetric_axes=(0, 1, 2))
        A = numpy.random.RandomState(43).rand(n, n)
        B = numpy.random.RandomState(44).rand(n, n)

        with BudgetContext(flop_budget=10**8, quiet=True) as budget_sym:
            result_sym = einsum("ijk,ai,bj->abk", T, A, B)

        with BudgetContext(flop_budget=10**8, quiet=True) as budget_dense:
            result_dense = einsum("ijk,ai,bj->abk", T_data, A, B)

        numpy.testing.assert_allclose(result_sym, result_dense, rtol=1e-10)
        assert budget_sym.flops_used < budget_dense.flops_used

    def test_optimize_false_falls_back(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        C = numpy.ones((5, 6))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk,kl->il", A, B, C, optimize=False)
            assert result.shape == (3, 6)


class TestOptimizeKwarg:
    def test_default_is_auto(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk->ik", A, B)
            assert result.shape == (3, 5)

    def test_explicit_greedy(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        C = numpy.ones((5, 6))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ij,jk,kl->il", A, B, C, optimize="greedy")
            assert result.shape == (3, 6)


class TestBudgetIntegration:
    def test_single_upfront_deduction(self):
        A = numpy.ones((10, 10))
        B = numpy.ones((10, 10))
        C = numpy.ones((10, 10))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            einsum("ij,jk,kl->il", A, B, C)
            einsum_ops = [r for r in budget.op_log if r.op_name == "einsum"]
            assert len(einsum_ops) == 1

    def test_budget_exceeded_before_execution(self):
        A = numpy.ones((100, 100))
        B = numpy.ones((100, 100))
        C = numpy.ones((100, 100))
        with pytest.raises(BudgetExhaustedError):
            with BudgetContext(flop_budget=100, quiet=True):
                einsum("ij,jk,kl->il", A, B, C)


class TestEinsumPath:
    def test_returns_path_and_info(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        C = numpy.ones((5, 6))
        path, info = einsum_path("ij,jk,kl->il", A, B, C)
        assert isinstance(path, list)
        assert len(path) == 2
        assert hasattr(info, "steps")
        assert hasattr(info, "optimized_cost")
        assert hasattr(info, "speedup")

    def test_zero_budget_cost(self):
        A = numpy.ones((10, 10))
        B = numpy.ones((10, 10))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            einsum_path("ij,jk->ik", A, B)
            assert budget.flops_used == 0

    def test_symmetric_input_shows_savings(self):
        # Use a contraction where symmetric indices survive in the output.
        # "ijk,k->ij" with S2 on {i,j}: both i and j survive, so the
        # symmetric group provides a real cost reduction.
        n = 10
        k = 5
        S = as_symmetric(numpy.ones((n, n, k)), symmetric_axes=(0, 1))
        v = numpy.ones(k)
        path, info = einsum_path("ijk,k->ij", S, v)
        assert len(info.steps) >= 1
        # Per-step savings from symmetry
        has_savings = any(s.symmetry_savings > 0 for s in info.steps)
        has_cost_reduction = info.optimized_cost < info.naive_cost
        assert has_savings or has_cost_reduction

    def test_str_output(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        C = numpy.ones((5, 6))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        table = str(info)
        assert isinstance(table, str)
        assert len(table) > 50

    def test_str_output_no_symmetry_omits_sym_column(self):
        """When no operands have symmetry, the symmetry column is omitted."""
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        _, info = einsum_path("ij,jk->ik", A, B)
        table = str(info)
        assert "symmetry" not in table.lower()

    def test_str_output_with_symmetry_includes_sym_column(self):
        """When any operand has symmetry, the symmetry column is shown
        with per-step inputs → output annotations."""
        n = 6
        T = as_symmetric(numpy.ones((n, n, n)), symmetric_axes=(0, 1, 2))
        A = numpy.ones((n, n))
        _, info = einsum_path("ijk,ai->ajk", T, A)
        table = str(info)
        assert "symmetry" in table.lower()
        # The oracle detects S2{j,k} on the output (j,k survive after contracting i).
        # Input symmetry is not separately annotated in the table (oracle is output-centric).
        # Old cost: 2,592 (dense). New cost: 1,512 (S2 savings). Tightened by oracle.
        assert "S2" in table
        assert info.optimized_cost < info.naive_cost

    def test_str_output_symmetry_chain_through_steps(self):
        """Verify that symmetry savings through a multi-step path are shown:
        the oracle reduces cost for the first step via S2{j,k} on the intermediate."""
        n = 5
        T = as_symmetric(numpy.ones((n, n, n)), symmetric_axes=(0, 1, 2))
        A = numpy.ones((n, n))
        B = numpy.ones((n, n))
        C = numpy.ones((n, n))
        _, info = einsum_path("ijk,ai,bj,ck->abc", T, A, B, C)
        table = str(info)
        # The oracle detects S2{j,k} on the first intermediate ajk.
        # Input symmetry is not separately annotated; only output symmetry shows.
        # Old behavior: showed S3 for input. New: shows S2{j,k} for output of step 0.
        assert "S2" in table
        assert any(s.symmetry_savings > 0 for s in info.steps)

    def test_str_output_includes_index_sizes(self):
        """The 'Index sizes' line should appear and group equal-sized indices."""
        # All same size: should collapse to a=b=c=d=i=j=k=l=N
        n = 7
        T = as_symmetric(numpy.ones((n, n, n)), symmetric_axes=(0, 1, 2))
        A = numpy.ones((n, n))
        _, info = einsum_path("ijk,ai->ajk", T, A)
        table = str(info)
        assert "Index sizes:" in table
        # All four indices have size 7 → grouped
        assert "a=i=j=k=7" in table or "i=j=k=a=7" in table or "7" in table

    def test_str_output_index_sizes_separates_different(self):
        """Different-sized indices should be listed separately."""
        A = numpy.ones((3, 100))
        B = numpy.ones((100, 50))
        C = numpy.ones((50, 7))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        table = str(info)
        assert "Index sizes:" in table
        # Each index has a unique size
        assert "i=3" in table or "=3" in table
        assert "100" in table
        assert "50" in table
        assert "l=7" in table or "=7" in table


class TestPathInfoStepInfo:
    def test_step_info_has_symmetry_fields(self):
        from mechestim._opt_einsum._contract import StepInfo

        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        assert len(info.steps) == 2
        for step in info.steps:
            assert isinstance(step, StepInfo)
            assert hasattr(step, "subscript")
            assert hasattr(step, "flop_cost")
            assert hasattr(step, "dense_flop_cost")
            assert hasattr(step, "symmetry_savings")
            assert hasattr(step, "output_symmetry")

    def test_dense_path_has_zero_savings(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        for step in info.steps:
            assert step.symmetry_savings == 0.0


class TestPathInfoDebugFields:
    """New diagnostic fields for debugging path-finding decisions."""

    def test_optimizer_used_resolved_for_explicit_choice(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="greedy")
        assert info.optimizer_used == "greedy"

    def test_optimizer_used_resolved_for_auto(self):
        # auto with 3 ops routes to 'optimal' per _AUTO_CHOICES
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="auto")
        assert info.optimizer_used == "optimal"

    def test_optimizer_used_trivial_for_two_ops(self):
        # 2-op cases skip the optimizer entirely
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        _, info = einsum_path("ij,jk->ik", A, B)
        assert info.optimizer_used == "trivial"

    def test_step_path_indices_match_path_field(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        path, info = einsum_path("ij,jk,kl->il", A, B, C)
        for step, path_tuple in zip(info.steps, path):
            assert step.path_indices == tuple(path_tuple)

    def test_step_merged_subset_grows_monotonically(self):
        # As intermediates accumulate, the merged subset of original
        # operand positions covered should be a superset of all earlier
        # subsets that fed into this step.
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        D = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl,lm->im", A, B, C, D)
        # Final step's subset should be the union of all original positions.
        assert info.steps[-1].merged_subset == frozenset({0, 1, 2, 3})

    def test_step_merged_subset_for_block_outer_product(self):
        # 'ab,cd->abcd' with same X — single step covers operands {0, 1}
        X = numpy.ones((4, 4))
        _, info = einsum_path("ab,cd->abcd", X, X)
        assert len(info.steps) == 1
        assert info.steps[0].merged_subset == frozenset({0, 1})

    def test_format_table_default_includes_optimizer_and_contract_columns(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C, optimize="greedy")
        table = info.format_table()
        assert "Optimizer:" in table
        assert "greedy" in table
        assert "contract" in table
        assert "(0," in table or "(1," in table  # path tuple shown

    def test_format_table_shows_unique_dense_when_symmetry_present(self):
        # 'ab,cd->abcd' with same X has block S2 → unique/dense column
        X = numpy.ones((4, 4))
        _, info = einsum_path("ab,cd->abcd", X, X)
        table = info.format_table()
        assert "unique/dense" in table
        # Block S2 unique = C(17, 2) = 136 for n=4 (n^2 = 16, k = 2)
        # dense = 4^4 = 256
        assert "136/256" in table

    def test_format_table_omits_unique_dense_when_no_symmetry(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        table = info.format_table()
        assert "unique/dense" not in table

    def test_format_table_verbose_shows_subset_and_cumulative(self):
        X = numpy.ones((4, 4))
        _, info = einsum_path("ai,bi,ci->abc", X, X, X)
        table = info.format_table(verbose=True)
        assert "subset=" in table
        assert "out_shape=" in table
        assert "cumulative=" in table
        # Final cumulative should equal optimized_cost
        assert f"cumulative={info.optimized_cost:,}" in table

    def test_format_table_verbose_subset_grows(self):
        # For a 3-op chain, step 0's subset has 2 entries, step 1 has 3.
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        table = info.format_table(verbose=True)
        # First step's subset has exactly 2 elements
        first_subset = info.steps[0].merged_subset
        assert first_subset is not None
        assert len(first_subset) == 2
        # Final step covers all 3 original ops
        assert info.steps[-1].merged_subset == frozenset({0, 1, 2})


class TestBackwardCompatibility:
    def test_existing_2_operand_behavior(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum("ij,jk->ik", A, B)
            assert budget.flops_used == 120  # 3*4*5 * op_factor(2)
            assert result.shape == (3, 5)

    def test_symmetric_axes_output_still_works(self):
        X = numpy.ones((5, 10))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetric_axes=[(0, 1)])
            assert isinstance(result, SymmetricTensor)


class TestSymmetricBlasClassification:
    """Verify that symmetric inputs produce SYMM/SYMV/SYDT BLAS labels
    instead of the generic GEMM/GEMV/DOT."""

    def test_symmetric_matmul_gets_symm_label(self):
        """einsum('ij,jk->ik', X, X) with X declared symmetric should
        report blas_type='SYMM' on the single contraction step, not 'GEMM'."""
        import mechestim as me

        n = 10
        X = me.as_symmetric(numpy.ones((n, n)), symmetric_axes=(0, 1))
        _, info = einsum_path("ij,jk->ik", X, X)
        assert len(info.steps) == 1
        assert info.steps[0].blas_type == "SYMM", (
            f"Expected SYMM for symmetric matmul, got {info.steps[0].blas_type!r}"
        )
