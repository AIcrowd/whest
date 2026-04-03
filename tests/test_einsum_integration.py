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
        T_data = (T_data + T_data.transpose(1, 0, 2) + T_data.transpose(2, 1, 0) +
                  T_data.transpose(0, 2, 1) + T_data.transpose(1, 2, 0) + T_data.transpose(2, 0, 1)) / 6
        T = as_symmetric(T_data, dims=(0, 1, 2))
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
        assert hasattr(info, 'steps')
        assert hasattr(info, 'optimized_cost')
        assert hasattr(info, 'speedup')

    def test_zero_budget_cost(self):
        A = numpy.ones((10, 10))
        B = numpy.ones((10, 10))
        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            einsum_path("ij,jk->ik", A, B)
            assert budget.flops_used == 0

    def test_symmetric_input_shows_savings(self):
        n = 10
        S = as_symmetric(numpy.eye(n), dims=(0, 1))
        v = numpy.ones(n)
        path, info = einsum_path("ij,j->i", S, v)
        # With a symmetric matrix, there should be some savings
        assert len(info.steps) >= 1
        # Either per-step savings or overall cost reduction
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
            assert hasattr(step, 'subscript')
            assert hasattr(step, 'flop_cost')
            assert hasattr(step, 'dense_flop_cost')
            assert hasattr(step, 'symmetry_savings')
            assert hasattr(step, 'output_symmetry')

    def test_dense_path_has_zero_savings(self):
        A = numpy.ones((5, 5))
        B = numpy.ones((5, 5))
        C = numpy.ones((5, 5))
        _, info = einsum_path("ij,jk,kl->il", A, B, C)
        for step in info.steps:
            assert step.symmetry_savings == 0.0


class TestBackwardCompatibility:
    def test_existing_2_operand_behavior(self):
        A = numpy.ones((3, 4))
        B = numpy.ones((4, 5))
        with BudgetContext(flop_budget=10**6, quiet=True) as budget:
            result = einsum("ij,jk->ik", A, B)
            assert budget.flops_used == 3 * 4 * 5
            assert result.shape == (3, 5)

    def test_symmetric_dims_output_still_works(self):
        X = numpy.ones((5, 10))
        with BudgetContext(flop_budget=10**8, quiet=True):
            result = einsum("ki,kj->ij", X, X, symmetric_dims=[(0, 1)])
            assert isinstance(result, SymmetricTensor)
