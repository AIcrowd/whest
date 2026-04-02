"""Tests for linalg solver wrappers with FLOP counting."""

import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestSolve:
    def test_result_matches_numpy(self):
        A = numpy.array([[3.0, 1.0], [1.0, 2.0]])
        b = numpy.array([9.0, 8.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import solve

            result = solve(A, b)
            assert numpy.allclose(result, numpy.linalg.solve(A, b))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n) + numpy.eye(n) * 10
        b = numpy.random.randn(n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import solve

            solve(A, b)
            assert budget.flops_used == 2 * n**3 // 3 + n**2 * 1

    def test_op_log(self):
        A = numpy.eye(3)
        b = numpy.ones(3)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import solve

            solve(A, b)
            assert budget.op_log[-1].op_name == "linalg.solve"

    def test_outside_context_raises(self):
        from mechestim.linalg import solve

        with pytest.raises(NoBudgetContextError):
            solve(numpy.eye(3), numpy.ones(3))


class TestInv:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import inv

            result = inv(A)
            assert numpy.allclose(result, numpy.linalg.inv(A))

    def test_cost(self):
        n = 5
        A = numpy.random.randn(n, n) + numpy.eye(n) * 10
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import inv

            inv(A)
            assert budget.flops_used == n**3


class TestLstsq:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(6, 4)
        b = numpy.random.randn(6)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import lstsq

            x, residuals, rank, sv = lstsq(A, b, rcond=None)
            x_np, _, _, _ = numpy.linalg.lstsq(A, b, rcond=None)
            assert numpy.allclose(x, x_np)

    def test_cost(self):
        m, n = 6, 4
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import lstsq

            lstsq(A, b, rcond=None)
            assert budget.flops_used == m * n * min(m, n)


class TestPinv:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(4, 3)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import pinv

            result = pinv(A)
            assert numpy.allclose(result, numpy.linalg.pinv(A))

    def test_cost(self):
        m, n = 4, 3
        A = numpy.random.randn(m, n)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import pinv

            pinv(A)
            assert budget.flops_used == m * n * min(m, n)


class TestTensorsolve:
    def test_runs_without_error(self):
        # a must satisfy prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])
        a = numpy.eye(6).reshape(2, 3, 2, 3)
        b = numpy.ones((2, 3))
        with BudgetContext(flop_budget=10**9):
            from mechestim.linalg import tensorsolve

            tensorsolve(a, b)


class TestTensorinv:
    def test_runs_without_error(self):
        a = numpy.eye(24).reshape(6, 4, 2, 3, 4)
        with BudgetContext(flop_budget=10**9):
            from mechestim.linalg import tensorinv

            tensorinv(a, ind=2)
