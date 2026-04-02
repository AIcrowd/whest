# tests/test_linalg_aliases.py
"""Tests for linalg namespace aliases that delegate to top-level mechestim ops."""

import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim.errors import NoBudgetContextError


class TestLinalgMatmul:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(4, 3)
        B = numpy.random.randn(3, 5)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import matmul

            result = matmul(A, B)
            assert numpy.allclose(result, numpy.matmul(A, B))

    def test_cost_deducted(self):
        A = numpy.random.randn(4, 3)
        B = numpy.random.randn(3, 5)
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import matmul

            matmul(A, B)
            assert budget.flops_used > 0

    def test_outside_context_raises(self):
        from mechestim.linalg import matmul

        with pytest.raises(NoBudgetContextError):
            matmul(numpy.ones((2, 2)), numpy.ones((2, 2)))


class TestLinalgCross:
    def test_result_matches_numpy(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import cross

            result = cross(a, b)
            assert numpy.allclose(result, numpy.cross(a, b))


class TestLinalgOuter:
    def test_result_matches_numpy(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import outer

            result = outer(a, b)
            assert numpy.allclose(result, numpy.outer(a, b))


class TestLinalgTensordot:
    def test_result_matches_numpy(self):
        A = numpy.random.randn(3, 4)
        B = numpy.random.randn(4, 5)
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import tensordot

            result = tensordot(A, B, axes=1)
            assert numpy.allclose(result, numpy.tensordot(A, B, axes=1))


class TestLinalgVecdot:
    def test_result_matches_numpy(self):
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([4.0, 5.0, 6.0])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import vecdot

            result = vecdot(a, b)
            assert numpy.allclose(result, numpy.vecdot(a, b))


class TestLinalgDiagonal:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import diagonal

            result = diagonal(A)
            assert numpy.allclose(result, numpy.diagonal(A))

    def test_zero_cost(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import diagonal

            diagonal(A)
            assert budget.flops_used == 0


class TestLinalgMatrixTranspose:
    def test_result_matches_numpy(self):
        A = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with BudgetContext(flop_budget=10**6):
            from mechestim.linalg import matrix_transpose

            result = matrix_transpose(A)
            assert numpy.allclose(result, numpy.matrix_transpose(A))

    def test_zero_cost(self):
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        with BudgetContext(flop_budget=10**6) as budget:
            from mechestim.linalg import matrix_transpose

            matrix_transpose(A)
            assert budget.flops_used == 0
