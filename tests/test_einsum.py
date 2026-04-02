"""Tests for mechestim einsum with FLOP counting and symmetry."""

import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim._einsum import einsum
from mechestim.errors import BudgetExhaustedError, NoBudgetContextError, SymmetryError


def test_matmul_result():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        C = einsum("ij,jk->ik", A, B)
        assert numpy.allclose(C, numpy.einsum("ij,jk->ik", A, B))


def test_matmul_flop_cost():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum("ij,jk->ik", A, B)
        assert budget.flops_used == 3 * 4 * 5


def test_trace():
    A = numpy.eye(10)
    with BudgetContext(flop_budget=10**6) as budget:
        result = einsum("ii->", A)
        assert result == 10.0
        assert budget.flops_used == 10


def test_outer_product():
    a = numpy.ones((3,))
    b = numpy.ones((4,))
    with BudgetContext(flop_budget=10**6) as budget:
        result = einsum("i,j->ij", a, b)
        assert result.shape == (3, 4)
        assert budget.flops_used == 12


def test_batch_matmul():
    A = numpy.ones((2, 3, 4))
    B = numpy.ones((2, 4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum("bij,bjk->bik", A, B)
        assert budget.flops_used == 2 * 3 * 4 * 5


def test_repeated_operand_symmetry():
    x = numpy.ones((10, 256))
    A = numpy.ones((10, 10))
    with BudgetContext(flop_budget=10**8) as budget:
        einsum("ai,bi,ab->", x, x, A)
        assert budget.flops_used == (10 * 10 * 256) // 2


def test_no_symmetry_different_objects():
    x1 = numpy.ones((10, 256))
    x2 = numpy.ones((10, 256))  # different object
    A = numpy.ones((10, 10))
    with BudgetContext(flop_budget=10**8) as budget:
        einsum("ai,bi,ab->", x1, x2, A)
        assert budget.flops_used == 10 * 10 * 256


def test_symmetric_dims_valid():
    x = numpy.ones((3, 10))
    y = numpy.ones((3, 10))
    A = numpy.eye(3)
    with BudgetContext(flop_budget=10**8) as budget:
        result = einsum("ai,bj,ab->ij", x, y, A, symmetric_dims=[(0, 1)])
        assert budget.flops_used == (3 * 3 * 10 * 10) // 2


def test_symmetric_dims_invalid():
    x = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    y = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    with BudgetContext(flop_budget=10**8):
        with pytest.raises(SymmetryError):
            einsum("ij,jk->ik", x, y, symmetric_dims=[(0, 1)])


def test_outside_context():
    with pytest.raises(NoBudgetContextError):
        einsum("ij,jk->ik", numpy.ones((3, 4)), numpy.ones((4, 5)))


def test_budget_exceeded():
    A = numpy.ones((256, 256))
    with pytest.raises(BudgetExhaustedError):
        with BudgetContext(flop_budget=100):
            einsum("ij,jk->ik", A, A)


def test_op_log_records_subscripts():
    A = numpy.ones((3, 4))
    B = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        einsum("ij,jk->ik", A, B)
        assert budget.op_log[0].subscripts == "ij,jk->ik"
        assert budget.op_log[0].op_name == "einsum"
