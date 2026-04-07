"""Tests for counted pointwise and reduction operations."""

import numpy
import pytest

from mechestim._budget import BudgetContext
from mechestim._pointwise import (
    add,
    argmax,
    clip,
    cumsum,
    dot,
    exp,
    log,
    matmul,
    maximum,
    mean,
    sqrt,
    std,
    sum,
)


def test_exp_result():
    x = numpy.array([0.0, 1.0, 2.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(exp(x), numpy.exp(x))


def test_exp_flop_cost():
    x = numpy.ones((10, 20))
    with BudgetContext(flop_budget=10**6) as budget:
        exp(x)
        assert budget.flops_used == 200


def test_sqrt_result():
    x = numpy.array([1.0, 4.0, 9.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(sqrt(x), numpy.sqrt(x))


def test_add_result():
    a = numpy.array([1.0, 2.0])
    b = numpy.array([3.0, 4.0])
    expected = a + b
    with BudgetContext(flop_budget=10**6) as budget:
        result = add(a, b)
        assert budget.flops_used == 2
    assert numpy.allclose(result, expected)


def test_add_broadcast_cost():
    a = numpy.ones((3, 4))
    b = numpy.ones((4,))
    with BudgetContext(flop_budget=10**6) as budget:
        result = add(a, b)
        assert result.shape == (3, 4)
        assert budget.flops_used == 12


def test_maximum_result():
    a = numpy.array([1.0, 5.0, 3.0])
    b = numpy.array([2.0, 4.0, 6.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(maximum(a, b), numpy.maximum(a, b))


def test_clip_result():
    x = numpy.array([-1.0, 0.5, 2.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(clip(x, 0.0, 1.0), numpy.clip(x, 0.0, 1.0))


def test_sum_full():
    x = numpy.ones((5, 3))
    with BudgetContext(flop_budget=10**6) as budget:
        result = sum(x)
        assert budget.flops_used == 15
    assert float(result) == 15.0


def test_sum_axis():
    x = numpy.ones((5, 3))
    with BudgetContext(flop_budget=10**6) as budget:
        result = sum(x, axis=0)
        assert result.shape == (3,)
        assert budget.flops_used == 15


def test_mean_cost():
    x = numpy.ones((10, 20))
    with BudgetContext(flop_budget=10**6) as budget:
        mean(x, axis=0)
        assert budget.flops_used == 220  # numel(input) + numel(output) = 200 + 20


def test_std_cost():
    x = numpy.ones((10, 20))
    with BudgetContext(flop_budget=10**6) as budget:
        std(x, axis=0)
        assert budget.flops_used == 420  # 2*numel(input) + numel(output) = 400 + 20


def test_argmax_result():
    x = numpy.array([1.0, 5.0, 3.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert argmax(x) == 1


def test_cumsum():
    x = numpy.array([1.0, 2.0, 3.0])
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(cumsum(x), [1, 3, 6])


def test_dot_result():
    a = numpy.ones((3, 4))
    b = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        result = dot(a, b)
        assert numpy.allclose(result, numpy.dot(a, b))
        assert budget.flops_used == 120  # 3*4*5 * op_factor(2)


def test_matmul_result():
    a = numpy.ones((3, 4))
    b = numpy.ones((4, 5))
    with BudgetContext(flop_budget=10**6) as budget:
        assert numpy.allclose(matmul(a, b), numpy.matmul(a, b))


def test_counted_op_outside_context():
    # Operations now auto-activate the global default budget instead of raising
    result = exp(numpy.ones((3,)))
    assert result.shape == (3,)


def test_nan_warning():
    with BudgetContext(flop_budget=10**6):
        with pytest.warns(match="NaN"):
            log(numpy.array([0.0]))
