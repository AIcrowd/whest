"""Tests for free (zero-FLOP) operations."""
import numpy
import pytest
from mechestim._budget import BudgetContext
from mechestim import _free_ops as ops


def test_zeros():
    x = ops.zeros((3, 4))
    assert x.shape == (3, 4)
    assert numpy.all(x == 0)

def test_ones():
    assert numpy.all(ops.ones((2, 3)) == 1)

def test_array():
    x = ops.array([[1, 2], [3, 4]])
    assert x.shape == (2, 2)
    assert x[1, 1] == 4

def test_eye():
    assert numpy.allclose(ops.eye(3), numpy.eye(3))

def test_reshape():
    y = ops.reshape(ops.ones((6,)), (2, 3))
    assert y.shape == (2, 3)

def test_transpose():
    y = ops.transpose(ops.array([[1, 2, 3], [4, 5, 6]]))
    assert y.shape == (3, 2)

def test_concatenate():
    c = ops.concatenate([ops.ones((2, 3)), ops.zeros((2, 3))], axis=0)
    assert c.shape == (4, 3)

def test_where():
    result = ops.where(ops.array([True, False, True]), ops.array([1, 2, 3]), ops.array([4, 5, 6]))
    assert list(result) == [1, 5, 3]

def test_free_ops_dont_cost_flops():
    with BudgetContext(flop_budget=1) as budget:
        ops.zeros((1000, 1000))
        ops.ones((1000,))
        ops.reshape(ops.ones((6,)), (2, 3))
        assert budget.flops_used == 0

def test_free_ops_work_outside_context():
    x = ops.zeros((3,))
    assert x.shape == (3,)

def test_diag():
    assert numpy.allclose(ops.diag(ops.array([1, 2, 3])), numpy.diag([1, 2, 3]))

def test_arange():
    assert list(ops.arange(5)) == [0, 1, 2, 3, 4]

def test_copy():
    x = ops.array([1, 2, 3])
    y = ops.copy(x)
    assert numpy.array_equal(x, y) and x is not y

def test_stack():
    assert ops.stack([ops.ones((3,)), ops.zeros((3,))]).shape == (2, 3)

def test_squeeze():
    assert ops.squeeze(ops.ones((1, 3, 1))).shape == (3,)

def test_expand_dims():
    assert ops.expand_dims(ops.ones((3,)), axis=0).shape == (1, 3)

def test_triu():
    assert numpy.allclose(ops.triu(ops.ones((3, 3))), numpy.triu(numpy.ones((3, 3))))

def test_sort():
    assert list(ops.sort(ops.array([3, 1, 2]))) == [1, 2, 3]
