"""Tests for FLOP cost calculators."""
import pytest
from mechestim._flops import einsum_cost, parse_einsum_subscripts, pointwise_cost, reduction_cost, svd_cost


def test_parse_matmul():
    inputs, output = parse_einsum_subscripts("ij,jk->ik")
    assert inputs == [['i', 'j'], ['j', 'k']]
    assert output == ['i', 'k']

def test_parse_trace():
    inputs, output = parse_einsum_subscripts("ii->")
    assert inputs == [['i', 'i']]
    assert output == []

def test_parse_implicit():
    inputs, output = parse_einsum_subscripts("ij,jk")
    assert inputs == [['i', 'j'], ['j', 'k']]
    assert output == ['i', 'k']

def test_einsum_cost_matmul():
    assert einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)]) == 3 * 4 * 5

def test_einsum_cost_trace():
    assert einsum_cost("ii->", shapes=[(10, 10)]) == 10

def test_einsum_cost_batch_matmul():
    assert einsum_cost("bij,bjk->bik", shapes=[(2, 3, 4), (2, 4, 5)]) == 2 * 3 * 4 * 5

def test_einsum_cost_outer_product():
    assert einsum_cost("i,j->ij", shapes=[(3,), (4,)]) == 3 * 4

def test_einsum_cost_scalar_output():
    assert einsum_cost("i,i->", shapes=[(5,), (5,)]) == 5

def test_einsum_cost_symmetry_two_repeats():
    cost = einsum_cost("ai,bi,ab->", shapes=[(10, 256), (10, 256), (10, 10)], repeated_operand_indices=[0, 1])
    assert cost == (10 * 10 * 256) // 2

def test_einsum_cost_symmetry_three_repeats():
    cost = einsum_cost("ai,bj,ck,abc->ijk", shapes=[(2, 10), (2, 10), (2, 10), (2, 2, 2)], repeated_operand_indices=[0, 1, 2])
    assert cost == (2 * 2 * 2 * 10 * 10 * 10) // 6

def test_pointwise_cost():
    assert pointwise_cost(shape=(256, 256)) == 256 * 256

def test_pointwise_cost_scalar():
    assert pointwise_cost(shape=()) == 1

def test_reduction_cost():
    assert reduction_cost(input_shape=(256, 256), axis=None) == 256 * 256

def test_svd_cost():
    assert svd_cost(m=100, n=50, k=10) == 100 * 50 * 10

def test_svd_cost_full():
    assert svd_cost(m=100, n=50, k=None) == 100 * 50 * 50


from mechestim._symmetric import SymmetryInfo

def test_pointwise_cost_symmetric():
    info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
    assert pointwise_cost(shape=(5, 5), symmetry_info=info) == 15

def test_pointwise_cost_partial_symmetry():
    info = SymmetryInfo(symmetric_dims=[(0, 1), (2, 3)], shape=(4, 4, 3, 3))
    assert pointwise_cost(shape=(4, 4, 3, 3), symmetry_info=info) == 60

def test_pointwise_cost_no_symmetry_unchanged():
    assert pointwise_cost(shape=(5, 5)) == 25

def test_reduction_cost_symmetric():
    info = SymmetryInfo(symmetric_dims=[(0, 1)], shape=(5, 5))
    assert reduction_cost(input_shape=(5, 5), axis=None, symmetry_info=info) == 15

def test_reduction_cost_no_symmetry_unchanged():
    assert reduction_cost(input_shape=(5, 5), axis=None) == 25
