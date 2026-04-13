"""Tests for FLOP cost calculators."""

from whest._flops import (
    einsum_cost,
    parse_einsum_subscripts,
    pointwise_cost,
    reduction_cost,
    svd_cost,
)


def test_parse_matmul():
    inputs, output = parse_einsum_subscripts("ij,jk->ik")
    assert inputs == [["i", "j"], ["j", "k"]]
    assert output == ["i", "k"]


def test_parse_trace():
    inputs, output = parse_einsum_subscripts("ii->")
    assert inputs == [["i", "i"]]
    assert output == []


def test_parse_implicit():
    inputs, output = parse_einsum_subscripts("ij,jk")
    assert inputs == [["i", "j"], ["j", "k"]]
    assert output == ["i", "k"]


def test_einsum_cost_matmul():
    assert (
        einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)]) == 60
    )  # 3*4*5 * op_factor(1), FMA=1


def test_einsum_cost_trace():
    assert einsum_cost("ii->", shapes=[(10, 10)]) == 10  # 10 * op_factor(1), FMA=1


def test_einsum_cost_batch_matmul():
    assert (
        einsum_cost("bij,bjk->bik", shapes=[(2, 3, 4), (2, 4, 5)]) == 120
    )  # 2*3*4*5 * op_factor(1), FMA=1


def test_einsum_cost_outer_product():
    assert (
        einsum_cost("i,j->ij", shapes=[(3,), (4,)]) == 12
    )  # no inner product, op_factor=1


def test_einsum_cost_scalar_output():
    assert einsum_cost("i,i->", shapes=[(5,), (5,)]) == 5  # 5 * op_factor(1), FMA=1


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


from whest._symmetric import SymmetryInfo


def test_pointwise_cost_symmetric():
    info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5))
    assert pointwise_cost(shape=(5, 5), symmetry_info=info) == 15


def test_pointwise_cost_partial_symmetry():
    info = SymmetryInfo(symmetric_axes=[(0, 1), (2, 3)], shape=(4, 4, 3, 3))
    assert pointwise_cost(shape=(4, 4, 3, 3), symmetry_info=info) == 60


def test_pointwise_cost_no_symmetry_unchanged():
    assert pointwise_cost(shape=(5, 5)) == 25


def test_reduction_cost_symmetric():
    info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5))
    assert reduction_cost(input_shape=(5, 5), axis=None, symmetry_info=info) == 15


def test_reduction_cost_no_symmetry_unchanged():
    assert reduction_cost(input_shape=(5, 5), axis=None) == 25


def test_einsum_cost_symmetric_input():
    # Use a case where symmetric indices survive in the output:
    # "ijk,k->ij" with S2 on {i,j}. Both i and j survive, so the
    # symmetric group provides a real cost reduction.
    info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(10, 10, 5))
    cost = einsum_cost(
        "ijk,k->ij", shapes=[(10, 10, 5), (5,)], operand_symmetries=[info, None]
    )
    dense_cost = 10 * 10 * 5  # 500 with FMA=1 op_factor
    assert cost < dense_cost  # symmetry reduces cost
    assert cost > 0


def test_einsum_cost_no_operand_symmetry_unchanged():
    cost = einsum_cost("ij,j->i", shapes=[(10, 10), (10,)])
    assert cost == 100  # 10*10 * op_factor(1), FMA=1


def test_einsum_cost_matches_contract_path():
    from whest._opt_einsum import contract_path

    # Verify einsum_cost delegates correctly for a simple matmul
    cost = einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)])
    _, info = contract_path("ij,jk->ik", (3, 4), (4, 5), shapes=True)
    assert cost == info.optimized_cost


from whest._flops import _ceil_log2, search_cost, sort_cost


class TestCeilLog2:
    def test_one(self):
        assert _ceil_log2(1) == 1

    def test_two(self):
        assert _ceil_log2(2) == 1

    def test_three(self):
        assert _ceil_log2(3) == 2

    def test_power_of_two(self):
        assert _ceil_log2(8) == 3

    def test_large(self):
        assert _ceil_log2(1000) == 10

    def test_zero(self):
        assert _ceil_log2(0) == 1


class TestSortCost:
    def test_basic(self):
        # sort 8 elements: 8 * ceil(log2(8)) = 8 * 3 = 24
        assert sort_cost(8) == 24

    def test_one_element(self):
        assert sort_cost(1) == 1

    def test_zero_elements(self):
        assert sort_cost(0) == 1


class TestSearchCost:
    def test_basic(self):
        # 10 queries into sorted array of 8: 10 * ceil(log2(8)) = 10 * 3 = 30
        assert search_cost(10, 8) == 30

    def test_one_query(self):
        assert search_cost(1, 1024) == 10

    def test_empty_queries(self):
        assert search_cost(0, 100) == 1
