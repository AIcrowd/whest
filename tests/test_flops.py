"""Tests for analytical helpers and public weighted FLOP APIs."""

import json
from importlib import resources

import pytest

from flopscope import accounting as public_flops
from flopscope._flops import (
    _ceil_log2,
    analytical_pointwise_cost,
    analytical_reduction_cost,
    parse_einsum_subscripts,
    search_cost,
    sort_cost,
)
from flopscope._flops import (
    einsum_cost as analytical_einsum_cost,
)
from flopscope._flops import (
    svd_cost as analytical_svd_cost,
)
from flopscope._symmetric import SymmetryInfo
from flopscope._weights import load_weights, reset_weights


@pytest.fixture(autouse=True)
def _reset_weights():
    reset_weights()
    yield
    reset_weights()


def _write_weights(tmp_path, weights):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"weights": weights}), encoding="utf-8")
    return str(path)


def _packaged_weight(op_name):
    resource = resources.files("flopscope").joinpath("data/default_weights.json")
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)["weights"][op_name]


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


def test_analytical_einsum_cost_matmul():
    assert analytical_einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)]) == 60


def test_analytical_einsum_cost_trace():
    assert analytical_einsum_cost("ii->", shapes=[(10, 10)]) == 10


def test_analytical_einsum_cost_batch_matmul():
    assert analytical_einsum_cost("bij,bjk->bik", shapes=[(2, 3, 4), (2, 4, 5)]) == 120


def test_analytical_einsum_cost_outer_product():
    assert analytical_einsum_cost("i,j->ij", shapes=[(3,), (4,)]) == 12


def test_analytical_einsum_cost_scalar_output():
    assert analytical_einsum_cost("i,i->", shapes=[(5,), (5,)]) == 5


def test_analytical_pointwise_cost():
    assert analytical_pointwise_cost(shape=(256, 256)) == 256 * 256


def test_analytical_pointwise_cost_scalar():
    assert analytical_pointwise_cost(shape=()) == 1


def test_analytical_reduction_cost():
    assert analytical_reduction_cost(input_shape=(256, 256), axis=None) == 256 * 256


def test_analytical_svd_cost():
    assert analytical_svd_cost(m=100, n=50, k=10) == 100 * 50 * 10


def test_analytical_svd_cost_full():
    assert analytical_svd_cost(m=100, n=50, k=None) == 100 * 50 * 50


def test_analytical_pointwise_cost_symmetric():
    info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5))
    assert analytical_pointwise_cost(shape=(5, 5), symmetry_info=info) == 15


def test_analytical_pointwise_cost_partial_symmetry():
    info = SymmetryInfo(symmetric_axes=[(0, 1), (2, 3)], shape=(4, 4, 3, 3))
    assert analytical_pointwise_cost(shape=(4, 4, 3, 3), symmetry_info=info) == 60


def test_analytical_pointwise_cost_no_symmetry_unchanged():
    assert analytical_pointwise_cost(shape=(5, 5)) == 25


def test_analytical_reduction_cost_symmetric():
    info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5))
    assert (
        analytical_reduction_cost(input_shape=(5, 5), axis=None, symmetry_info=info)
        == 15
    )


def test_analytical_reduction_cost_no_symmetry_unchanged():
    assert analytical_reduction_cost(input_shape=(5, 5), axis=None) == 25


def test_analytical_einsum_cost_symmetric_input():
    info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(10, 10, 5))
    cost = analytical_einsum_cost(
        "ijk,k->ij", shapes=[(10, 10, 5), (5,)], operand_symmetries=[info, None]
    )
    dense_cost = 10 * 10 * 5
    assert cost < dense_cost
    assert cost > 0


def test_analytical_einsum_cost_no_operand_symmetry_unchanged():
    cost = analytical_einsum_cost("ij,j->i", shapes=[(10, 10), (10,)])
    assert cost == 100


def test_analytical_einsum_cost_matches_contract_path():
    from flopscope._opt_einsum import contract_path

    cost = analytical_einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)])
    _, info = contract_path("ij,jk->ik", (3, 4), (4, 5), shapes=True)
    assert cost == info.optimized_cost


def test_public_pointwise_cost_is_weighted(tmp_path):
    load_weights(_write_weights(tmp_path, {"exp": 2.5}), use_packaged_default=False)
    assert public_flops.pointwise_cost("exp", shape=(3, 3)) == 22


def test_public_reduction_cost_is_weighted(tmp_path):
    load_weights(_write_weights(tmp_path, {"sum": 3.25}), use_packaged_default=False)
    assert public_flops.reduction_cost("sum", input_shape=(4, 5), axis=None) == 65


def test_public_einsum_cost_is_weighted(tmp_path):
    load_weights(_write_weights(tmp_path, {"einsum": 2.0}), use_packaged_default=False)
    assert public_flops.einsum_cost("ij,jk->ik", shapes=[(3, 4), (4, 5)]) == 120


def test_public_helpers_can_use_packaged_default_weights():
    load_weights(use_packaged_default=True)
    assert public_flops.pointwise_cost("exp", shape=(2, 2)) == int(
        analytical_pointwise_cost((2, 2)) * _packaged_weight("exp")
    )


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
        assert sort_cost(8) == 24

    def test_one_element(self):
        assert sort_cost(1) == 1

    def test_zero_elements(self):
        assert sort_cost(0) == 1


class TestSearchCost:
    def test_basic(self):
        assert search_cost(10, 8) == 30

    def test_one_query(self):
        assert search_cost(1, 1024) == 10

    def test_empty_queries(self):
        assert search_cost(0, 100) == 1
