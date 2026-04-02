"""Verify that mechestim produces identical results to NumPy for all supported ops."""

import numpy
import pytest

import mechestim as me
from mechestim._budget import BudgetContext


@pytest.fixture
def budget():
    with BudgetContext(flop_budget=10**9) as b:
        yield b


class TestUnaryOps:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget
        numpy.random.seed(42)
        self.x = numpy.random.randn(5, 4).astype(numpy.float64)
        self.x_pos = numpy.abs(self.x) + 0.01

    @pytest.mark.parametrize(
        "op_name",
        [
            "exp",
            "log",
            "log2",
            "log10",
            "abs",
            "negative",
            "sqrt",
            "square",
            "sin",
            "cos",
            "tanh",
            "sign",
            "ceil",
            "floor",
        ],
    )
    def test_unary(self, op_name):
        me_fn = getattr(me, op_name)
        np_fn = getattr(numpy, op_name)
        inp = self.x_pos if op_name in ("log", "log2", "log10", "sqrt") else self.x
        assert numpy.allclose(me_fn(inp), np_fn(inp), equal_nan=True)


class TestBinaryOps:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget
        numpy.random.seed(42)
        self.a = numpy.random.randn(3, 4)
        self.b = numpy.random.randn(3, 4) + 2.0

    @pytest.mark.parametrize(
        "op_name",
        [
            "add",
            "subtract",
            "multiply",
            "divide",
            "maximum",
            "minimum",
            "power",
            "mod",
        ],
    )
    def test_binary(self, op_name):
        me_fn = getattr(me, op_name)
        np_fn = getattr(numpy, op_name)
        a, b = numpy.abs(self.a) + 0.1, numpy.abs(self.b) + 0.1
        assert numpy.allclose(me_fn(a, b), np_fn(a, b))


class TestReductions:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget
        numpy.random.seed(42)
        self.x = numpy.random.randn(5, 4)

    @pytest.mark.parametrize("op_name", ["sum", "max", "min", "mean", "prod"])
    def test_reduction_full(self, op_name):
        me_fn = getattr(me, op_name)
        np_fn = getattr(numpy, op_name)
        assert numpy.allclose(me_fn(self.x), np_fn(self.x))

    @pytest.mark.parametrize("op_name", ["sum", "max", "min", "mean"])
    def test_reduction_axis(self, op_name):
        me_fn = getattr(me, op_name)
        np_fn = getattr(numpy, op_name)
        assert numpy.allclose(me_fn(self.x, axis=0), np_fn(self.x, axis=0))
        assert numpy.allclose(me_fn(self.x, axis=1), np_fn(self.x, axis=1))


class TestEinsum:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget

    def test_matmul(self):
        A = numpy.random.randn(4, 5)
        B = numpy.random.randn(5, 3)
        assert numpy.allclose(
            me.einsum("ij,jk->ik", A, B), numpy.einsum("ij,jk->ik", A, B)
        )

    def test_trace(self):
        A = numpy.random.randn(4, 4)
        assert numpy.allclose(me.einsum("ii->", A), numpy.einsum("ii->", A))

    def test_batch_matmul(self):
        A = numpy.random.randn(2, 3, 4)
        B = numpy.random.randn(2, 4, 5)
        assert numpy.allclose(
            me.einsum("bij,bjk->bik", A, B), numpy.einsum("bij,bjk->bik", A, B)
        )


class TestDotMatmul:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget

    def test_dot_2d(self):
        A = numpy.random.randn(3, 4)
        B = numpy.random.randn(4, 5)
        assert numpy.allclose(me.dot(A, B), numpy.dot(A, B))

    def test_dot_1d(self):
        a = numpy.random.randn(5)
        b = numpy.random.randn(5)
        assert numpy.allclose(me.dot(a, b), numpy.dot(a, b))

    def test_matmul_2d(self):
        A = numpy.random.randn(3, 4)
        B = numpy.random.randn(4, 5)
        assert numpy.allclose(me.matmul(A, B), numpy.matmul(A, B))


class TestSVD:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget

    def test_svd_singular_values(self):
        numpy.random.seed(42)
        A = numpy.random.randn(10, 5)
        _, S_me, _ = me.linalg.svd(A)
        _, S_np, _ = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S_me, S_np)

    def test_svd_truncated_values(self):
        numpy.random.seed(42)
        A = numpy.random.randn(10, 5)
        _, S_me, _ = me.linalg.svd(A, k=3)
        _, S_np, _ = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S_me, S_np[:3])


class TestFreeOps:
    def test_zeros_no_context(self):
        x = me.zeros((3, 4))
        assert x.shape == (3, 4)

    def test_constants(self):
        assert me.pi == numpy.pi
        assert me.inf == numpy.inf
        assert me.ndarray is numpy.ndarray
