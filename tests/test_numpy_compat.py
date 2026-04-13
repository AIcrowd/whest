"""Verify that whest produces identical results to NumPy for all supported ops."""

import numpy
import pytest

import whest as we
from whest._budget import BudgetContext


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
        we_fn = getattr(we, op_name)
        np_fn = getattr(numpy, op_name)
        inp = self.x_pos if op_name in ("log", "log2", "log10", "sqrt") else self.x
        assert numpy.allclose(we_fn(inp), np_fn(inp), equal_nan=True)


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
        we_fn = getattr(we, op_name)
        np_fn = getattr(numpy, op_name)
        a, b = numpy.abs(self.a) + 0.1, numpy.abs(self.b) + 0.1
        assert numpy.allclose(we_fn(a, b), np_fn(a, b))


class TestReductions:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget
        numpy.random.seed(42)
        self.x = numpy.random.randn(5, 4)

    @pytest.mark.parametrize("op_name", ["sum", "max", "min", "mean", "prod"])
    def test_reduction_full(self, op_name):
        we_fn = getattr(we, op_name)
        np_fn = getattr(numpy, op_name)
        assert numpy.allclose(we_fn(self.x), np_fn(self.x))

    @pytest.mark.parametrize("op_name", ["sum", "max", "min", "mean"])
    def test_reduction_axis(self, op_name):
        we_fn = getattr(we, op_name)
        np_fn = getattr(numpy, op_name)
        assert numpy.allclose(we_fn(self.x, axis=0), np_fn(self.x, axis=0))
        assert numpy.allclose(we_fn(self.x, axis=1), np_fn(self.x, axis=1))


class TestEinsum:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget

    def test_matmul(self):
        A = numpy.random.randn(4, 5)
        B = numpy.random.randn(5, 3)
        assert numpy.allclose(
            we.einsum("ij,jk->ik", A, B), numpy.einsum("ij,jk->ik", A, B)
        )

    def test_trace(self):
        A = numpy.random.randn(4, 4)
        assert numpy.allclose(we.einsum("ii->", A), numpy.einsum("ii->", A))

    def test_batch_matmul(self):
        A = numpy.random.randn(2, 3, 4)
        B = numpy.random.randn(2, 4, 5)
        assert numpy.allclose(
            we.einsum("bij,bjk->bik", A, B), numpy.einsum("bij,bjk->bik", A, B)
        )


class TestDotMatmul:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget

    def test_dot_2d(self):
        A = numpy.random.randn(3, 4)
        B = numpy.random.randn(4, 5)
        assert numpy.allclose(we.dot(A, B), numpy.dot(A, B))

    def test_dot_1d(self):
        a = numpy.random.randn(5)
        b = numpy.random.randn(5)
        assert numpy.allclose(we.dot(a, b), numpy.dot(a, b))

    def test_matmul_2d(self):
        A = numpy.random.randn(3, 4)
        B = numpy.random.randn(4, 5)
        assert numpy.allclose(we.matmul(A, B), numpy.matmul(A, B))


class TestSVD:
    @pytest.fixture(autouse=True)
    def setup(self, budget):
        self.budget = budget

    def test_svd_singular_values(self):
        numpy.random.seed(42)
        A = numpy.random.randn(10, 5)
        _, S_me, _ = we.linalg.svd(A)
        _, S_np, _ = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S_me, S_np)

    def test_svd_truncated_values(self):
        numpy.random.seed(42)
        A = numpy.random.randn(10, 5)
        _, S_me, _ = we.linalg.svd(A, k=3)
        _, S_np, _ = numpy.linalg.svd(A, full_matrices=False)
        assert numpy.allclose(S_me, S_np[:3])


class TestFreeOps:
    def test_zeros_no_context(self):
        x = we.zeros((3, 4))
        assert x.shape == (3, 4)

    def test_constants(self):
        assert we.pi == numpy.pi
        assert we.inf == numpy.inf
        assert issubclass(we.ndarray, numpy.ndarray)


class TestInputCoercion:
    """Verify whest accepts scalars, lists, and arrays like NumPy."""

    def test_unary_scalar(self):
        assert numpy.allclose(we.sqrt(0.5), numpy.sqrt(0.5))

    def test_unary_python_int(self):
        assert numpy.allclose(we.exp(1), numpy.exp(1))

    def test_unary_list(self):
        assert numpy.allclose(we.sqrt([0.25, 1.0, 4.0]), numpy.sqrt([0.25, 1.0, 4.0]))

    def test_reduction_scalar(self):
        assert numpy.allclose(we.sum(5.0), numpy.sum(5.0))

    def test_reduction_list(self):
        assert numpy.allclose(we.sum([1, 2, 3]), numpy.sum([1, 2, 3]))

    def test_clip_list(self):
        assert numpy.allclose(we.clip([1, 5, 10], 2, 8), numpy.clip([1, 5, 10], 2, 8))

    def test_dot_lists(self):
        assert numpy.allclose(
            we.dot([1, 2, 3], [4, 5, 6]), numpy.dot([1, 2, 3], [4, 5, 6])
        )

    def test_linalg_solve_lists(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        b = [5.0, 6.0]
        assert numpy.allclose(we.linalg.solve(A, b), numpy.linalg.solve(A, b))

    def test_fft_list(self):
        assert numpy.allclose(we.fft.fft([1, 2, 3, 4]), numpy.fft.fft([1, 2, 3, 4]))
