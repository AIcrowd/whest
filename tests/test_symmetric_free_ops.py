"""Tests for automatic symmetric annotation on tensor creation operations."""

import numpy as np

import mechestim as me
from mechestim._symmetric import SymmetricTensor

# ---------------------------------------------------------------------------
# Tier 1: Always symmetric (identity / diagonal)
# ---------------------------------------------------------------------------


class TestEye:
    def test_square_is_symmetric(self):
        result = me.eye(3)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]
        np.testing.assert_array_equal(result, np.eye(3))

    def test_square_explicit_M_equal(self):
        result = me.eye(4, M=4)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]

    def test_non_square_not_symmetric(self):
        result = me.eye(3, M=4)
        assert not isinstance(result, SymmetricTensor)

    def test_off_diagonal_not_symmetric(self):
        result = me.eye(3, k=1)
        assert not isinstance(result, SymmetricTensor)

    def test_off_diagonal_non_square_not_symmetric(self):
        result = me.eye(3, M=4, k=2)
        assert not isinstance(result, SymmetricTensor)

    def test_dtype_preserved(self):
        result = me.eye(3, dtype=np.float32)
        assert isinstance(result, SymmetricTensor)
        assert result.dtype == np.float32


class TestIdentity:
    def test_returns_symmetric(self):
        result = me.identity(4)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]
        np.testing.assert_array_equal(result, np.identity(4))

    def test_size_1(self):
        result = me.identity(1)
        assert isinstance(result, SymmetricTensor)


class TestDiag:
    def test_1d_input_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        result = me.diag(v)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]
        np.testing.assert_array_equal(result, np.diag(v))

    def test_1d_off_diagonal_not_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        result = me.diag(v, k=1)
        assert not isinstance(result, SymmetricTensor)

    def test_2d_input_extracts_diagonal(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = me.diag(m)
        assert not isinstance(result, SymmetricTensor)
        assert result.ndim == 1

    def test_list_input(self):
        result = me.diag([1, 2, 3])
        assert isinstance(result, SymmetricTensor)


class TestDiagflat:
    def test_returns_symmetric(self):
        result = me.diagflat([1, 2, 3])
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]
        np.testing.assert_array_equal(result, np.diagflat([1, 2, 3]))

    def test_off_diagonal_not_symmetric(self):
        result = me.diagflat([1, 2, 3], k=1)
        assert not isinstance(result, SymmetricTensor)

    def test_2d_input_flattened(self):
        result = me.diagflat([[1, 2], [3, 4]])
        assert isinstance(result, SymmetricTensor)
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# Tier 2: Constant-fill with square shape
# ---------------------------------------------------------------------------


class TestZeros:
    def test_square_symmetric(self):
        result = me.zeros((3, 3))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]

    def test_non_square_not_symmetric(self):
        result = me.zeros((3, 4))
        assert not isinstance(result, SymmetricTensor)

    def test_1d_not_symmetric(self):
        result = me.zeros((5,))
        assert not isinstance(result, SymmetricTensor)

    def test_3d_not_symmetric(self):
        result = me.zeros((3, 3, 3))
        assert not isinstance(result, SymmetricTensor)

    def test_scalar_not_symmetric(self):
        result = me.zeros(())
        assert not isinstance(result, SymmetricTensor)


class TestOnes:
    def test_square_symmetric(self):
        result = me.ones((4, 4))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]

    def test_non_square_not_symmetric(self):
        result = me.ones((2, 3))
        assert not isinstance(result, SymmetricTensor)

    def test_1d_not_symmetric(self):
        result = me.ones((10,))
        assert not isinstance(result, SymmetricTensor)


class TestFull:
    def test_square_symmetric(self):
        result = me.full((3, 3), 5.0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]
        assert np.all(result == 5.0)

    def test_non_square_not_symmetric(self):
        result = me.full((3, 4), 5.0)
        assert not isinstance(result, SymmetricTensor)


# ---------------------------------------------------------------------------
# Tier 3: Propagation from input / shape detection
# ---------------------------------------------------------------------------


class TestZerosLike:
    def test_propagates_from_symmetric_tensor(self):
        S = me.as_symmetric(np.eye(3), symmetric_axes=(0, 1))
        result = me.zeros_like(S)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]
        np.testing.assert_array_equal(result, np.zeros((3, 3)))

    def test_plain_input_not_symmetric(self):
        a = np.ones((3, 3))
        result = me.zeros_like(a)
        assert not isinstance(result, SymmetricTensor)

    def test_propagates_multi_group(self):
        data = np.zeros((3, 3, 3, 3))
        S = SymmetricTensor(data, symmetric_axes=[(0, 1), (2, 3)])
        result = me.zeros_like(S)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1), (2, 3)]


class TestOnesLike:
    def test_propagates_from_symmetric_tensor(self):
        S = me.as_symmetric(np.eye(4), symmetric_axes=(0, 1))
        result = me.ones_like(S)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]

    def test_plain_input_not_symmetric(self):
        result = me.ones_like(np.zeros((5, 5)))
        assert not isinstance(result, SymmetricTensor)


class TestFullLike:
    def test_propagates_from_symmetric_tensor(self):
        S = me.as_symmetric(np.eye(3), symmetric_axes=(0, 1))
        result = me.full_like(S, 7.0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetric_axes == [(0, 1)]
        assert np.all(result == 7.0)

    def test_non_square_not_symmetric(self):
        result = me.full_like(np.ones((2, 3)), 1.0)
        assert not isinstance(result, SymmetricTensor)


# ---------------------------------------------------------------------------
# Integration: symmetry flows through downstream operations
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_eye_through_unary_pointwise(self):
        with me.BudgetContext(flop_budget=10**6) as budget:
            eye5 = me.eye(5)
            result = me.exp(eye5)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetric_axes == [(0, 1)]
            # 5*6/2 = 15 unique elements
            assert budget.flops_used == 15

    def test_zeros_add_symmetric_preserves(self):
        with me.BudgetContext(flop_budget=10**6):
            Z = me.zeros((3, 3))
            S = me.as_symmetric(
                np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float),
                symmetric_axes=(0, 1),
            )
            result = me.add(Z, S)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetric_axes == [(0, 1)]

    def test_identity_inv_round_trip(self):
        with me.BudgetContext(flop_budget=10**6):
            ident = me.identity(4)
            inv_ident = me.linalg.inv(ident)
            assert isinstance(inv_ident, SymmetricTensor)
            np.testing.assert_allclose(inv_ident, np.eye(4))

    def test_creation_ops_cost(self):
        with me.BudgetContext(flop_budget=10**6) as budget:
            me.eye(10)  # free
            me.identity(10)  # free
            me.zeros((10, 10))  # free
            me.ones((10, 10))  # free
            me.full((10, 10), 3.14)  # numel(output)=100
            me.diag(np.arange(10, dtype=float))  # numel(input)=10
            me.diagflat(np.arange(5, dtype=float))  # numel(input)=5
            assert budget.flops_used == 100 + 10 + 5
