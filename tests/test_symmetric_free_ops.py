"""Tests for automatic symmetric annotation on tensor creation operations."""

import numpy as np
import pytest

import whest as we
from whest._symmetric import SymmetricTensor


def _s2():
    return we.SymmetryGroup.symmetric(axes=(0, 1))


def _young_2x2():
    return we.SymmetryGroup.young(blocks=((0, 1), (2, 3)))


# ---------------------------------------------------------------------------
# Tier 1: Always symmetric (identity / diagonal)
# ---------------------------------------------------------------------------


class TestEye:
    def test_eye_still_returns_symmetric_tensor_with_exact_group(self):
        result = we.eye(3)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_square_is_symmetric(self):
        result = we.eye(3)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.eye(3))

    def test_square_explicit_M_equal(self):
        result = we.eye(4, M=4)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_non_square_not_symmetric(self):
        result = we.eye(3, M=4)
        assert not isinstance(result, SymmetricTensor)

    def test_off_diagonal_not_symmetric(self):
        result = we.eye(3, k=1)
        assert not isinstance(result, SymmetricTensor)

    def test_off_diagonal_non_square_not_symmetric(self):
        result = we.eye(3, M=4, k=2)
        assert not isinstance(result, SymmetricTensor)

    def test_dtype_preserved(self):
        result = we.eye(3, dtype=np.float32)
        assert isinstance(result, SymmetricTensor)
        assert result.dtype == np.float32


class TestIdentity:
    def test_returns_symmetric(self):
        result = we.identity(4)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.identity(4))

    def test_size_1(self):
        result = we.identity(1)
        assert isinstance(result, SymmetricTensor)


class TestDiag:
    def test_1d_input_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        result = we.diag(v)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.diag(v))

    def test_1d_off_diagonal_not_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        result = we.diag(v, k=1)
        assert not isinstance(result, SymmetricTensor)

    def test_2d_input_extracts_diagonal(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = we.diag(m)
        assert not isinstance(result, SymmetricTensor)
        assert result.ndim == 1

    def test_list_input(self):
        result = we.diag([1, 2, 3])
        assert isinstance(result, SymmetricTensor)


class TestDiagflat:
    def test_returns_symmetric(self):
        result = we.diagflat([1, 2, 3])
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.diagflat([1, 2, 3]))

    def test_off_diagonal_not_symmetric(self):
        result = we.diagflat([1, 2, 3], k=1)
        assert not isinstance(result, SymmetricTensor)

    def test_2d_input_flattened(self):
        result = we.diagflat([[1, 2], [3, 4]])
        assert isinstance(result, SymmetricTensor)
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# Tier 2: Constant-fill with square shape
# ---------------------------------------------------------------------------


class TestZeros:
    def test_square_symmetric(self):
        result = we.zeros((3, 3))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_non_square_not_symmetric(self):
        result = we.zeros((3, 4))
        assert not isinstance(result, SymmetricTensor)

    def test_1d_not_symmetric(self):
        result = we.zeros((5,))
        assert not isinstance(result, SymmetricTensor)

    def test_3d_not_symmetric(self):
        result = we.zeros((3, 3, 3))
        assert not isinstance(result, SymmetricTensor)

    def test_scalar_not_symmetric(self):
        result = we.zeros(())
        assert not isinstance(result, SymmetricTensor)


class TestOnes:
    def test_square_symmetric(self):
        result = we.ones((4, 4))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_non_square_not_symmetric(self):
        result = we.ones((2, 3))
        assert not isinstance(result, SymmetricTensor)

    def test_1d_not_symmetric(self):
        result = we.ones((10,))
        assert not isinstance(result, SymmetricTensor)


class TestFull:
    def test_square_symmetric(self):
        result = we.full((3, 3), 5.0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        assert np.all(result == 5.0)

    def test_non_square_not_symmetric(self):
        result = we.full((3, 4), 5.0)
        assert not isinstance(result, SymmetricTensor)


# ---------------------------------------------------------------------------
# Tier 3: Propagation from input / shape detection
# ---------------------------------------------------------------------------


class TestZerosLike:
    def test_propagates_from_symmetric_tensor(self):
        S = we.as_symmetric(np.eye(3), symmetry=(0, 1))
        result = we.zeros_like(S)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        np.testing.assert_array_equal(result, np.zeros((3, 3)))

    def test_plain_input_not_symmetric(self):
        a = np.ones((3, 3))
        result = we.zeros_like(a)
        assert not isinstance(result, SymmetricTensor)

    def test_propagates_multi_group(self):
        data = np.zeros((3, 3, 3, 3))
        S = we.as_symmetric(data, symmetry=((0, 1), (2, 3)))
        result = we.zeros_like(S)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _young_2x2()


class TestOnesLike:
    def test_propagates_from_symmetric_tensor(self):
        S = we.as_symmetric(np.eye(4), symmetry=(0, 1))
        result = we.ones_like(S)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()

    def test_plain_input_not_symmetric(self):
        result = we.ones_like(np.zeros((5, 5)))
        assert not isinstance(result, SymmetricTensor)


class TestFullLike:
    def test_propagates_from_symmetric_tensor(self):
        S = we.as_symmetric(np.eye(3), symmetry=(0, 1))
        result = we.full_like(S, 7.0)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _s2()
        assert np.all(result == 7.0)

    def test_non_square_not_symmetric(self):
        result = we.full_like(np.ones((2, 3)), 1.0)
        assert not isinstance(result, SymmetricTensor)

    @pytest.mark.parametrize(
        ("factory", "args"),
        [
            (we.zeros_like, ()),
            (we.ones_like, ()),
            (we.full_like, (5.0,)),
        ],
    )
    def test_shape_override_drops_incompatible_symmetry(self, factory, args):
        source = we.as_symmetric(np.eye(3), symmetry=(0, 1))
        result = factory(source, *args, shape=(2, 3))
        assert result.shape == (2, 3)
        assert not isinstance(result, SymmetricTensor)


# ---------------------------------------------------------------------------
# Integration: symmetry flows through downstream operations
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_eye_through_unary_pointwise(self):
        with we.BudgetContext(flop_budget=10**6) as budget:
            eye5 = we.eye(5)
            result = we.exp(eye5)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == _s2()
            # 5*6/2 = 15 unique elements
            assert budget.flops_used == 15

    def test_zeros_add_symmetric_preserves(self):
        with we.BudgetContext(flop_budget=10**6):
            Z = we.zeros((3, 3))
            S = we.as_symmetric(
                np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float),
                symmetry=(0, 1),
            )
            result = we.add(Z, S)
            assert isinstance(result, SymmetricTensor)
            assert result.symmetry == _s2()

    def test_identity_inv_round_trip(self):
        with we.BudgetContext(flop_budget=10**6):
            ident = we.identity(4)
            inv_ident = we.linalg.inv(ident)
            assert isinstance(inv_ident, SymmetricTensor)
            np.testing.assert_allclose(inv_ident, np.eye(4))

    def test_issue_42_broadcast_and_pointwise_keep_block_symmetry(self):
        n = 3
        A = we.as_symmetric(
            np.ones((n, n, n, n)),
            symmetry=we.SymmetryGroup.symmetric(axes=(0, 1, 2, 3)),
        )
        B = we.as_symmetric(np.ones((n, n)), symmetry=(0, 1))

        B_bc = we.broadcast_to(B, A.shape)
        assert isinstance(B_bc, SymmetricTensor)
        assert B_bc.symmetry == _young_2x2()

        result = we.multiply(A, B)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == _young_2x2()

    def test_batched_inv_preserves_exact_group(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        B = np.array([[5.0, 0.5], [0.5, 2.5]])
        C = np.array([[6.0, 1.5], [1.5, 4.0]])
        batched = np.empty((2, 2, 2, 2), dtype=float)
        batched[0, 0] = A
        batched[0, 1] = B
        batched[1, 0] = B
        batched[1, 1] = C

        exact = we.as_symmetric(batched, symmetry=((0, 1), (2, 3)))

        with we.BudgetContext(flop_budget=10**6):
            result = we.linalg.inv(exact)

        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == exact.symmetry

    def test_creation_ops_cost(self):
        with we.BudgetContext(flop_budget=10**6) as budget:
            we.eye(10)  # free
            we.identity(10)  # free
            we.zeros((10, 10))  # free
            we.ones((10, 10))  # free
            we.full((10, 10), 3.14)  # numel(output)=100
            we.diag(np.arange(10, dtype=float))  # numel(output)=100
            we.diagflat(np.arange(5, dtype=float))  # numel(output)=25
            assert budget.flops_used == 100 + 100 + 25
