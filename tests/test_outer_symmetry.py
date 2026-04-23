import numpy as np
import pytest

import whest as we
from whest._budget import BudgetContext
from whest._symmetric import SymmetricTensor


class TestOuterSymmetryPropagation:
    def test_outer_same_vector_wraps_s2(self):
        v = np.array([1.0, 2.0, 3.0])

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.outer(v, v)

        np.testing.assert_allclose(result, np.outer(v, v))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == we.SymmetryGroup.symmetric(axes=(0, 1))

    def test_linalg_outer_same_vector_wraps_s2(self):
        v = np.array([1.0, 2.0, 3.0])

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.linalg.outer(v, v)

        np.testing.assert_allclose(result, np.linalg.outer(v, v))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == we.SymmetryGroup.symmetric(axes=(0, 1))

    @pytest.mark.parametrize("shape", [(2, 2), (2, 2, 2)])
    def test_outer_same_object_any_rank_uses_ravel_semantics(self, shape):
        a = np.arange(np.prod(shape), dtype=float).reshape(shape)

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.outer(a, a)

        expected = np.outer(a, a)
        np.testing.assert_allclose(result, expected)
        assert result.shape == (a.size, a.size)
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == we.SymmetryGroup.symmetric(axes=(0, 1))

    def test_outer_equal_values_but_distinct_objects_stays_plain(self):
        v = np.array([1.0, 2.0, 3.0])

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.outer(v.copy(), v)

        np.testing.assert_allclose(result, np.outer(v, v))
        assert not isinstance(result, SymmetricTensor)

    def test_outer_distinct_operands_stays_plain(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([3.0, 4.0, 5.0])

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.outer(v, w)

        np.testing.assert_allclose(result, np.outer(v, w))
        assert not isinstance(result, SymmetricTensor)

    def test_outer_plain_out_preserves_identity(self):
        v = np.array([1.0, 2.0, 3.0])
        out = np.empty((v.size, v.size))

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.outer(v, v, out=out)

        assert result is out
        assert not isinstance(result, SymmetricTensor)
        np.testing.assert_allclose(out, np.outer(v, v))

    def test_outer_symmetric_out_with_s2_succeeds(self):
        v = np.array([1.0, 2.0, 3.0])
        out = we.SymmetricTensor(
            np.zeros((v.size, v.size)),
            symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
        )

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.outer(v, v, out=out)

        assert result is out
        assert result.symmetry == we.SymmetryGroup.symmetric(axes=(0, 1))
        np.testing.assert_allclose(np.asarray(out), np.outer(v, v))

    def test_outer_symmetric_out_with_wrong_symmetry_fails(self):
        v = np.array([1.0, 2.0, 3.0])
        out = we.SymmetricTensor(
            np.zeros((v.size, v.size)),
            symmetry=we.SymmetryGroup.from_generators([[0, 1, 2]], axes=(0, 1, 2)),
        )

        with BudgetContext(flop_budget=10**8, quiet=True):
            with pytest.raises(ValueError, match="out symmetry does not match"):
                we.outer(v, v, out=out)

    def test_outer_ignores_operand_symmetry_after_ravel(self):
        matrix = we.as_symmetric(
            np.array([[1.0, 2.0], [2.0, 4.0]]),
            symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
        )

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.outer(matrix, matrix)

        np.testing.assert_allclose(result, np.outer(matrix, matrix))
        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == we.SymmetryGroup.symmetric(axes=(0, 1))
