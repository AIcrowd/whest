"""Tests for SymmetryInfo, SymmetricTensor, and as_symmetric."""

import pickle

import numpy
import numpy as np
import pytest

from mechestim._budget import BudgetContext
from mechestim._symmetric import (
    SymmetricTensor,
    SymmetryInfo,
    as_symmetric,
)
from mechestim.errors import SymmetryError


class TestSymmetryInfo:
    """Tests for the SymmetryInfo frozen dataclass."""

    def test_single_group_unique_elements(self):
        """Single group (0,1) on (5,5): C(5+2-1, 2) = C(6,2) = 15."""
        info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5))
        assert info.unique_elements == 15

    def test_single_group_symmetry_factor(self):
        """Single group (0,1) on (5,5): 2! = 2."""
        info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5))
        assert info.symmetry_factor == 2

    def test_partial_symmetry(self):
        """Two groups [(0,1),(2,3)] on (4,4,3,3): C(4+1,2)*C(3+1,2) = 10*6 = 60."""
        info = SymmetryInfo(symmetric_axes=[(0, 1), (2, 3)], shape=(4, 4, 3, 3))
        assert info.unique_elements == 60
        assert info.symmetry_factor == 4  # 2! * 2! = 4

    def test_three_way_symmetry(self):
        """Three-way (0,1,2) on (3,3,3): 3! = 6, C(3+2, 3) = C(5,3) = 10."""
        info = SymmetryInfo(symmetric_axes=[(0, 1, 2)], shape=(3, 3, 3))
        assert info.symmetry_factor == 6
        assert info.unique_elements == 10

    def test_mixed_symmetric_and_free(self):
        """(0,1) on (5,5,8): C(6,2) * 8 = 15 * 8 = 120."""
        info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5, 8))
        assert info.unique_elements == 120

    def test_frozen(self):
        """SymmetryInfo is frozen; reassignment raises."""
        info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5))
        with pytest.raises(AttributeError):
            info.shape = (3, 3)

    def test_post_init_normalizes_axes(self):
        """Axes are normalized to sorted tuples."""
        info = SymmetryInfo(symmetric_axes=[(1, 0)], shape=(5, 5))
        assert info.symmetric_axes == [(0, 1)]


# ---------------------------------------------------------------------------
# Task 2: SymmetricTensor and as_symmetric
# ---------------------------------------------------------------------------


def _make_symmetric_matrix(n: int = 5) -> np.ndarray:
    """Create a random symmetric matrix."""
    a = np.random.default_rng(42).standard_normal((n, n))
    return (a + a.T) / 2


class TestSymmetricTensor:
    """Tests for SymmetricTensor ndarray subclass."""

    def test_is_ndarray_and_symmetric_tensor(self):
        data = _make_symmetric_matrix()
        st = as_symmetric(data, (0, 1))
        assert isinstance(st, np.ndarray)
        assert isinstance(st, SymmetricTensor)

    def test_symmetric_axes_attribute(self):
        data = _make_symmetric_matrix()
        st = as_symmetric(data, (0, 1))
        assert st.symmetric_axes == [(0, 1)]

    def test_symmetry_info_property(self):
        data = _make_symmetric_matrix()
        st = as_symmetric(data, (0, 1))
        info = st.symmetry_info
        assert isinstance(info, SymmetryInfo)
        assert info.shape == (5, 5)
        assert info.unique_elements == 15

    def test_accepts_symmetric_data(self):
        """Valid symmetric data should not raise."""
        data = _make_symmetric_matrix()
        st = as_symmetric(data, (0, 1))  # should not raise
        assert st.shape == (5, 5)

    def test_rejects_non_symmetric_data(self):
        """Non-symmetric data raises SymmetryError."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((5, 5))  # not symmetric
        with pytest.raises(SymmetryError):
            as_symmetric(data, (0, 1))

    def test_multiple_groups(self):
        """Multiple symmetry groups work."""
        rng = np.random.default_rng(7)
        a = rng.standard_normal((4, 4, 3, 3))
        # Make dims (0,1) and (2,3) symmetric
        a = (a + a.transpose(1, 0, 2, 3)) / 2
        a = (a + a.transpose(0, 1, 3, 2)) / 2
        st = as_symmetric(a, [(0, 1), (2, 3)])
        assert st.symmetric_axes == [(0, 1), (2, 3)]

    def test_single_tuple_shorthand(self):
        """Single tuple symmetric_axes shorthand: (0,1) treated as [(0,1)]."""
        data = _make_symmetric_matrix()
        st = as_symmetric(data, (0, 1))
        assert st.symmetric_axes == [(0, 1)]

    def test_copy_preserves_symmetry(self):
        data = _make_symmetric_matrix()
        st = as_symmetric(data, (0, 1))
        cp = st.copy()
        assert isinstance(cp, SymmetricTensor)
        assert cp.symmetric_axes == [(0, 1)]

    def test_shape_dtype_preserved(self):
        data = _make_symmetric_matrix().astype(np.float32)
        st = as_symmetric(data, (0, 1))
        assert st.shape == (5, 5)
        assert st.dtype == np.float32

    def test_slicing_loses_symmetry(self):
        """Slicing returns plain ndarray, not SymmetricTensor."""
        data = _make_symmetric_matrix()
        st = as_symmetric(data, (0, 1))
        row = st[0]
        assert not isinstance(row, SymmetricTensor)
        assert isinstance(row, np.ndarray)

    def test_within_tolerance_passes(self):
        """Deviation of 1e-8 is within default tolerance."""
        data = _make_symmetric_matrix()
        data[0, 1] += 1e-8
        data[1, 0] -= 1e-8  # tiny asymmetry
        as_symmetric(data, (0, 1))  # should not raise

    def test_exceeds_tolerance_fails(self):
        """Deviation of 1e-3 exceeds tolerance."""
        data = _make_symmetric_matrix()
        data[0, 1] += 1e-3
        with pytest.raises(SymmetryError):
            as_symmetric(data, (0, 1))

    def test_pickle_roundtrip(self):
        """SymmetricTensor survives pickle."""
        data = _make_symmetric_matrix()
        st = as_symmetric(data, (0, 1))
        loaded = pickle.loads(pickle.dumps(st))
        assert isinstance(loaded, SymmetricTensor)
        assert loaded.symmetric_axes == [(0, 1)]
        np.testing.assert_array_equal(loaded, st)


class TestPublicAPI:
    def test_import_from_mechestim(self):
        import mechestim as me

        assert hasattr(me, "SymmetricTensor")
        assert hasattr(me, "SymmetryInfo")
        assert hasattr(me, "as_symmetric")

    def test_import_symmetry_info_from_flops(self):
        from mechestim.flops import SymmetryInfo

        assert SymmetryInfo is not None


class TestEndToEnd:
    def test_covprop_workflow(self):
        """Simulate a covprop-like workflow: build covariance, do pointwise, solve."""
        import mechestim as me

        n, d = 5, 20
        X = numpy.random.randn(d, n)

        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            # Build symmetric covariance: X^T X -> symmetric
            cov = me.einsum("ki,kj->ij", X, X, symmetric_axes=[(0, 1)])
            assert isinstance(cov, SymmetricTensor)
            cov_cost = budget.flops_used

            # Pointwise on symmetric matrix — should get savings
            before = budget.flops_used
            exp_cov = me.exp(cov)
            pointwise_cost_actual = budget.flops_used - before
            assert isinstance(exp_cov, SymmetricTensor)
            assert pointwise_cost_actual == n * (n + 1) // 2  # 15

            # Solve with symmetric matrix — should use Cholesky cost
            # Make it positive definite first
            cov_pd = cov + me.multiply(
                me.as_symmetric(numpy.eye(n), symmetric_axes=(0, 1)),
                numpy.asarray(float(n)),
            )
            b = numpy.ones(n)
            before = budget.flops_used
            x = me.linalg.solve(cov_pd, b)
            solve_cost_actual = budget.flops_used - before
            assert not isinstance(x, SymmetricTensor)
            assert (
                solve_cost_actual == n**3 // 3 + 2 * n**2 * 1
            )  # Cholesky + two triangular solves

    def test_symmetry_preserved_through_chain(self):
        """Chain of unary ops preserves symmetry."""
        import mechestim as me

        data = numpy.eye(4) + 0.5
        S = me.as_symmetric(data, symmetric_axes=(0, 1))

        with BudgetContext(flop_budget=10**8, quiet=True):
            r1 = me.exp(S)
            assert isinstance(r1, SymmetricTensor)
            r2 = me.log(r1)
            assert isinstance(r2, SymmetricTensor)
            r3 = me.sqrt(me.abs(r2))
            assert isinstance(r3, SymmetricTensor)

    def test_symmetry_lost_on_matmul(self):
        """Matmul does not preserve symmetry."""
        import mechestim as me

        A = me.as_symmetric(numpy.eye(3), symmetric_axes=(0, 1))
        B = numpy.ones((3, 3))

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = me.einsum("ij,jk->ik", A, B)
            assert not isinstance(result, SymmetricTensor)
