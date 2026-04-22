"""Tests for SymmetryInfo, SymmetricTensor, and as_symmetric."""

import pickle

import numpy
import numpy as np
import pytest

from whest._budget import BudgetContext
from whest._ndarray import WhestArray
from whest._perm_group import SymmetryGroup as PermutationGroup
from whest._symmetric import (
    SymmetricTensor,
    SymmetryInfo,
    as_symmetric,
    is_symmetric,
    symmetrize,
)
from whest.errors import SymmetryError


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
        assert info.symmetry_factor == 4

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
# Task 3: SymmetricTensor and as_symmetric
# ---------------------------------------------------------------------------


def _make_symmetric_matrix(n: int = 5) -> np.ndarray:
    """Create a random symmetric matrix."""
    a = np.random.default_rng(42).standard_normal((n, n))
    return (a + a.T) / 2


def test_as_symmetric_exposes_single_symmetry_object():
    g = PermutationGroup.symmetric(axes=(0, 1))
    t = as_symmetric(np.eye(3), symmetry=g)
    assert t.symmetry == g
    assert not hasattr(t, "symmetry_info")
    assert not hasattr(t, "symmetric_axes")


def test_as_symmetric_rejects_list_of_groups():
    with pytest.raises(TypeError, match="single SymmetryGroup"):
        as_symmetric(
            np.zeros((2, 2, 2, 2)),
            symmetry=[
                PermutationGroup.symmetric(axes=(0, 1)),
                PermutationGroup.symmetric(axes=(2, 3)),
            ],
        )


def test_old_symmetry_payload_raises_explicit_error():
    g = PermutationGroup.symmetric(axes=(0, 1))
    t = as_symmetric(np.eye(3), symmetry=g)
    payload = t.__reduce__()
    legacy_state = payload[2] + ([(0, 1)],)
    rebuilt = SymmetricTensor(np.zeros((3, 3)), symmetry=g)
    with pytest.raises(ValueError, match="legacy symmetry payload"):
        rebuilt.__setstate__(legacy_state)


def test_array_finalize_is_conservative():
    g = PermutationGroup.symmetric(axes=(0, 1))
    t = as_symmetric(np.eye(3), symmetry=g)
    finalized = np.asarray(t).view(SymmetricTensor)
    assert finalized.symmetry is None


def test_copy_preserves_symmetry_but_reshape_and_ravel_drop():
    g = PermutationGroup.symmetric(axes=(0, 1))
    t = as_symmetric(np.eye(3), symmetry=g)

    copied = t.copy()
    reshaped = t.reshape(-1)
    raveled = t.ravel()
    flattened = t.flatten()
    cast = t.astype(np.float32)

    assert isinstance(copied, SymmetricTensor)
    assert copied.symmetry == g
    assert not isinstance(reshaped, SymmetricTensor)
    assert not isinstance(raveled, SymmetricTensor)
    assert not isinstance(flattened, SymmetricTensor)
    assert not isinstance(cast, SymmetricTensor)


def test_transpose_remaps_symmetry_explicitly():
    g = PermutationGroup.symmetric(axes=(0, 2))
    data = np.arange(27.0).reshape(3, 3, 3)
    t = symmetrize(data, symmetry=g)

    out = t.transpose((2, 1, 0))

    assert isinstance(out, SymmetricTensor)
    assert out.symmetry == PermutationGroup.symmetric(axes=(0, 2))


def test_swapaxes_remaps_symmetry_explicitly():
    g = PermutationGroup.symmetric(axes=(0, 2))
    data = np.arange(27.0).reshape(3, 3, 3)
    t = symmetrize(data, symmetry=g)

    out = t.swapaxes(0, 1)

    assert isinstance(out, SymmetricTensor)
    assert out.symmetry == PermutationGroup.symmetric(axes=(1, 2))


def test_T_remaps_symmetry_explicitly():
    g = PermutationGroup.symmetric(axes=(0, 1))
    t = as_symmetric(np.eye(3), symmetry=g)

    out = t.T

    assert isinstance(out, SymmetricTensor)
    assert out.symmetry == g


def test_is_symmetric_false_for_non_symmetric_data():
    g = PermutationGroup.symmetric(axes=(0, 1))
    x = np.array([[1, 2], [3, 4]])
    assert is_symmetric(x, symmetry=g) is False


def test_offset_slice_drops_symmetry():
    g = PermutationGroup.symmetric(axes=(0, 1))
    t = as_symmetric(np.eye(4), symmetry=g)

    out = t[1:, 1:]

    assert isinstance(out, WhestArray)
    assert not isinstance(out, SymmetricTensor)


def test_stepped_slice_drops_symmetry():
    g = PermutationGroup.symmetric(axes=(0, 1))
    t = as_symmetric(np.eye(4), symmetry=g)

    out = t[::-1, ::-1]

    assert isinstance(out, WhestArray)
    assert not isinstance(out, SymmetricTensor)


def test_numpy_negative_downgrades_without_symmetry_metadata():
    g = PermutationGroup.symmetric(axes=(0, 1))
    t = as_symmetric(np.eye(3), symmetry=g)

    out = np.negative(t)

    assert isinstance(out, WhestArray)
    assert not isinstance(out, SymmetricTensor)


def test_squeeze_downgrades_without_symmetry_metadata():
    g = PermutationGroup.symmetric(axes=(1, 2))
    data = np.zeros((1, 3, 3))
    data[0] = np.eye(3)
    t = as_symmetric(data, symmetry=g)

    out = t.squeeze()

    assert isinstance(out, WhestArray)
    assert not isinstance(out, SymmetricTensor)


class TestSymmetricTensor:
    """Tests for SymmetricTensor ndarray subclass."""

    def test_is_ndarray_and_symmetric_tensor(self):
        data = _make_symmetric_matrix()
        st = as_symmetric(data, symmetry=(0, 1))
        assert isinstance(st, np.ndarray)
        assert isinstance(st, SymmetricTensor)

    def test_accepts_symmetric_data(self):
        data = _make_symmetric_matrix()
        st = as_symmetric(data, symmetry=(0, 1))
        assert st.shape == (5, 5)

    def test_rejects_non_symmetric_data(self):
        rng = np.random.default_rng(99)
        data = rng.standard_normal((5, 5))
        with pytest.raises(SymmetryError):
            as_symmetric(data, symmetry=(0, 1))

    def test_multiple_groups_shorthand(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal((4, 4, 3, 3))
        a = (a + a.transpose(1, 0, 2, 3)) / 2
        a = (a + a.transpose(0, 1, 3, 2)) / 2
        st = as_symmetric(a, symmetry=((0, 1), (2, 3)))
        assert st.symmetry == PermutationGroup.young(blocks=((0, 1), (2, 3)))

    def test_single_tuple_shorthand(self):
        data = _make_symmetric_matrix()
        st = as_symmetric(data, symmetry=(0, 1))
        assert st.symmetry == PermutationGroup.symmetric(axes=(0, 1))

    def test_shape_dtype_preserved(self):
        data = _make_symmetric_matrix().astype(np.float32)
        st = as_symmetric(data, symmetry=(0, 1))
        assert st.shape == (5, 5)
        assert st.dtype == np.float32

    def test_slicing_loses_symmetry(self):
        data = _make_symmetric_matrix()
        st = as_symmetric(data, symmetry=(0, 1))
        row = st[0]
        assert not isinstance(row, SymmetricTensor)
        assert isinstance(row, np.ndarray)

    def test_within_tolerance_passes(self):
        data = _make_symmetric_matrix()
        data[0, 1] += 1e-8
        data[1, 0] -= 1e-8
        as_symmetric(data, symmetry=(0, 1))

    def test_symmetric_plus_dense_same_shape_drops_symmetry(self):
        st = as_symmetric(_make_symmetric_matrix(4), symmetry=(0, 1))
        dense = np.arange(16.0).reshape(4, 4)

        with BudgetContext(flop_budget=10**6):
            result = st + dense

        assert isinstance(result, WhestArray)
        assert not isinstance(result, SymmetricTensor)
        assert np.allclose(result, np.asarray(st) + dense)

    def test_symmetric_plus_scalar_preserves_shared_symmetry(self):
        st = as_symmetric(_make_symmetric_matrix(4), symmetry=(0, 1))

        with BudgetContext(flop_budget=10**6):
            result = st + 2.0

        assert isinstance(result, SymmetricTensor)
        assert result.symmetry == PermutationGroup.symmetric(axes=(0, 1))
        assert np.allclose(result, np.asarray(st) + 2.0)

    def test_exceeds_tolerance_fails(self):
        data = _make_symmetric_matrix()
        data[0, 1] += 1e-3
        with pytest.raises(SymmetryError):
            as_symmetric(data, symmetry=(0, 1))

    def test_pickle_roundtrip(self):
        g = PermutationGroup.symmetric(axes=(0, 1))
        st = as_symmetric(_make_symmetric_matrix(), symmetry=g)
        loaded = pickle.loads(pickle.dumps(st))
        assert isinstance(loaded, SymmetricTensor)
        assert loaded.symmetry == g
        np.testing.assert_array_equal(np.asarray(loaded), np.asarray(st))


class TestPublicAPI:
    def test_import_from_whest(self):
        import whest as we

        assert hasattr(we, "SymmetricTensor")
        assert hasattr(we, "SymmetryInfo")
        assert hasattr(we, "as_symmetric")
        assert hasattr(we, "symmetrize")

    def test_symmetrize(self):
        group = PermutationGroup.symmetric(axes=(0, 1))
        base = np.arange(16.0).reshape(4, 4)

        S = symmetrize(base, symmetry=group)

        assert isinstance(S, SymmetricTensor)
        assert S.symmetry == group
        assert S.is_symmetric()
        assert S.shape == (4, 4)

    def test_symmetrize_invalid_shape_raises(self):
        with pytest.raises(SymmetryError):
            symmetrize(np.ones((2, 3)), symmetry=PermutationGroup.symmetric(axes=(0, 1)))

    def test_random_symmetric(self):
        import whest as we

        group = PermutationGroup.symmetric(axes=(0, 1))
        S = we.random.symmetric((4, 4), group)
        assert isinstance(S, SymmetricTensor)
        assert S.symmetry == group
        assert S.is_symmetric()

    def test_import_symmetry_info_from_flops(self):
        from whest.flops import SymmetryInfo

        assert SymmetryInfo is not None


class TestEndToEnd:
    def test_covprop_workflow(self):
        import whest as we

        n, d = 5, 20
        X = numpy.random.randn(d, n)

        with BudgetContext(flop_budget=10**8, quiet=True) as budget:
            cov = we.einsum("ki,kj->ij", X, X, symmetric_axes=[(0, 1)])
            assert isinstance(cov, SymmetricTensor)
            cov_cost = budget.flops_used

            before = budget.flops_used
            exp_cov = we.exp(cov)
            pointwise_cost_actual = budget.flops_used - before
            assert isinstance(exp_cov, SymmetricTensor)
            assert pointwise_cost_actual == n * (n + 1) // 2

            cov_pd = cov + we.multiply(
                we.as_symmetric(numpy.eye(n), symmetry=(0, 1)),
                numpy.asarray(float(n)),
            )
            b = numpy.ones(n)
            before = budget.flops_used
            x = we.linalg.solve(cov_pd, b)
            solve_cost_actual = budget.flops_used - before
            assert not isinstance(x, SymmetricTensor)
            assert solve_cost_actual == n**3
            assert cov_cost > 0

    def test_symmetry_preserved_through_chain(self):
        import whest as we

        data = numpy.eye(4) + 0.5
        S = we.as_symmetric(data, symmetry=(0, 1))

        with BudgetContext(flop_budget=10**8, quiet=True):
            r1 = we.exp(S)
            assert isinstance(r1, SymmetricTensor)
            r2 = we.log(r1)
            assert isinstance(r2, SymmetricTensor)
            r3 = we.sqrt(we.abs(r2))
            assert isinstance(r3, SymmetricTensor)

    def test_symmetry_lost_on_matmul(self):
        import whest as we

        A = we.as_symmetric(numpy.eye(3), symmetry=(0, 1))
        B = numpy.ones((3, 3))

        with BudgetContext(flop_budget=10**8, quiet=True):
            result = we.einsum("ij,jk->ik", A, B)
            assert not isinstance(result, SymmetricTensor)


class TestSymmetryInfoPermGroup:
    def test_groups_property_from_symmetric_axes_uses_axes_only_constructor(self):
        info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5))
        assert len(info.groups) == 1
        assert info.groups[0].axes == (0, 1)
        assert info.groups[0].degree == 2

    def test_groups_property_from_symmetric_axes(self):
        info = SymmetryInfo(symmetric_axes=[(0, 1)], shape=(5, 5))
        assert len(info.groups) == 1
        assert isinstance(info.groups[0], PermutationGroup)
        assert info.groups[0].is_symmetric()
        assert info.groups[0].degree == 2
        assert info.groups[0].axes == (0, 1)

    def test_unique_elements_s3_unchanged(self):
        info = SymmetryInfo(symmetric_axes=[(0, 1, 2)], shape=(5, 5, 5))
        assert info.unique_elements == 35

    def test_unique_elements_c3_via_groups(self):
        c3 = PermutationGroup.cyclic(axes=(0, 1, 2))
        info = SymmetryInfo(groups=[c3], shape=(5, 5, 5))
        expected = (125 + 10) // 3
        assert info.unique_elements == expected
        info_s3 = SymmetryInfo(symmetric_axes=[(0, 1, 2)], shape=(5, 5, 5))
        assert info.unique_elements > info_s3.unique_elements

    def test_symmetric_axes_backward_compat(self):
        info = SymmetryInfo(symmetric_axes=[(0, 1), (2, 3)], shape=(4, 4, 3, 3))
        assert info.symmetric_axes == [(0, 1), (2, 3)]
        assert info.unique_elements == 60
        assert info.symmetry_factor == 4


class TestAsSymmetricPermGroup:
    def test_symmetry_param_s2(self):
        data = numpy.array([[2.0, 1.0], [1.0, 3.0]])
        g = PermutationGroup.symmetric(axes=(0, 1))
        T = as_symmetric(data, symmetry=g)
        assert isinstance(T, SymmetricTensor)
        assert T.symmetry == g

    def test_symmetry_param_c3(self):
        n = 4
        data = numpy.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    data[i, j, k] = i * 100 + j * 10 + k
        rotated1 = data.transpose(1, 2, 0)
        rotated2 = data.transpose(2, 0, 1)
        sym_data = (data + rotated1 + rotated2) / 3.0
        g = PermutationGroup.cyclic(axes=(0, 1, 2))
        T = as_symmetric(sym_data, symmetry=g)
        assert isinstance(T, SymmetricTensor)
        assert T.symmetry == g

    def test_legacy_symmetric_axes_keyword_rejected(self):
        data = numpy.eye(3)
        with pytest.raises(TypeError):
            as_symmetric(data, symmetric_axes=(0, 1))
