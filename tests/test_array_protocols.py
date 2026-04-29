"""Tests verifying numpy's __array_ufunc__ and __array_function__ protocols
route numpy calls through flopscope's counted functions when the operands are
FlopscopeArray (or SymmetricTensor).

Includes adversarial coverage for recursion, out= tuples, kwargs passthrough,
mixed operands, unsupported ufunc methods, and identity preservation.

Translated against post-PR-#51 unified SymmetryGroup API.
"""

from __future__ import annotations

import numpy as np
import pytest

import flopscope as flops


# ----- __array_ufunc__: ufunc.__call__ -----

UFUNC_CALL_CASES = [
    ("add", lambda a, b: np.add(a, b), "add"),
    ("multiply", lambda a, b: np.multiply(a, b), "multiply"),
    ("subtract", lambda a, b: np.subtract(a, b), "subtract"),
    ("maximum", lambda a, b: np.maximum(a, b), "maximum"),
]


@pytest.mark.parametrize("name,op,we_name", UFUNC_CALL_CASES)
def test_np_ufunc_call_tracks_flops(name, op, we_name):
    a = flops.random.randn(8)
    b = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        op(a, b)
    we_func = getattr(we, we_name)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        we_func(a, b)
    assert b1.flops_used == b2.flops_used > 0


def test_np_unary_ufunc_call_tracks_flops():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.sin(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        flops.sin(a)
    assert b1.flops_used == b2.flops_used > 0


# ----- __array_ufunc__: ufunc.reduce -----

def test_np_add_reduce_tracks_flops():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.add.reduce(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        flops.sum(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_maximum_reduce_tracks_flops():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.maximum.reduce(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        flops.max(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_add_reduce_2d_matches_axis0_semantics():
    """``ufunc.reduce`` defaults to ``axis=0``, whereas ``flops.sum`` defaults
    to ``axis=None`` (full reduction). The ``__array_ufunc__`` dispatch
    must inject ``axis=0`` so 2D inputs get partial-reduction semantics."""
    a = flops.random.randn(4, 5)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        r1 = np.add.reduce(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        r2 = flops.sum(a, axis=0)
    assert r1.shape == r2.shape == (5,)
    assert b1.flops_used == b2.flops_used > 0


# ----- __array_ufunc__: ufunc.accumulate -----

def test_np_add_accumulate_tracks_flops():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.add.accumulate(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        flops.cumsum(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_add_accumulate_2d_matches_axis0_semantics():
    """Same axis-default issue as reduce: ``ufunc.accumulate`` defaults to
    ``axis=0``, ``flops.cumsum`` defaults to ``axis=None`` (flatten)."""
    a = flops.random.randn(4, 5)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        r1 = np.add.accumulate(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        r2 = flops.cumsum(a, axis=0)
    assert r1.shape == r2.shape == (4, 5)
    assert b1.flops_used == b2.flops_used > 0


# ----- Recursion guards (regression-only) -----

def test_dunder_does_not_recurse_after_protocol_enabled():
    """FlopscopeArray.__add__ → me.add → _np.add must NOT re-dispatch through
    __array_ufunc__ → me.add → ∞.

    The strip-before-NumPy invariant in counted wrappers prevents this.
    """
    a = flops.random.randn(8)
    b = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        c = a + b
    assert bc.flops_used > 0
    assert c.shape == (8,)


def test_np_add_does_not_recurse():
    a = flops.random.randn(8)
    b = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        np.add(a, b)
    assert bc.flops_used > 0


def test_np_sort_does_not_recurse():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        np.sort(a)
    assert bc.flops_used > 0


# ----- ufunc kwargs passthrough -----

def test_np_add_out_unwraps_single_output_tuple():
    """NumPy passes out=(out_arr,) to __array_ufunc__; flopscope expects out=arr."""
    a = flops.random.randn(8)
    b = flops.random.randn(8)
    out = flops.empty_like(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        returned = np.add(a, b, out=out)
    assert returned is out
    assert bc.flops_used > 0


def test_np_add_out_refuses_when_out_symmetry_would_be_destroyed():
    """``np.add(A_sym, B_unsymmetric, out=A_sym)`` would write
    unsymmetric bytes into a buffer whose metadata still claims
    symmetry. Same correctness issue as in-place dunders, just via an
    explicit ``out=``. The wrapper's symmetry validation must refuse
    before the NumPy call writes any bytes.

    Post-PR-#51, this is enforced by ``_prepare_symmetric_out`` in
    ``_pointwise.py``: if the out's symmetry doesn't match the result's,
    it raises (via ``SymmetryError`` or ``ValueError``).
    """
    A = flops.symmetrize(
        flops.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    B = flops.random.randn(4, 4)  # plain FlopscopeArray, no symmetry
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises((ValueError, flops.errors.SymmetryError)):
            np.add(A, B, out=A)


def test_np_add_out_allows_matching_symmetric_out():
    """Positive case: when the operation's output symmetry matches the
    declared symmetry on ``out=``, the call goes through cleanly.
    ``A + 1.0`` preserves every group exactly (binary-with-scalar), so
    writing into a SymmetricTensor ``out`` with the same axes is safe."""
    A = flops.symmetrize(
        flops.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    out = flops.symmetrize(
        flops.zeros((4, 4)),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        ret = np.add(A, 1.0, out=out)
    assert ret is out
    assert isinstance(out, flops.SymmetricTensor)
    assert bc.flops_used > 0


def test_np_transpose_on_whest_fails_loudly_until_supported():
    """Structural NumPy ops are intentionally NOT in the
    ``__array_function__`` allowlist. ``np.transpose(flopscope)`` must
    therefore raise ``TypeError`` rather than silently strip
    symmetry. ``A.T`` (property form) goes through ``__array_finalize__``
    and continues to work — covered by
    ``test_transpose_of_symmetric_preserves_type`` below."""
    a = flops.random.randn(2, 3)
    with pytest.raises(TypeError):
        np.transpose(a)


def test_np_add_where_kwarg_tracks():
    a = flops.random.randn(8)
    b = flops.random.randn(8)
    mask = np.array([True, False] * 4)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        np.add(a, b, where=mask)
    assert bc.flops_used > 0


def test_np_sin_where_kwarg_tracks():
    """Unary ufunc with ``where=`` mask. Mirrors the binary-``where=``
    test above, exercising ``_counted_unary``'s ``where`` strip path."""
    a = flops.random.randn(8)
    mask = np.array([True, False] * 4)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        np.sin(a, where=mask)
    assert bc.flops_used > 0


def test_np_add_dtype_kwarg_tracks():
    a = flops.random.randn(8).astype(np.float32)
    b = flops.random.randn(8).astype(np.float32)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        r = np.add(a, b, dtype=np.float64)
    assert bc.flops_used > 0
    assert r.dtype == np.float64


# ----- Mixed operands: numpy on left, flopscope on right -----

def test_mixed_numpy_left_operand_dispatches():
    """np.ndarray + FlopscopeArray must still dispatch through flopscope tracking
    (NEP 13: NumPy defers to subclasses' __array_ufunc__)."""
    a = np.ones(8)
    b = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        c = a + b
    assert bc.flops_used > 0
    assert isinstance(c, flops.ndarray)


def test_mixed_python_scalar_left_dispatches():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        c = 2.0 + a
    assert bc.flops_used > 0


# ----- In-place dunder identity preservation -----

def test_inplace_add_preserves_identity():
    """a += b must mutate a, not rebind it. Currently broken on
    post-PR-#51 main — fixed in Task 5."""
    a = flops.random.randn(8)
    old_id = id(a)
    with flops.BudgetContext(flop_budget=int(1e9)):
        a += 1
    assert id(a) == old_id


def test_inplace_add_scalar_on_symmetric_tensor_preserves_identity():
    """``A_sym += 1.0`` is symmetry-preserving (binary-with-scalar keeps
    every group). It must mutate ``A_sym`` in place AND keep the
    SymmetricTensor type — no spurious downgrade, no rebinding."""
    A = flops.symmetrize(
        flops.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    old_id = id(A)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        A += 1.0
    assert id(A) == old_id
    assert isinstance(A, flops.SymmetricTensor)
    assert bc.flops_used > 0


def test_inplace_add_refuses_when_symmetry_would_be_destroyed():
    """``A_sym += B_unsymmetric`` would silently overwrite ``A_sym``'s
    bytes with a non-symmetric result. The in-place dunder must refuse
    rather than corrupt the symmetry metadata."""
    A = flops.symmetrize(
        flops.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    B = flops.random.randn(4, 4)  # plain FlopscopeArray, no symmetry
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises(ValueError, match="destroy or weaken symmetry"):
            A += B


# ----- Unsupported ufunc methods fail loudly -----

def test_np_add_outer_fails_loudly():
    """ufunc.outer is not yet implemented; must raise rather than silently
    bypass tracking."""
    a = flops.random.randn(4)
    b = flops.random.randn(3)
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises((NotImplementedError, TypeError)):
            np.add.outer(a, b)


def test_np_add_reduceat_fails_loudly():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises((NotImplementedError, TypeError)):
            np.add.reduceat(a, [0, 4])


def test_np_add_at_fails_loudly():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises((NotImplementedError, TypeError)):
            np.add.at(a, [0, 1], 1.0)


# ----- Multi-output ufuncs fail loudly -----

def test_np_modf_fails_loudly():
    """``np.modf`` is a multi-output ufunc (nout=2). Until flopscope supports
    multi-output ``out=(out1, out2)`` properly, it must raise rather than
    silently bypass tracking on either return slot."""
    a = flops.random.randn(8) + 0.5
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises((NotImplementedError, TypeError)):
            np.modf(a)


def test_np_frexp_fails_loudly():
    a = flops.random.randn(8) + 1.5
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises((NotImplementedError, TypeError)):
            np.frexp(a)


def test_we_modf_rejects_out_kwarg():
    """The direct ``flops.modf(a, out=...)`` path goes through
    ``_counted_unary_multi`` (not ``__array_ufunc__``) because the
    operands are already flopscope. It must reject ``out=`` explicitly with
    NotImplementedError rather than silently letting NumPy write into
    flopscope-typed buffers (which would recurse through
    ``__array_function__`` and double-charge)."""
    a = flops.random.randn(8) + 0.5
    out1 = flops.empty_like(a)
    out2 = flops.empty_like(a)
    with flops.BudgetContext(flop_budget=int(1e9)):
        with pytest.raises(NotImplementedError, match="out"):
            flops.modf(a, out=(out1, out2))


# ----- Recursion guard for raw flopscope functions -----

def test_we_add_does_not_recurse_after_protocol_enabled():
    """``flops.add(FlopscopeArray, FlopscopeArray)`` must not enter an infinite loop
    via ``_np.add`` → ``__array_ufunc__`` → ``flops.add`` → ... after Task
    3 enables ``__array_ufunc__``. The strip in ``_counted_binary``
    breaks the cycle by viewing both operands as plain ndarray
    before calling ``_np.add``.
    """
    a = flops.random.randn(8)
    b = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        c = flops.add(a, b)
    assert bc.flops_used > 0
    assert c.shape == (8,)
    assert isinstance(c, flops.ndarray)


# ----- __array_function__: np.<func>(flopscope) -----

def test_np_sort_tracks_flops():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.sort(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        flops.sort(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_linalg_norm_tracks_flops():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.linalg.norm(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        flops.linalg.norm(a)
    assert b1.flops_used == b2.flops_used > 0


def test_np_where_tracks_flops():
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.where(a > 0, a, 0.0)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        flops.where(a > 0, a, 0.0)
    assert b1.flops_used == b2.flops_used > 0


def test_np_sum_routes_to_whest():
    """np.sum(flopscope) goes through __array_function__ (not __array_ufunc__)."""
    a = flops.random.randn(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as b1:
        np.sum(a)
    with flops.BudgetContext(flop_budget=int(1e9)) as b2:
        flops.sum(a)
    assert b1.flops_used == b2.flops_used > 0


# ----- Structural ops on SymmetricTensor: type follows surviving symmetry -----

def test_diagonal_of_3sym_downgrades_when_no_tensor_axis_symmetry_remains():
    """Diagonal of a (n,n,n) tensor with full S_3 symmetry along (0,1,2)
    collapses axes 0/1 (or any pair) into a single diagonal axis, which
    cannot retain a multi-axis permutation group → result must be
    ``FlopscopeArray``, not ``SymmetricTensor``."""
    n = 4
    A = flops.symmetrize(
        flops.random.randn(n, n, n),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1, 2)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        d = flops.diagonal(A)
    assert not isinstance(d, flops.SymmetricTensor)
    assert isinstance(d, flops.ndarray)


def test_transpose_of_symmetric_preserves_type():
    """Transposing a 2-axis symmetric matrix preserves the symmetry."""
    A = flops.symmetrize(
        flops.random.randn(4, 4),
        symmetry=flops.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    with flops.BudgetContext(flop_budget=int(1e9)):
        AT = A.T
    assert isinstance(AT, flops.SymmetricTensor)


# ----- Helper unit tests -----

def test_to_base_ndarray_strips_whest_subclass():
    from flopscope._ndarray import _to_base_ndarray, FlopscopeArray
    a = flops.random.randn(4)
    base = _to_base_ndarray(a)
    assert type(base) is np.ndarray
    assert not isinstance(base, FlopscopeArray)
    base[0] = 99.0
    assert a[0] == 99.0  # zero-copy view


def test_to_base_ndarray_preserves_python_scalar():
    from flopscope._ndarray import _to_base_ndarray
    assert _to_base_ndarray(2.0) == 2.0
    assert isinstance(_to_base_ndarray(2.0), float)


def test_to_base_ndarray_tree_strips_in_tuple():
    from flopscope._ndarray import _to_base_ndarray_tree
    a = flops.random.randn(4)
    b = flops.random.randn(4)
    out = _to_base_ndarray_tree((a, b))
    assert all(type(x) is np.ndarray for x in out)


def test_to_base_ndarray_tree_strips_in_list():
    from flopscope._ndarray import _to_base_ndarray_tree
    a = flops.random.randn(4)
    b = flops.random.randn(4)
    out = _to_base_ndarray_tree([a, b])
    assert all(type(x) is np.ndarray for x in out)


def test_to_base_ndarray_tree_preserves_scalars():
    from flopscope._ndarray import _to_base_ndarray_tree
    out = _to_base_ndarray_tree((1.0, 2, "x"))
    assert out == (1.0, 2, "x")


def test_to_base_ndarray_tree_recurses_into_nested():
    from flopscope._ndarray import _to_base_ndarray_tree
    a = flops.random.randn(4)
    out = _to_base_ndarray_tree([(a, 1.0), [a]])
    assert type(out[0][0]) is np.ndarray
    assert type(out[1][0]) is np.ndarray


# ----- _PASSTHROUGH lock-in -----

def test_np_type_query_passthrough_does_not_charge_flops():
    """``np.result_type``, ``np.can_cast``, ``np.min_scalar_type``, etc.
    are zero-FLOP type-query functions that must bypass flopscope's
    FLOP-tracking dispatch. They are added to ``_PASSTHROUGH_NAMES`` for
    this reason."""
    a = flops.empty(8)
    with flops.BudgetContext(flop_budget=int(1e9)) as bc:
        assert np.result_type(a) == np.float64
        assert np.can_cast(a, np.float64)
        assert np.min_scalar_type(a) is not None
    assert bc.flops_used == 0


# ----- Cache-verification test (Stage 2 helper added in Step 2.2) -----

def test_signature_kwargs_accepted_is_cached():
    """The signature lookup must be cached — it sits on the per-ufunc
    hot path. PR #51 memoized similar helpers; we do the same here."""
    from flopscope._ndarray import _signature_kwargs_accepted
    # Same callable should return the same frozenset object (cached).
    a = _signature_kwargs_accepted(np.add)
    b = _signature_kwargs_accepted(np.add)
    assert a is b, "_signature_kwargs_accepted is not cached"


# ----- Performance regression guards (against PR #51 hot paths) -----
# Reference: https://github.com/AIcrowd/flopscope/pull/51#issuecomment-4340098399
# These are SEED tests demonstrating the pattern. Add stage-specific
# perf tests in later stages when implementer reasoning identifies
# hot paths their changes touch.


def test_perf_warm_rank8_scalar_comparison_is_fast():
    """Warm rank-8 scalar comparison ``flops.full((2,)*8, 1) == 1`` must
    finish well under 100ms.

    PR #51 fixed this from ~930ms warm to ~0.04ms warm via:
    (a) ``SymmetryGroup.__eq__`` identity short-circuit,
    (b) per-instance ``_canonical_axis_action`` cache,
    (c) ``@functools.cache`` on ``unique_elements_for_shape``.
    Generous bound (2500× margin over the post-fix figure) catches
    order-of-magnitude regressions without flaking on machine variance.
    """
    import time
    a = flops.full((2,) * 8, 1)  # rank-8, 256 elements, full S_8 symmetric group
    # Warm-up: prime the per-instance _canonical_axis_action cache and the
    # @functools.cache on unique_elements_for_shape.
    with flops.BudgetContext(flop_budget=int(1e12)):
        _ = (a == 1)
    # Measure the warm path.
    with flops.BudgetContext(flop_budget=int(1e12)):
        t0 = time.perf_counter()
        _ = (a == 1)
        elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, (
        f"warm rank-8 scalar comparison took {elapsed*1000:.1f}ms; "
        f"PR #51 fixed this to ~0.04ms. Are we re-introducing O(|G|) "
        f"work in __eq__ / _canonical_axis_action / "
        f"unique_elements_for_shape, or making OWNDATA-preserving "
        f"copies in __array_ufunc__?"
    )


def test_perf_array_ufunc_dispatch_does_not_copy():
    """100 invocations of ``np.add(flopscope, flopscope)`` on 1024-element
    arrays must complete well under 1 second.

    PR #51 dropped OWNDATA-preserving copies in ``__array_wrap__``,
    ``_asflopscope``, ``_asplainflopscope`` to keep this path cheap. Our
    ``__array_ufunc__`` handler must not re-introduce them by, e.g.,
    calling ``np.array(x, copy=True)`` or wrapping each result in a
    redundant view-cast that triggers a finalize chain.
    """
    import time
    a = flops.random.randn(1024)
    b = flops.random.randn(1024)
    # Warm-up.
    with flops.BudgetContext(flop_budget=int(1e12)):
        _ = np.add(a, b)
    # Measure 100 calls.
    with flops.BudgetContext(flop_budget=int(1e15)):
        t0 = time.perf_counter()
        for _ in range(100):
            _ = np.add(a, b)
        elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, (
        f"100x np.add(flopscope, flopscope) took {elapsed:.3f}s; "
        f"are we re-introducing per-call copies or O(|G|) work in "
        f"__array_ufunc__ / _filter_to_np_signature?"
    )
