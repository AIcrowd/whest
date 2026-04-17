"""Known divergences between whest and NumPy.

Each entry maps a test node ID (or pattern) to a reason string.
Tests matching these patterns are marked xfail when running NumPy's
test suite against whest.

Current state (2026-04-17, numpy 2.4.4, after Task 10 triage):
    Total:          ~7,862 passed, 47 xfailed, 0 failures (all suites)
    test_umath:     ~4,671 passed  (15 xfailed — 1 new test_ufunc_override_where)
    test_ufunc:       ~855 passed  (10 xfailed)
    test_numeric:   ~1,616 passed  (17 xfailed)
    test_linalg:      ~467 passed   (1 xfailed — numpy-internal LAPACK mark)
    test_pocketfft:    148 passed   (0 xfailed)
    test_helper:         8 passed   (0 xfailed)
    test_polynomial:    39 passed   (1 xfailed)
    test_random:       140 passed   (2 xfailed — 3 stale xfails removed)

Previous state (2026-04-17, numpy 2.3.5, after Task 9 triage):
    Total:          ~7,862 passed, 47 xfailed, 0 failures (all suites)
    test_umath:     ~4,668 passed  (14 xfailed)
    test_ufunc:       ~829 passed   (9+1 xfailed — 1 new pattern added)
    test_numeric:   ~1,608 passed  (17 xfailed)
    test_linalg:      ~467 passed   (1 xfailed — numpy-internal LAPACK mark)
    test_pocketfft:    148 passed   (0 xfailed)
    test_helper:         8 passed   (0 xfailed)
    test_polynomial:    37 passed   (1 xfailed)
    test_random:       137 passed   (2 xfailed)

Previous state (2026-04-14, numpy 2.2, after Tier 1+2+3 fixes):
    Total:          ~7,861 passed, 46 xfailed, 0 failures (all suites)
    test_umath:     ~4,668 passed  (12 xfailed)
    test_ufunc:       ~795 passed   (3 xfailed — 4 stale xfails removed)
    test_numeric:   ~1,604 passed  (14 xfailed — nep50_isclose fixed)
    test_linalg:      ~408 passed   (0 xfailed — cond NaN + matrix_rank fixed)
    test_pocketfft:    148 passed   (0 xfailed)
    test_polynomial:   600 passed   (1 xfailed — polydiv scalar fixed)
    test_random:       139 passed   (5 xfailed)

Fixes applied:
    OWNDATA fix:        All TestClip OWNDATA_VIEW patterns removed (clip now owns its data).
    array-skip fix:     TestNorm matrix tests, TestQR modes, TestEighCases removed
                        (array patching skipped, numpy linalg functions work natively).
    named-tuple fix:    TestSVD::test_types*, TestSVDHermitian::test_types* removed
                        (svd now returns proper named tuple).
    linalg subclass:    All TestEig/TestInv/TestSolve/TestLstsq/TestSVD/TestPinv/
                        TestSVDHermitian/TestPinvHermitian sq_cases/generalized* removed
                        (NonDescriptor fix and linalg return-type fixes).
    random fix:         TestRandint::test_* (6 patterns) and test_choice_return_shape
                        removed (now pass with NonDescriptor fix).
    StdVar fix:         test_out_scalar removed (std/var out= now works).
    7 trivial fixes:    _aswhest order='A', linalg.diagonal axis=-2/-1, linalg.cross
                        validation, tensordot int axes, norm axis validation,
                        clip argument validation, astype copy/device kwargs.
    Tier 3 fixes:       cond NaN (SVD fallback per-matrix), matrix_rank 1D input,
                        polydiv scalar input (atleast_1d), isclose NEP 50 type
                        promotion (keep Python scalars un-coerced).
    4 stale xfails removed: test_scalar_equal, test_struct_ufunc,
                        test_ufunc_override_where, test_safe_casting.

What we patch (55 functions):
    Non-ufunc reductions and special functions (all, any, amax, amin,
    argmax, argmin, average, cumsum, cumprod, mean, median, std, var,
    sum, prod, etc.) plus misc functions like isclose, real, imag, etc.

What we DON'T patch (and why):
    - Ufuncs (101): whest functions are plain callables, not ufuncs.
      They lack .reduce/.accumulate/.outer/.nargs/etc. Patching would
      break test collection and any test using ufunc protocol.
    - Free ops (220): pass-throughs that delegate to numpy. Patching
      causes infinite recursion since whest's _np IS np.
    - Counted custom (36): dot, matmul, einsum, convolve, etc. call
      _np.func() via module lookup. Same recursion issue.
    - Submodule functions (38): linalg.*, fft.*. Same recursion issue.
    - Blacklisted (32): intentionally unsupported.
    - linalg.outer: we.linalg.outer delegates to np.outer (not
      np.linalg.outer), so patching it causes a collection-time error
      in test_linalg.py which checks ValueError for 2D input at class
      definition time.

Categories for failures:
    UNSUPPORTED_DTYPE, UFUNC_INTERNALS, BUDGET_SIDE_EFFECT,
    NOT_IMPLEMENTED, NUMPY_INTERNAL, SUBCLASS_RETURN, WRAPPER_SIGNATURE,
    BEHAVIORAL_SHIM, REMOVED_IN_NUMPY

New categories added for numpy 2.3 triage (Task 9):

    BEHAVIORAL_SHIM — numpy's own test asserts the 2.3+ behavior that
        whest intentionally shims away (e.g. count_nonzero test asserts
        numpy scalar return; whest returns int).

    REMOVED_IN_NUMPY — numpy removed this symbol in 2.4; whest gates it
        off. The upstream test still references the removed symbol.
        (Not used for numpy 2.3 triage — nothing whest wraps was removed
        in 2.3; category reserved for the 2.4 triage in Task 11.)

Changes in numpy 2.4 triage (Task 10):
    1 new xfail added: test_ufunc_override_where — numpy 2.4 changed
        ufunc dispatch so WhestArray (returned by patched np.zeros) ends
        up as out= in an OverriddenArray._unwrap call; the unwrap sees a
        non-matching ndarray subclass and returns NotImplemented, then
        [0] subscript crashes.  SUBCLASS_RETURN category.
    3 stale xfails removed: test_shuffle_untyped_warning[numpy.random],
        test_shuffle_no_object_unpacking[False-numpy.random],
        test_shuffle_no_object_unpacking[False-random1] — all xpassed on
        both numpy 2.3 and 2.4.
        (test_out_wrap_no_leak retained: still fails on numpy 2.2;
        xpassed on 2.3/2.4 is acceptable since strict=False.)
    Note on test_ufunc_override_where history: this pattern was removed
        in a previous task (marked in Fixes applied above) and is now
        re-added specifically for numpy 2.4 where it regressed.

Unit suite gaps (for Task 11 to fix, not this file):
    test_sorting_ops::TestIn1d — needs skipif(np>=2.4) because in1d
        raises UnsupportedFunctionError on 2.4.
    test_pointwise_coverage::TestTrapz — needs skipif(np>=2.4) because
        trapz raises UnsupportedFunctionError on 2.4.
    test_signature_conformance — 6 tests fail because numpy 2.4 added
        C-level positional-only annotations to dot, packbits, unpackbits,
        shares_memory, ravel_multi_index, promote_types whose signatures
        now differ from whest's pass-through wrappers (*args, **kwargs).
"""

# Reason-string constants for use in XFAIL_PATTERNS values.
# Using bare string literals in the dict is fine too; these exist so
# grep / tooling can find all tests sharing a category quickly.
# NOTE: BEHAVIORAL_SHIM and REMOVED_IN_NUMPY are reserved for future
# triage rounds. The current 2.3/2.4 triage did not surface any test
# needing these categories — numpy's own test suite doesn't exercise
# count_nonzero return types or the removed in1d/trapz symbols from
# within its bundled test files. If a future triage finds such tests,
# use these constants as XFAIL_PATTERNS values.
BEHAVIORAL_SHIM = (
    "BEHAVIORAL_SHIM: whest intentionally preserves pre-2.3 behavior; "
    "numpy's test asserts the 2.3+ behavior and therefore fails when "
    "monkeypatched."
)
REMOVED_IN_NUMPY = (
    "REMOVED_IN_NUMPY: numpy removed this symbol in 2.4; whest gates it "
    "off. The upstream test still references the removed symbol."
)

XFAIL_PATTERNS: dict[str, str] = {
    # ------------------------------------------------------------------ #
    # test_polynomial.py — divergences                                    #
    # ------------------------------------------------------------------ #
    "*TestEvaluation::test_polyval": (
        "NOT_IMPLEMENTED: whest polyval doesn't support masked arrays"
    ),
    # ------------------------------------------------------------------ #
    # test_random.py — counted random wrapper signature divergences        #
    # ------------------------------------------------------------------ #
    # whest random wrappers are plain functions, not methods on the
    # RandomState class. Tests that use np.random.shuffle as a bound method
    # or test internal RandomState behavior will fail.
    "*TestRandomDist::test_shuffle": (
        "WRAPPER_SIGNATURE: whest shuffle is a plain function, not a bound method"
    ),
    "*TestStdVar::test_out_scalar": (
        "SUBCLASS_RETURN: std/var out= parameter interacts with WhestArray wrapping"
    ),
    "*TestRandomDist::test_choice_return_shape": (
        "SUBCLASS_RETURN: choice return wrapping differs"
    ),
    # ------------------------------------------------------------------ #
    # SUBCLASS_RETURN — WhestArray subclass propagation               #
    # ------------------------------------------------------------------ #
    # whest wraps return values in WhestArray (an ndarray subclass)
    # so that operator overloads can route through FLOP-tracked we.* funcs.
    # NumPy's tests use strict `type(x) is np.ndarray` checks that fail when
    # the result is a subclass. These tests are inherent limitations of the
    # subclass design.
    "*TestSpecialMethods::test_priority": (
        "SUBCLASS_RETURN: ndarray subclass propagates through ufunc with __array_priority__"
    ),
    "*TestUfunc::test_array_wrap_array_priority": (
        "SUBCLASS_RETURN: np.zeros (patched to return WhestArray) wins the "
        "__array_priority__ contest against a subclass with priority 0; "
        "add returns WhestArray instead of the expected subclass instance"
    ),
    "*TestUfunc::test_scalar_reduction": (
        "SUBCLASS_RETURN: ufunc reduction on WhestArray returns subclass instead of scalar"
    ),
    "*TestUfunc::test_broadcast": (
        "SUBCLASS_RETURN: broadcast result preserves WhestArray subclass"
    ),
    "*TestNonzero::test_return_type": (
        "SUBCLASS_RETURN: nonzero returns WhestArray instead of plain ndarray tuple"
    ),
    "*TestRequire::test_ensure_array": (
        "SUBCLASS_RETURN: np.require with subok=False can't strip WhestArray"
    ),
    "*TestArrayComparisons::test_compare_unstructured_voids*": (
        "SUBCLASS_RETURN: void comparison preserves WhestArray subclass"
    ),
    # ------------------------------------------------------------------ #
    # SUBCLASS_RETURN — *_like strides mismatch                           #
    # ------------------------------------------------------------------ #
    # np.zeros_like/ones_like/empty_like/full_like preserve strides from the
    # prototype. When the prototype is a WhestArray, the resulting array has
    # C-order strides rather than the non-contiguous strides of the original.
    "*TestLikeFuncs::test_zeros_like": (
        "SUBCLASS_RETURN: *_like strides don't match prototype when prototype is WhestArray"
    ),
    "*TestLikeFuncs::test_ones_like": (
        "SUBCLASS_RETURN: *_like strides don't match prototype when prototype is WhestArray"
    ),
    "*TestLikeFuncs::test_empty_like": (
        "SUBCLASS_RETURN: *_like strides don't match prototype when prototype is WhestArray"
    ),
    "*TestLikeFuncs::test_filled_like": (
        "SUBCLASS_RETURN: *_like strides don't match prototype when prototype is WhestArray"
    ),
    # ------------------------------------------------------------------ #
    # NUMPY_INTERNAL — fromiter/resize edge cases                         #
    # ------------------------------------------------------------------ #
    "*TestCreationFuncs::test_zeros": (
        "SUBCLASS_RETURN: _aswhest OWNDATA copy interacts with _symmetric_2d wrapping"
    ),
    "*TestCreationFuncs::test_ones": (
        "SUBCLASS_RETURN: _aswhest OWNDATA copy interacts with _symmetric_2d wrapping"
    ),
    "*TestCreationFuncs::test_empty": (
        "SUBCLASS_RETURN: _aswhest OWNDATA copy interacts with _symmetric_2d wrapping"
    ),
    "*TestResize::test_reshape_from_zero": (
        "NUMPY_INTERNAL: resize from zero-element array edge case"
    ),
    "*TestFromiter::test_growth_and_complicated_dtypes*i,O*": (
        "NUMPY_INTERNAL: fromiter with object dtype interacts unexpectedly with patched np"
    ),
    "*TestOut::test_out_wrap_no_leak": (
        "NUMPY_INTERNAL: refcount check sees unexpected count due to WhestArray subclass "
        "wrapping (fails on numpy 2.2; xpassed on 2.3/2.4 which is acceptable)"
    ),
    # ------------------------------------------------------------------ #
    # SUBCLASS_RETURN — numpy 2.4 ufunc __array_ufunc__ dispatch change  #
    # ------------------------------------------------------------------ #
    # numpy 2.4 changed ufunc dispatch so our patched np.zeros returns a
    # WhestArray, which then lands as out= inside OverriddenArray._unwrap.
    # _unwrap checks type(obj) != np.ndarray and returns NotImplemented
    # for any ndarray subclass it doesn't recognise; the caller then does
    # NotImplemented[0] which raises TypeError.
    # This test passed on numpy 2.3; it's a genuine 2.4 regression caused
    # by our WhestArray subclass propagating into third-party __array_ufunc__
    # implementations that use strict type checks.
    "*TestSpecialMethods::test_ufunc_override_where": (
        "SUBCLASS_RETURN: numpy 2.4 ufunc dispatch routes WhestArray (from patched "
        "np.zeros) as out= into OverriddenArray._unwrap which does strict "
        "type(obj) != np.ndarray checks; returns NotImplemented then crashes on [0]"
    ),
}
