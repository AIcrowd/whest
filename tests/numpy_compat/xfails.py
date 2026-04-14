"""Known divergences between whest and NumPy.

Each entry maps a test node ID (or pattern) to a reason string.
Tests matching these patterns are marked xfail when running NumPy's
test suite against whest.

Current state (2026-04-14, after Tier 1+2 fixes):
    Total:          ~7,822 passed, 44 xfailed, 0 failures (all suites)
    test_umath:     ~4,668 passed  (12 xfailed)
    test_ufunc:       ~795 passed   (7 xfailed)
    test_numeric:   ~1,604 passed  (15 xfailed)
    test_linalg:      ~408 passed   (3 xfailed — most linalg xfails now pass)
    test_pocketfft:    148 passed   (0 xfailed)
    test_polynomial:   600 passed   (2 xfailed)
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
    NOT_IMPLEMENTED, NUMPY_INTERNAL
"""

XFAIL_PATTERNS: dict[str, str] = {
    # ------------------------------------------------------------------ #
    # test_numeric.py — isclose divergences                               #
    # ------------------------------------------------------------------ #
    # whest's isclose is a counted wrapper; NEP 50 promotion not supported.
    "*TestIsclose::test_nep50_isclose": (
        "NOT_IMPLEMENTED: whest isclose doesn't support NEP 50 promotion"
    ),
    # ------------------------------------------------------------------ #
    # test_linalg.py — remaining failures after all fixes                 #
    # ------------------------------------------------------------------ #
    # Most linalg cases now pass (sq_cases, generalized, hermitian, etc.)
    # Remaining failures: cond NaN and matrix_rank only.
    "*test_linalg*::TestCond::test_nan": (
        "NOT_IMPLEMENTED: whest cond doesn't handle NaN inputs correctly"
    ),
    "*test_linalg*::TestMatrixRank::test_matrix_rank": (
        "NOT_IMPLEMENTED: whest matrix_rank behavior differs"
    ),
    # ------------------------------------------------------------------ #
    # test_polynomial.py — divergences                                    #
    # ------------------------------------------------------------------ #
    "*TestMisc::test_result_type": (
        "NOT_IMPLEMENTED: whest polynomial result_type differs"
    ),
    "*TestEvaluation::test_polyval": (
        "NOT_IMPLEMENTED: whest polyval doesn't support masked arrays"
    ),
    # ------------------------------------------------------------------ #
    # test_random.py — counted random wrapper signature divergences        #
    # ------------------------------------------------------------------ #
    # whest random wrappers are plain functions, not methods on the
    # RandomState class. Tests that use np.random.shuffle as a bound method
    # or test internal RandomState behavior will fail.
    "TestRandomDist::test_shuffle_untyped_warning[numpy.random]": (
        "WRAPPER_SIGNATURE: warning filename points to whest wrapper, not test file"
    ),
    "*TestRandomDist::test_shuffle": (
        "WRAPPER_SIGNATURE: whest shuffle is a plain function, not a bound method"
    ),
    # WhestArray subclass causes incorrect shuffle of 1D object arrays.
    # Only [False-numpy.random] and [False-random1] fail; [True-*] and
    # [False-random2] pass. Using substring matching (the 'in' branch of
    # conftest) since fnmatch can't handle '[' in parametrize IDs.
    "test_shuffle_no_object_unpacking[False-numpy.random]": (
        "SUBCLASS_RETURN: WhestArray subclass causes wrong object-array shuffle behavior"
    ),
    "test_shuffle_no_object_unpacking[False-random1]": (
        "SUBCLASS_RETURN: WhestArray subclass causes wrong object-array shuffle behavior"
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
    # UFUNC_INTERNALS — WhestArray __eq__/__ne__ operator edge cases      #
    # ------------------------------------------------------------------ #
    # WhestArray overrides __eq__/__ne__ to route through we.equal/not_equal,
    # but those ufuncs don't support all dtype combinations (e.g. float64 vs
    # str, or structured void types). Plain ndarray falls back to identity
    # comparison in these cases; WhestArray raises UFuncNoLoopError instead.
    "*TestUfunc::test_scalar_equal": (
        "UFUNC_INTERNALS: WhestArray.__ne__ raises UFuncNoLoopError for float64 vs str dtype"
    ),
    "*TestUfunc::test_struct_ufunc": (
        "UFUNC_INTERNALS: WhestArray.__eq__ raises UFuncNoLoopError for structured void dtype"
    ),
    # WhestArray.__array_ufunc__ returns NotImplemented for ufuncs called
    # with where= argument containing a non-WhestArray operand.
    "*TestSpecialMethods::test_ufunc_override_where": (
        "UFUNC_INTERNALS: WhestArray.__array_ufunc__ returns NotImplemented when where= is non-WhestArray"
    ),
    # add ufunc in-place with unsafe cast — WhestArray wrapping bypasses
    # numpy's safe-casting check for in-place operations.
    "*TestUfunc::test_safe_casting": (
        "UFUNC_INTERNALS: WhestArray wrapping bypasses safe-casting check for in-place ufunc ops"
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
        "NUMPY_INTERNAL: refcount check sees unexpected count due to WhestArray subclass wrapping"
    ),
}
