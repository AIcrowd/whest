"""Known divergences between whest and NumPy.

Each entry maps a test node ID (or pattern) to a reason string.
Tests matching these patterns are marked xfail when running NumPy's
test suite against whest.

Current state (2026-04-14):
    Total:          7,760 passed, 28 failed, 63 xfailed, 0 xpassed
    test_umath:     ~4,668 passed  (11 xfailed — removed test_ufunc_override_where)
    test_ufunc:       ~795 passed   (4 xfailed — removed scalar_equal, struct_ufunc, safe_casting)
    test_numeric:   ~1,567 passed   (4 xfailed — removed all OWNDATA_VIEW TestClip patterns)
    test_linalg:      ~395 passed  (30 xfailed — removed matrix_*, QR modes, EighCases, SVD types)
    test_pocketfft:    148 passed   (0 xfailed)
    test_polynomial:   600 passed   (2 xfailed)
    test_random:       139 passed   (5 xfailed — removed test_shuffle_no_object_unpacking[False*])

Fixes applied:
    OWNDATA fix:        All TestClip OWNDATA_VIEW patterns removed (clip now owns its data).
    array-skip fix:     TestNorm matrix tests, TestQR modes, TestEighCases removed
                        (array patching skipped, numpy linalg functions work natively).
    named-tuple fix:    TestSVD::test_types*, TestSVDHermitian::test_types* removed
                        (svd now returns proper named tuple).
    subclass fixes:     test_ufunc_override_where, test_scalar_equal, test_struct_ufunc,
                        test_safe_casting, test_non_array_input removed.
    random fix:         test_shuffle_no_object_unpacking[False*] removed.

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
    # test_numeric.py — new divergences from expanded numpy patching      #
    # ------------------------------------------------------------------ #
    "*TestTensordot::test_zero_dimension": (
        "NOT_IMPLEMENTED: whest clip doesn't handle 0-d arrays identically"
    ),
    "*TestClip::test_clip_min_max_args": (
        "NOT_IMPLEMENTED: whest clip doesn't enforce strict a_min/a_max/min/max argument validation"
    ),
    "*TestAsType::test_astype": (
        "NOT_IMPLEMENTED: whest free ops astype behavior differs"
    ),
    # ------------------------------------------------------------------ #
    # test_linalg.py — remaining failures after ndim guards/batch fixes   #
    # ------------------------------------------------------------------ #
    # Most batch/0-size/generalized cases now pass. Remaining failures
    # are: cross/diagonal edge cases, cond NaN, eig/inv/lstsq/solve
    # sq_cases precision differences, matrix_rank, bad_args validation,
    # pinv hermitian/generalized cases.
    "*test_linalg*::test_cross": (
        "NOT_IMPLEMENTED: whest cross doesn't raise ValueError for 2D arrays"
    ),
    "*test_linalg*::test_diagonal": (
        "NOT_IMPLEMENTED: whest linalg.diagonal behavior differs"
    ),
    "*test_linalg*::TestCond::test_nan": (
        "NOT_IMPLEMENTED: whest cond doesn't handle NaN inputs correctly"
    ),
    "*test_linalg*::TestEig::test_sq_cases": (
        "NOT_IMPLEMENTED: whest eig sq_cases differ (precision/dtype)"
    ),
    "*test_linalg*::TestEig::test_generalized*": (
        "SUBCLASS_RETURN: consistent_subclass check fails — returns WhestArray, not input subclass"
    ),
    "*test_linalg*::TestInv::test_sq_cases": (
        "NOT_IMPLEMENTED: whest inv sq_cases differ (precision/dtype)"
    ),
    "*test_linalg*::TestInv::test_generalized*": (
        "SUBCLASS_RETURN: consistent_subclass check fails — returns WhestArray, not input subclass"
    ),
    "*test_linalg*::TestLstsq::test_sq_cases": (
        "NOT_IMPLEMENTED: whest lstsq sq_cases differ"
    ),
    "*test_linalg*::TestLstsq::test_nonsq_cases": (
        "SUBCLASS_RETURN: consistent_subclass check fails — returns WhestArray, not input subclass"
    ),
    "*test_linalg*::TestPinv::test_nonsq_cases": (
        "SUBCLASS_RETURN: consistent_subclass check fails — returns WhestArray, not input subclass"
    ),
    "*test_linalg*::TestMatrixRank::test_matrix_rank": (
        "NOT_IMPLEMENTED: whest matrix_rank behavior differs"
    ),
    "*test_linalg*::TestNormDouble::test_bad_args": (
        "NOT_IMPLEMENTED: whest norm doesn't validate ord argument like np.linalg.norm"
    ),
    "*test_linalg*::TestNormInt64::test_bad_args": (
        "NOT_IMPLEMENTED: whest norm doesn't validate ord argument like np.linalg.norm"
    ),
    "*test_linalg*::TestNormSingle::test_bad_args": (
        "NOT_IMPLEMENTED: whest norm doesn't validate ord argument like np.linalg.norm"
    ),
    "*test_linalg*::TestPinvHermitian::test_herm_cases": (
        "NOT_IMPLEMENTED: whest pinv hermitian cases differ"
    ),
    "*test_linalg*::TestSolve::test_sq_cases": (
        "NOT_IMPLEMENTED: whest solve sq_cases differ (precision/dtype)"
    ),
    "*test_linalg*::TestSolve::test_generalized*": (
        "SUBCLASS_RETURN: consistent_subclass check fails — returns WhestArray, not input subclass"
    ),
    "*test_linalg*::TestSVD::test_sq_cases": (
        "NOT_IMPLEMENTED: whest svd sq_cases differ"
    ),
    "*test_linalg*::TestSVD::test_generalized*": (
        "SUBCLASS_RETURN: consistent_subclass check fails — returns WhestArray, not input subclass"
    ),
    "*test_linalg*::TestSVDHermitian::test_herm_cases": (
        "NOT_IMPLEMENTED: whest svd hermitian cases differ"
    ),
    "*test_linalg*::TestSVDHermitian::test_generalized*": (
        "SUBCLASS_RETURN: consistent_subclass check fails — returns WhestArray, not input subclass"
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
    # RandomState class. Tests that use np.random.randint as a bound method
    # (passing self) or test internal RandomState behavior will fail.
    "*TestRandint::test_in_bounds_fuzz": (
        "WRAPPER_SIGNATURE: whest randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_rng_zero_and_extremes": (
        "WRAPPER_SIGNATURE: whest randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_respect_dtype_singleton": (
        "WRAPPER_SIGNATURE: whest randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_bounds_checking": (
        "WRAPPER_SIGNATURE: whest randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_repeatability": (
        "WRAPPER_SIGNATURE: whest randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_full_range": (
        "WRAPPER_SIGNATURE: whest randint is a plain function, not a bound method"
    ),
    "TestRandomDist::test_shuffle_untyped_warning[numpy.random]": (
        "WRAPPER_SIGNATURE: warning filename points to whest wrapper, not test file"
    ),
    "*TestRandomDist::test_shuffle": (
        "WRAPPER_SIGNATURE: whest shuffle is a plain function, not a bound method"
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
    "*TestPinv::test_sq_cases": (
        "SUBCLASS_RETURN: pinv result type assertion sees WhestArray subclass"
    ),
    "*TestPinv::test_generalized*": (
        "SUBCLASS_RETURN: consistent_subclass check fails — pinv returns WhestArray, not input subclass"
    ),
    "*TestPinvHermitian::test_generalized*": (
        "SUBCLASS_RETURN: consistent_subclass check fails — pinv returns WhestArray, not input subclass"
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
    "*TestCreationFuncs::test_full": (
        "SUBCLASS_RETURN: _aswhest OWNDATA copy interacts with _symmetric_2d wrapping"
    ),
    "*TestLikeFuncs::test_empty_like": (
        "SUBCLASS_RETURN: _aswhest OWNDATA copy interacts with _symmetric_2d wrapping"
    ),
    "*TestLikeFuncs::test_filled_like": (
        "SUBCLASS_RETURN: _aswhest OWNDATA copy interacts with _symmetric_2d wrapping"
    ),
    "*TestLikeFuncs::test_ones_like": (
        "SUBCLASS_RETURN: _aswhest OWNDATA copy interacts with _symmetric_2d wrapping"
    ),
    "*TestLikeFuncs::test_zeros_like": (
        "SUBCLASS_RETURN: _aswhest OWNDATA copy interacts with _symmetric_2d wrapping"
    ),
    "*TestStdVar::test_out_scalar": (
        "SUBCLASS_RETURN: std/var out= parameter interacts with WhestArray wrapping"
    ),
    "*TestRandomDist::test_choice_return_shape": (
        "SUBCLASS_RETURN: choice return wrapping differs"
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
