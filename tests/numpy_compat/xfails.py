"""Known divergences between whest and NumPy.

Each entry maps a test node ID (or pattern) to a reason string.
Tests matching these patterns are marked xfail when running NumPy's
test suite against whest.

Current state (2026-04-14):
    Total:          7,788 passed, 0 failed, 121 xfailed, 2 xpassed
    test_umath:     4,668 passed  (14 xfailed)
    test_ufunc:       795 passed   (7 xfailed)
    test_numeric:   1,567 passed   (4 xfailed — was 20, fixed 16; refined 3 patterns)
    test_linalg:      395 passed  (42 xfailed — was 255, fixed 213 after ndim guards/batch)
    test_pocketfft:   148 passed   (0 xfailed — was 34, fixed all)
    test_polynomial:  600 passed   (2 xfailed)
    test_random:      139 passed   (6 xfailed — was 8, fixed shuffle + unpacking)

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
    # are: matrix norm (no ord arg), raw QR mode, sq_cases precision
    # differences, SVD full mode, and cross/diagonal edge cases.
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
    "*test_linalg*::TestEighCases::*": (
        "NOT_IMPLEMENTED: whest eigh doesn't support generalized/stacked cases"
    ),
    "*test_linalg*::TestInv::test_sq_cases": (
        "NOT_IMPLEMENTED: whest inv sq_cases differ (precision/dtype)"
    ),
    "*test_linalg*::TestLstsq::test_sq_cases": (
        "NOT_IMPLEMENTED: whest lstsq sq_cases differ"
    ),
    "*test_linalg*::TestMatrixRank::test_matrix_rank": (
        "NOT_IMPLEMENTED: whest matrix_rank behavior differs"
    ),
    "*test_linalg*::TestNormDouble::test_bad_args": (
        "NOT_IMPLEMENTED: whest norm doesn't validate ord argument like np.linalg.norm"
    ),
    "*test_linalg*::TestNormDouble::test_matrix_*": (
        "NOT_IMPLEMENTED: whest norm doesn't support matrix norm (ord arg)"
    ),
    "*test_linalg*::TestNormInt64::test_bad_args": (
        "NOT_IMPLEMENTED: whest norm doesn't validate ord argument like np.linalg.norm"
    ),
    "*test_linalg*::TestNormInt64::test_matrix_*": (
        "NOT_IMPLEMENTED: whest norm doesn't support matrix norm (ord arg)"
    ),
    "*test_linalg*::TestNormSingle::test_bad_args": (
        "NOT_IMPLEMENTED: whest norm doesn't validate ord argument like np.linalg.norm"
    ),
    "*test_linalg*::TestNormSingle::test_matrix_*": (
        "NOT_IMPLEMENTED: whest norm doesn't support matrix norm (ord arg)"
    ),
    "*test_linalg*::TestPinvHermitian::test_herm_cases": (
        "NOT_IMPLEMENTED: whest pinv hermitian cases differ"
    ),
    "*test_linalg*::TestQR::test_mode_all_but_economic": (
        "NOT_IMPLEMENTED: whest qr doesn't support non-economic modes"
    ),
    "*test_linalg*::TestQR::test_mode_raw": (
        "NOT_IMPLEMENTED: whest qr doesn't support raw mode"
    ),
    "*test_linalg*::TestSolve::test_sq_cases": (
        "NOT_IMPLEMENTED: whest solve sq_cases differ (precision/dtype)"
    ),
    "*test_linalg*::TestSVD::test_sq_cases": (
        "NOT_IMPLEMENTED: whest svd sq_cases differ"
    ),
    "*test_linalg*::TestSVD::test_types*": (
        "NOT_IMPLEMENTED: whest svd type cases differ"
    ),
    "*test_linalg*::TestSVDHermitian::test_herm_cases": (
        "NOT_IMPLEMENTED: whest svd hermitian cases differ"
    ),
    "*test_linalg*::TestSVDHermitian::test_types*": (
        "NOT_IMPLEMENTED: whest svd hermitian type cases differ"
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
    "*TestRandomDist::test_shuffle_no_object_unpacking[False*": (
        "SUBCLASS_RETURN: WhestArray subclass object unpacking differs from ndarray"
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
    "*TestSpecialMethods::test_ufunc_override_where": (
        "SUBCLASS_RETURN: WhestArray __array_ufunc__ interaction with other override classes"
    ),
    "*TestUfunc::test_scalar_reduction": (
        "SUBCLASS_RETURN: ufunc reduction on WhestArray returns subclass instead of scalar"
    ),
    "*TestUfunc::test_scalar_equal": (
        "SUBCLASS_RETURN: scalar comparison returns WhestArray instead of bool"
    ),
    "*TestUfunc::test_struct_ufunc": (
        "SUBCLASS_RETURN: structured ufunc result preserves WhestArray subclass"
    ),
    "*TestUfunc::test_safe_casting": (
        "SUBCLASS_RETURN: safe casting check sees WhestArray subclass"
    ),
    "*TestUfunc::test_broadcast": (
        "SUBCLASS_RETURN: broadcast result preserves WhestArray subclass"
    ),
    "*TestNonzero::test_return_type": (
        "SUBCLASS_RETURN: nonzero returns WhestArray instead of plain ndarray tuple"
    ),
    "*TestRequire::test_non_array_input": (
        "SUBCLASS_RETURN: np.require called on WhestArray returns subclass"
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
    # ------------------------------------------------------------------ #
    # OWNDATA_VIEW — _aswhest view-cast loses OWNDATA flag            #
    # ------------------------------------------------------------------ #
    # whest's _aswhest wraps numpy results via .view(WhestArray)
    # which produces a view (OWNDATA=False) rather than an owning array.
    # NumPy's TestClip tests use assert_array_strict_equal which compares
    # ndarray.flags including OWNDATA. The data and shape are correct;
    # only the OWNDATA flag differs. Same root cause for several other tests.
    "*TestClip::test_simple_int": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    "*TestClip::test_simple_double": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    "*TestClip::test_simple_complex": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_simple_inplace_01": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_simple_inplace_02": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_simple_int32_out": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_simple_int64_inout": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_simple_int64_out": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_array_double": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    "*TestClip::test_clip_func_takes_out": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_clip_inplace_array": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_clip_inplace_simple": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_clip_non_contig": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_clip_with_out_array_int32": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_clip_with_out_array_outint32": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_clip_with_out_simple2": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_clip_with_out_simple_int32": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"
    ),
    "*TestClip::test_type_cast_01": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    "*TestClip::test_type_cast_02": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    "*TestClip::test_type_cast_03": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    "*TestClip::test_type_cast_04": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    "*TestClip::test_type_cast_05": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    "*TestClip::test_type_cast_06": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    "*TestClip::test_type_cast_09": ("OWNDATA_VIEW: _aswhest view-cast loses OWNDATA"),
    # ------------------------------------------------------------------ #
    # NUMPY_INTERNAL — fromiter/resize edge cases                         #
    # ------------------------------------------------------------------ #
    "*TestResize::test_reshape_from_zero": (
        "NUMPY_INTERNAL: resize from zero-element array edge case"
    ),
    "*TestFromiter::test_growth_and_complicated_dtypes*i,O*": (
        "NUMPY_INTERNAL: fromiter with object dtype interacts unexpectedly with patched np"
    ),
    "*TestClip::test_simple_int32_inout*unsafe*": (
        "OWNDATA_VIEW: _aswhest view-cast loses OWNDATA on unsafe cast"
    ),
    "*TestOut::test_out_wrap_no_leak": (
        "NUMPY_INTERNAL: refcount check sees unexpected count due to WhestArray subclass wrapping"
    ),
}
