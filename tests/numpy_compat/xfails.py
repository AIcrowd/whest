"""Known divergences between mechestim and NumPy.

Each entry maps a test node ID (or pattern) to a reason string.
Tests matching these patterns are marked xfail when running NumPy's
test suite against mechestim.

Current state (2026-04-03):
    test_umath:     4,668 passed, 0 failed  (13 xfailed)
    test_ufunc:       795 passed, 0 failed   (7 xfailed)
    test_numeric:   1,560 passed, 0 failed  (20 xfailed)
    test_linalg:       49 passed, 0 failed  (255 xfailed — expanded submodule patching)
    test_pocketfft:   122 passed, 0 failed  (34 xfailed)
    test_polynomial:  600 passed, 0 failed   (2 xfailed)
    test_random:    1,319 passed, 0 failed  (8 xfailed — counted wrapper signatures)

What we patch (55 functions):
    Non-ufunc reductions and special functions (all, any, amax, amin,
    argmax, argmin, average, cumsum, cumprod, mean, median, std, var,
    sum, prod, etc.) plus misc functions like isclose, real, imag, etc.

What we DON'T patch (and why):
    - Ufuncs (101): mechestim functions are plain callables, not ufuncs.
      They lack .reduce/.accumulate/.outer/.nargs/etc. Patching would
      break test collection and any test using ufunc protocol.
    - Free ops (220): pass-throughs that delegate to numpy. Patching
      causes infinite recursion since mechestim's _np IS np.
    - Counted custom (36): dot, matmul, einsum, convolve, etc. call
      _np.func() via module lookup. Same recursion issue.
    - Submodule functions (38): linalg.*, fft.*. Same recursion issue.
    - Blacklisted (32): intentionally unsupported.
    - linalg.outer: me.linalg.outer delegates to np.outer (not
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
    # mechestim's isclose is a counted wrapper that doesn't support all
    # kwargs/edge cases that numpy.isclose does (timedelta, masked arrays,
    # inplace output, scalar return type preservation, NEP 50 promotion).
    "*TestIsclose::test_non_finite_scalar": (
        "NOT_IMPLEMENTED: mechestim isclose doesn't preserve scalar return type"
    ),
    "*TestIsclose::test_nep50_isclose": (
        "NOT_IMPLEMENTED: mechestim isclose doesn't support NEP 50 promotion"
    ),
    "*TestIsclose::test_ip_none_isclose": (
        "NOT_IMPLEMENTED: mechestim isclose budget deduction changes behavior"
    ),
    "*TestIsclose::test_ip_isclose": (
        "NOT_IMPLEMENTED: mechestim isclose budget deduction changes behavior"
    ),
    "*TestIsclose::test_timedelta": (
        "UNSUPPORTED_DTYPE: mechestim isclose doesn't support timedelta dtype"
    ),
    "*TestIsclose::test_equal_nan": (
        "NOT_IMPLEMENTED: mechestim isclose budget deduction changes behavior"
    ),
    "*TestIsclose::test_scalar_return": (
        "NOT_IMPLEMENTED: mechestim isclose doesn't preserve scalar return type"
    ),
    "*TestIsclose::test_ip_isclose_allclose": (
        "NOT_IMPLEMENTED: mechestim isclose budget deduction changes behavior"
    ),
    "*TestIsclose::test_masked_arrays": (
        "NOT_IMPLEMENTED: mechestim isclose doesn't support masked arrays"
    ),
    "*TestIsclose::test_ip_all_isclose": (
        "NOT_IMPLEMENTED: mechestim isclose budget deduction changes behavior"
    ),
    "*TestIsclose::test_no_parameter_modification": (
        "NOT_IMPLEMENTED: mechestim isclose budget deduction changes behavior"
    ),
    "*TestNonarrayArgs::test_round": (
        "NOT_IMPLEMENTED: mechestim round doesn't support non-array scalar args"
    ),
    # ------------------------------------------------------------------ #
    # test_numeric.py — new divergences from expanded numpy patching      #
    # ------------------------------------------------------------------ #
    "*test_out_of_bound_pyints*": (
        "NOT_IMPLEMENTED: mechestim round/clip don't handle out-of-bound Python ints the same way"
    ),
    "*TestTensordot::test_zero_dimension": (
        "NOT_IMPLEMENTED: mechestim clip doesn't handle 0-d arrays identically"
    ),
    "*TestNonarrayArgs::test_reshape_shape_arg": (
        "NOT_IMPLEMENTED: mechestim reshape doesn't support shape= kwarg"
    ),
    "*test_outer_out_param": (
        "NOT_IMPLEMENTED: mechestim outer doesn't support out= kwarg"
    ),
    "*TestClip::test_ones_pathological*": (
        "NOT_IMPLEMENTED: mechestim ones edge case with pathological dtypes"
    ),
    "*TestClip::test_object_clip": (
        "NOT_IMPLEMENTED: mechestim clip doesn't support object dtype"
    ),
    "*TestClip::test_clip_problem_cases*": (
        "NOT_IMPLEMENTED: mechestim clip edge cases differ"
    ),
    "*TestClip::test_clip_min_max_args": (
        "NOT_IMPLEMENTED: mechestim clip min/max kwarg handling differs"
    ),
    "*TestClip::test_clip_func_takes_out": (
        "NOT_IMPLEMENTED: mechestim clip doesn't support out= kwarg"
    ),
    "*TestClip::test_clip_all_none": (
        "NOT_IMPLEMENTED: mechestim clip(a, None, None) behavior differs"
    ),
    "*TestAsType::test_astype": (
        "NOT_IMPLEMENTED: mechestim free ops astype behavior differs"
    ),
    # ------------------------------------------------------------------ #
    # test_pocketfft.py — out= kwarg and s=None handling                 #
    # ------------------------------------------------------------------ #
    "*TestFFT1D::test_fft_out_argument*": (
        "NOT_IMPLEMENTED: mechestim FFT functions don't support out= kwarg"
    ),
    "*TestFFT1D::test_fftn_out_argument*": (
        "NOT_IMPLEMENTED: mechestim FFT functions don't support out= kwarg"
    ),
    "*TestFFT1D::test_fft_inplace_out*": (
        "NOT_IMPLEMENTED: mechestim FFT functions don't support out= kwarg"
    ),
    "*TestFFT1D::test_fft_bad_out": (
        "NOT_IMPLEMENTED: mechestim FFT functions don't support out= kwarg"
    ),
    "*TestFFT1D::test_s_contains_none*": (
        "NOT_IMPLEMENTED: mechestim FFT functions don't handle s=None per-axis"
    ),
    "*TestFFT1D::test_fftn_out_and_s_interaction*": (
        "NOT_IMPLEMENTED: mechestim FFT out= interaction"
    ),
    "*TestFFT1D::test_irfftn_out_and_s_interaction*": (
        "NOT_IMPLEMENTED: mechestim FFT out= interaction"
    ),
    # ------------------------------------------------------------------ #
    # test_linalg.py — failures from expanded submodule patching          #
    # ------------------------------------------------------------------ #
    # mechestim linalg functions don't support stacked/batched arrays,
    # 0-size arrays, generalized cases, raw mode, or the full set of
    # NumPy linalg kwargs that the numpy test suite exercises.
    "*test_linalg*::test_cross": (
        "NOT_IMPLEMENTED: mechestim cross doesn't raise ValueError for 2D arrays"
    ),
    "*test_linalg*::test_diagonal": (
        "NOT_IMPLEMENTED: mechestim linalg.diagonal behavior differs"
    ),
    "*test_linalg*::test_trace": (
        "NOT_IMPLEMENTED: mechestim linalg.trace behavior differs"
    ),
    "*test_linalg*::test_generalized_raise_multiloop": (
        "NOT_IMPLEMENTED: mechestim linalg doesn't support multiloop generalized cases"
    ),
    "*test_linalg*::test_pinv_rtol_arg": (
        "NOT_IMPLEMENTED: mechestim pinv doesn't support rtol= kwarg"
    ),
    "*test_linalg*::TestCholesky::*": (
        "NOT_IMPLEMENTED: mechestim cholesky doesn't support batched/0-size arrays"
    ),
    "*test_linalg*::TestCond::*": (
        "NOT_IMPLEMENTED: mechestim cond doesn't support stacked/generalized cases"
    ),
    "*test_linalg*::TestDet::test_generalized*": (
        "NOT_IMPLEMENTED: mechestim det doesn't support generalized/stacked cases"
    ),
    "*test_linalg*::TestEig::*": (
        "NOT_IMPLEMENTED: mechestim eig doesn't support 0-size/generalized cases"
    ),
    "*test_linalg*::TestEigh::*": (
        "NOT_IMPLEMENTED: mechestim eigh doesn't support 0-size arrays"
    ),
    "*test_linalg*::TestEighCases::*": (
        "NOT_IMPLEMENTED: mechestim eigh doesn't support generalized/stacked cases"
    ),
    "*test_linalg*::TestEigvals::*": (
        "NOT_IMPLEMENTED: mechestim eigvals doesn't support 0-size/generalized cases"
    ),
    "*test_linalg*::TestEigvalsh::*": (
        "NOT_IMPLEMENTED: mechestim eigvalsh doesn't support 0-size arrays"
    ),
    "*test_linalg*::TestEigvalshCases::*": (
        "NOT_IMPLEMENTED: mechestim eigvalsh doesn't support generalized/stacked cases"
    ),
    "*test_linalg*::TestInv::*": (
        "NOT_IMPLEMENTED: mechestim inv doesn't support 0-size/generalized cases"
    ),
    "*test_linalg*::TestLstsq::*": (
        "NOT_IMPLEMENTED: mechestim lstsq doesn't support 0-size/stacked cases"
    ),
    "*test_linalg*::TestMatrixPower::*": (
        "NOT_IMPLEMENTED: mechestim matrix_power behavior differs for edge cases"
    ),
    "*test_linalg*::TestMatrixRank::*": (
        "NOT_IMPLEMENTED: mechestim matrix_rank behavior differs"
    ),
    "*test_linalg*::TestNormDouble::*": (
        "NOT_IMPLEMENTED: mechestim norm behavior differs from np.linalg.norm"
    ),
    "*test_linalg*::TestNormInt64::*": (
        "NOT_IMPLEMENTED: mechestim norm behavior differs from np.linalg.norm"
    ),
    "*test_linalg*::TestNormSingle::*": (
        "NOT_IMPLEMENTED: mechestim norm behavior differs from np.linalg.norm"
    ),
    "*test_linalg*::TestPinv::test_generalized*": (
        "NOT_IMPLEMENTED: mechestim pinv doesn't support generalized cases"
    ),
    "*test_linalg*::TestPinvHermitian::*": (
        "NOT_IMPLEMENTED: mechestim pinv doesn't support generalized hermitian cases"
    ),
    "*test_linalg*::TestQR::*": (
        "NOT_IMPLEMENTED: mechestim qr doesn't support raw mode or stacked inputs"
    ),
    "*test_linalg*::TestRegression::*": (
        "NOT_IMPLEMENTED: mechestim linalg regression cases differ"
    ),
    "*test_regression*::TestRegression::*": (
        "NOT_IMPLEMENTED: mechestim linalg regression cases differ"
    ),
    "*test_linalg*::TestSolve::*": (
        "NOT_IMPLEMENTED: mechestim solve doesn't support 0-size/1-d/generalized cases"
    ),
    "*test_linalg*::TestSVD::*": (
        "NOT_IMPLEMENTED: mechestim svd doesn't support empty/generalized/stacked cases"
    ),
    "*test_linalg*::TestSVDHermitian::*": (
        "NOT_IMPLEMENTED: mechestim svd hermitian doesn't support generalized/stacked cases"
    ),
    # ------------------------------------------------------------------ #
    # test_polynomial.py — divergences                                    #
    # ------------------------------------------------------------------ #
    "*TestMisc::test_result_type": (
        "NOT_IMPLEMENTED: mechestim polynomial result_type differs"
    ),
    "*TestEvaluation::test_polyval": (
        "NOT_IMPLEMENTED: mechestim polyval doesn't support masked arrays"
    ),
    # ------------------------------------------------------------------ #
    # test_random.py — counted random wrapper signature divergences        #
    # ------------------------------------------------------------------ #
    # mechestim random wrappers are plain functions, not methods on the
    # RandomState class. Tests that use np.random.randint as a bound method
    # (passing self) or test internal RandomState behavior will fail.
    "*TestRandint::test_in_bounds_fuzz": (
        "WRAPPER_SIGNATURE: mechestim randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_rng_zero_and_extremes": (
        "WRAPPER_SIGNATURE: mechestim randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_respect_dtype_singleton": (
        "WRAPPER_SIGNATURE: mechestim randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_bounds_checking": (
        "WRAPPER_SIGNATURE: mechestim randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_repeatability": (
        "WRAPPER_SIGNATURE: mechestim randint is a plain function, not a bound method"
    ),
    "*TestRandint::test_full_range": (
        "WRAPPER_SIGNATURE: mechestim randint is a plain function, not a bound method"
    ),
    "*TestRandomDist::test_shuffle": (
        "WRAPPER_SIGNATURE: mechestim shuffle is a counted wrapper with different signature"
    ),
    "*TestRandomDist::test_shuffle_untyped_warning*": (
        "WRAPPER_SIGNATURE: mechestim shuffle is a counted wrapper with different signature"
    ),
}
