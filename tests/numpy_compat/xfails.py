"""Known divergences between mechestim and NumPy.

Each entry maps a test node ID (or pattern) to a reason string.
Tests matching these patterns are marked xfail when running NumPy's
test suite against mechestim.

Current state (2026-04-05):
    test_umath:     4,668 passed, 0 failed  (13 xfailed)
    test_ufunc:       795 passed, 0 failed   (7 xfailed)
    test_numeric:   1,567 passed, 0 failed   (4 xfailed — was 20, fixed 16)
    test_linalg:       49 passed, 0 failed  (255 xfailed — expanded submodule patching)
    test_pocketfft:   148 passed, 0 failed   (0 xfailed — was 34, fixed all)
    test_polynomial:  600 passed, 0 failed   (2 xfailed)
    test_random:      135 passed, 0 failed   (7 xfailed — was 8, fixed shuffle)

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
    # mechestim's isclose is a counted wrapper; NEP 50 promotion not supported.
    "*TestIsclose::test_nep50_isclose": (
        "NOT_IMPLEMENTED: mechestim isclose doesn't support NEP 50 promotion"
    ),
    # ------------------------------------------------------------------ #
    # test_numeric.py — new divergences from expanded numpy patching      #
    # ------------------------------------------------------------------ #
    "*TestTensordot::test_zero_dimension": (
        "NOT_IMPLEMENTED: mechestim clip doesn't handle 0-d arrays identically"
    ),
    "*TestClip::test_clip_min_max_args": (
        "NOT_IMPLEMENTED: mechestim clip doesn't enforce strict a_min/a_max/min/max argument validation"
    ),
    "*TestAsType::test_astype": (
        "NOT_IMPLEMENTED: mechestim free ops astype behavior differs"
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
    "TestRandomDist::test_shuffle_untyped_warning[numpy.random]": (
        "WRAPPER_SIGNATURE: warning filename points to mechestim wrapper, not test file"
    ),
}
