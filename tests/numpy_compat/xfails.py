"""Known divergences between mechestim and NumPy.

Each entry maps a test node ID (or pattern) to a reason string.
Tests matching these patterns are marked xfail when running NumPy's
test suite against mechestim.

Current state (2026-04-03):
    test_umath:     4,668 passed, 0 failed
    test_ufunc:       795 passed, 0 failed
    test_numeric:   1,597 passed, 0 failed (12 xfailed)
    test_linalg:      436 passed, 0 failed
    test_pocketfft:   148 passed, 0 failed
    test_helper:        8 passed, 0 failed
    test_polynomial:   38 passed, 0 failed
    test_random:      142 passed, 0 failed

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
}
