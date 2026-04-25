# NumPy Compatibility Test Harness

Runs NumPy's own test suite with flopscope monkeypatched in.

## How it works

`conftest.py` freezes the original numpy module, rebinds flopscope's internal
`_np` references to the frozen copy, then patches most non-ufunc flopscope
functions onto numpy. This lets NumPy's tests call flopscope's versions while
flopscope internally calls unpatched numpy (avoiding infinite recursion).

Ufuncs (101) and blacklisted ops (32) are skipped. Everything else -- free ops,
counted custom ops, submodule functions (linalg, fft, random) -- is patched.

See `docs/concepts/numpy-compatibility-testing.md` for full details.

## Running

```bash
# Run everything (recommended)
make test-numpy-compat

# Run a single suite
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -n auto -q

# Filter to specific functions
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -k "sqrt" -n auto -v
```

## Current results (2026-04-03)

| Suite | Module | Passed | xfailed |
|-------|--------|--------|---------|
| Core math | `test_umath` | 4,668 | 13 |
| Ufunc infra | `test_ufunc` | 795 | 7 |
| Numeric ops | `test_numeric` | 1,560 | 20 |
| Linear algebra | `test_linalg` | 48 | 255 |
| FFT | `test_pocketfft` | 114 | 34 |
| Polynomials | `test_polynomial` | 36 | 2 |
| Random | `test_random` | 142 | 0 |
| **Total** | | **7,363** | **331** |

## Triaging failures

1. Run a test module and capture failures
2. For each failure, determine the category (see xfails.py)
3. Add to XFAIL_PATTERNS with the category and explanation
4. Failures we WANT to fix go into GitHub issues instead
