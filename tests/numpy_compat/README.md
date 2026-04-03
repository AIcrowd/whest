# NumPy Compatibility Test Harness

Runs NumPy's own test suite with mechestim monkeypatched in.

## How it works

`conftest.py` replaces non-ufunc mechestim functions onto numpy, then
points pytest at NumPy's installed test files. Currently 55 functions
are patched (reductions and special functions). Ufuncs, free ops,
custom ops, and submodule functions are not patched due to the ufunc
protocol and delegation recursion constraints (see xfails.py header).

## Running

```bash
# Run all three suites in parallel (uses all CPU cores)
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -n auto -q &
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_ufunc -n auto -q &
uv run pytest tests/numpy_compat/ --pyargs numpy.linalg.tests.test_linalg -n auto -q &
wait

# Filter to specific functions
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -k "sqrt" -n auto -v

# Run a single suite verbosely
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -n auto -v
```

## Current results (2026-04-03)

| Suite | Passed | Failed |
|-------|--------|--------|
| test_umath | 4,668 | 0 |
| test_ufunc | 795 | 0 |
| test_linalg | 436 | 0 |

## Triaging failures

1. Run a test module and capture failures
2. For each failure, determine the category (see xfails.py)
3. Add to XFAIL_PATTERNS with the category and explanation
4. Failures we WANT to fix go into GitHub issues instead
