# NumPy Compatibility Test Harness

Runs NumPy's own test suite with mechestim monkeypatched in.

## How it works

`conftest.py` replaces every numpy function that mechestim implements
with the mechestim version. Then we point pytest at NumPy's installed
test files. Failures are triaged into `xfails.py`.

## Running

```bash
# Run all numpy umath tests through mechestim
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -v

# Filter to specific functions
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -k "sqrt" -v

# Run linalg tests
uv run pytest tests/numpy_compat/ --pyargs numpy.linalg.tests.test_linalg -v

# Run everything (slow, noisy)
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath numpy._core.tests.test_ufunc numpy._core.tests.test_numeric numpy.linalg.tests.test_linalg -v
```

## Triaging failures

1. Run a test module and capture failures
2. For each failure, determine the category (see xfails.py)
3. Add to XFAIL_PATTERNS with the category and explanation
4. Failures we WANT to fix go into GitHub issues instead

## Categories

- `UNSUPPORTED_DTYPE`: we only support float64/float32
- `UFUNC_INTERNALS`: test checks ufunc protocol we don't implement
- `BUDGET_SIDE_EFFECT`: test assumes no global state changes
- `NOT_IMPLEMENTED`: behavioral divergence we accept
