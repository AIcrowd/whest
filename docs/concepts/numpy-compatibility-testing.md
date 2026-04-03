# NumPy Compatibility Testing

mechestim's goal is to be a drop-in replacement for NumPy: `import mechestim as np` should work for all supported functions. To verify this, we run NumPy's own test suite against mechestim.

## How it works

A pytest conftest at `tests/numpy_compat/conftest.py` monkeypatches numpy functions with their mechestim equivalents at session start. When we point pytest at NumPy's installed test files using `--pyargs`, every test that calls `np.sum(...)`, `np.mean(...)`, etc. actually calls mechestim's version.

```
NumPy test file                conftest.py               mechestim
  calls np.sum(x)  ──────>   np.sum = me.sum   ──────>  me.sum(x)
  asserts result              (monkeypatch)              (FLOP-counted)
```

## What gets patched

Of mechestim's 482 registered functions, **55 are patched** onto numpy during testing. The rest are skipped for specific reasons:

| Category | Count | Why skipped |
|----------|-------|-------------|
| **Patched** | 55 | Non-ufunc reductions and special functions |
| Ufuncs | 101 | mechestim functions are plain callables, not ufuncs -- they lack `.reduce`, `.accumulate`, `.outer`, `.nargs` |
| Free ops | 220 | Pure pass-throughs that delegate to numpy. Patching causes infinite recursion |
| Counted custom | 36 | `dot`, `matmul`, `einsum`, etc. call `_np.func()` via module lookup. Same recursion issue |
| Submodule | 38 | `linalg.*`, `fft.*`. Same recursion issue |
| Blacklisted | 32 | Intentionally unsupported |

The patched functions include: `all`, `any`, `amax`, `amin`, `argmax`, `argmin`, `average`, `cumsum`, `cumprod`, `mean`, `median`, `std`, `var`, `sum`, `prod`, `isclose`, `real`, `imag`, and more.

## Test suites

We run 8 NumPy test modules covering core math, ufuncs, numerics, linear algebra, FFT, polynomials, and random:

| Suite | Module | Tests |
|-------|--------|-------|
| Core math | `numpy._core.tests.test_umath` | 4,668 |
| Ufunc infrastructure | `numpy._core.tests.test_ufunc` | 795 |
| Numeric operations | `numpy._core.tests.test_numeric` | 1,597 |
| Linear algebra | `numpy.linalg.tests.test_linalg` | 436 |
| FFT | `numpy.fft.tests.test_pocketfft` | 148 |
| FFT helpers | `numpy.fft.tests.test_helper` | 8 |
| Polynomials | `numpy.polynomial.tests.test_polynomial` | 38 |
| Random | `numpy.random.tests.test_random` | 142 |
| **Total** | | **7,832** |

## Running the tests

Tests use `pytest-xdist` for parallel execution across all CPU cores.

```bash
# Run everything (recommended)
make test-numpy-compat

# Run a single suite
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -n auto -q

# Filter to specific functions
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -k "sqrt" -n auto -v

# Run without parallelism (for debugging)
uv run pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -v --tb=short
```

The numpy_compat tests are excluded from the default `pytest` run (via `pyproject.toml` `addopts`) to prevent the monkeypatch from contaminating the main test suite. They run as a separate step in CI.

## Known divergences (xfails)

Tests that fail due to known, accepted differences are tracked in `tests/numpy_compat/xfails.py`. Each entry maps a test pattern to a categorized reason:

| Category | Meaning |
|----------|---------|
| `UNSUPPORTED_DTYPE` | mechestim doesn't support this dtype (timedelta, etc.) |
| `NOT_IMPLEMENTED` | Function exists but lacks a kwarg or edge case behavior |
| `UFUNC_INTERNALS` | Test relies on ufunc protocol mechestim doesn't implement |
| `BUDGET_SIDE_EFFECT` | Test assumes no global state changes |
| `NUMPY_INTERNAL` | Test uses numpy internals unrelated to our functions |

To triage new failures:

1. Run a suite and look at failures: `uv run pytest tests/numpy_compat/ --pyargs <module> -n auto --tb=line`
2. Categorize each failure
3. If it's a bug we should fix, create an issue
4. If it's an accepted divergence, add it to `xfails.py`

## Why monkeypatching (not subclassing)

We considered alternatives:

- **Array subclass with `__array_ufunc__`**: Would intercept ufunc calls, but mechestim arrays are plain `numpy.ndarray` by design -- no custom tensor class.
- **Running tests with `import mechestim as np`**: NumPy's test files import from `numpy._core`, `numpy.testing`, etc. -- can't redirect all internal imports.
- **Monkeypatching**: Simple, works with NumPy's existing test infrastructure, and tests exactly what users experience (same function signatures and behavior).

The trade-off is that ufuncs and internally-delegating functions can't be patched. But the 55 functions we do patch (reductions, special functions) cover the most common compatibility surface.
