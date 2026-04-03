"""Known divergences between mechestim and NumPy.

Each entry maps a test node ID (or pattern) to a reason string.
Tests matching these patterns are marked xfail when running NumPy's
test suite against mechestim.

Categories of expected failures:
- UNSUPPORTED_DTYPE: mechestim only supports float64/float32, not
  longdouble, complex, float16, etc.
- UFUNC_INTERNALS: test checks ufunc attributes (.types, .nargs,
  __array_ufunc__ protocol, etc.) that mechestim functions don't have
- BUDGET_SIDE_EFFECT: test expects no side effects, but mechestim
  deducts from budget
- NOT_IMPLEMENTED: function exists in registry but behavior differs
"""

# Start empty — populated by running the harness and triaging failures.
# Format: {"test_id_pattern": "CATEGORY: explanation"}
XFAIL_PATTERNS: dict[str, str] = {}
