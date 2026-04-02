"""Registry of all public numpy 2.1.3 callables with FLOP-counting categories.

Categories
----------
counted_unary      scalar math on each element, cost = numel(output)
counted_binary     element-wise binary op, cost = numel(output)
counted_reduction  reduce array, cost = numel(input)
counted_custom     bespoke cost formulas
free               zero FLOP cost (allocation, indexing, shape ops, etc.)
blacklisted        intentionally unsupported
"""

from __future__ import annotations

REGISTRY_META: dict = {
    "numpy_version": "2.1.3",
    "last_updated": "2026-04-01",
}

# ---------------------------------------------------------------------------
# Full registry — every entry has:  category, module, notes
# ---------------------------------------------------------------------------
REGISTRY: dict[str, dict] = {
    # ------------------------------------------------------------------
    # counted_unary — implemented in _pointwise.py
    # ------------------------------------------------------------------
    "abs": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise absolute value; alias for absolute.",
    },
    "absolute": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise absolute value.",
    },
    "negative": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise negation.",
    },
    "positive": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise unary plus (copy with sign preserved).",
    },
    "exp": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise e^x.",
    },
    "exp2": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise 2^x.",
    },
    "expm1": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise e^x - 1 (accurate near zero).",
    },
    "log": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise natural logarithm.",
    },
    "log2": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise base-2 logarithm.",
    },
    "log10": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise base-10 logarithm.",
    },
    "log1p": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise log(1+x) (accurate near zero).",
    },
    "sqrt": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise square root.",
    },
    "cbrt": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise cube root.",
    },
    "square": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise x^2.",
    },
    "reciprocal": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise 1/x.",
    },
    "sin": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise sine.",
    },
    "cos": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise cosine.",
    },
    "tan": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise tangent.",
    },
    "arcsin": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse sine.",
    },
    "arccos": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse cosine.",
    },
    "arctan": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse tangent.",
    },
    "sinh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise hyperbolic sine.",
    },
    "cosh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise hyperbolic cosine.",
    },
    "tanh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise hyperbolic tangent.",
    },
    "arcsinh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse hyperbolic sine.",
    },
    "arccosh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse hyperbolic cosine.",
    },
    "arctanh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise inverse hyperbolic tangent.",
    },
    # asin/acos/atan/asinh/acosh/atanh — NumPy 2.x aliases
    "asin": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arcsin (NumPy 2.x).",
    },
    "acos": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arccos (NumPy 2.x).",
    },
    "atan": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arctan (NumPy 2.x).",
    },
    "asinh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arcsinh (NumPy 2.x).",
    },
    "acosh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arccosh (NumPy 2.x).",
    },
    "atanh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for arctanh (NumPy 2.x).",
    },
    "degrees": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Convert radians to degrees element-wise.",
    },
    "radians": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Convert degrees to radians element-wise.",
    },
    "deg2rad": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for radians.",
    },
    "rad2deg": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for degrees.",
    },
    "sign": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise sign function.",
    },
    "signbit": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Returns True for elements with negative sign bit.",
    },
    "fabs": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise absolute value (always float).",
    },
    "ceil": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise ceiling.",
    },
    "floor": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise floor.",
    },
    "rint": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Round to nearest integer element-wise.",
    },
    "trunc": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Truncate toward zero element-wise.",
    },
    "fix": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Round toward zero element-wise (alias for trunc).",
    },
    "round": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Round to given number of decimals element-wise.",
    },
    "around": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Alias for round.",
    },
    "spacing": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return ULP spacing for each element.",
    },
    "logical_not": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise logical NOT.",
    },
    "bitwise_not": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise bitwise NOT.",
    },
    "bitwise_invert": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise bitwise invert (alias for bitwise_not).",
    },
    "bitwise_count": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Count set bits element-wise (popcount).",
    },
    "invert": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Bitwise NOT element-wise.",
    },
    "conj": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Complex conjugate element-wise.",
    },
    "conjugate": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Complex conjugate element-wise.",
    },
    "angle": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return angle of complex argument element-wise.",
    },
    "real": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return real part of complex array.",
    },
    "imag": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return imaginary part of complex array.",
    },
    "sinc": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Normalized sinc function element-wise.",
    },
    "i0": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Modified Bessel function of order 0, element-wise.",
    },
    "nan_to_num": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Replace NaN/inf with finite numbers element-wise.",
    },
    "modf": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return fractional and integral parts element-wise.",
    },
    "frexp": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Decompose x into mantissa and exponent element-wise.",
    },
    "isclose": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise approximate equality test.",
    },
    "isnat": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Test for NaT (not-a-time) element-wise.",
    },
    "isneginf": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Test for negative infinity element-wise.",
    },
    "isposinf": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Test for positive infinity element-wise.",
    },
    "isreal": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Test if element is real (imag == 0) element-wise.",
    },
    "iscomplex": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Test if element is complex element-wise.",
    },
    "iscomplexobj": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return True if input is a complex type or array.",
    },
    "isrealobj": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return True if x is a not complex type or array.",
    },
    "real_if_close": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Return real array if imaginary part is negligible.",
    },
    "sort_complex": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Sort complex array by real then imaginary part.",
    },
    # ------------------------------------------------------------------
    # counted_binary — implemented in _pointwise.py
    # ------------------------------------------------------------------
    "add": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise addition.",
    },
    "subtract": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise subtraction.",
    },
    "multiply": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise multiplication.",
    },
    "divide": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise true division.",
    },
    "true_divide": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise true division (explicit).",
    },
    "floor_divide": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise floor division.",
    },
    "power": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise exponentiation x**y.",
    },
    "pow": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Alias for power (NumPy 2.x).",
    },
    "float_power": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise exponentiation in float64.",
    },
    "mod": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise modulo.",
    },
    "remainder": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise remainder (same as mod).",
    },
    "fmod": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise C-style fmod (remainder toward zero).",
    },
    "maximum": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise maximum (propagates NaN).",
    },
    "minimum": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise minimum (propagates NaN).",
    },
    "fmax": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise maximum ignoring NaN.",
    },
    "fmin": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise minimum ignoring NaN.",
    },
    "logaddexp": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "log(exp(x1) + exp(x2)) element-wise.",
    },
    "logaddexp2": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "log2(2**x1 + 2**x2) element-wise.",
    },
    "arctan2": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise arctan(y/x) considering quadrant.",
    },
    "atan2": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Alias for arctan2 (NumPy 2.x).",
    },
    "hypot": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise Euclidean norm sqrt(x1^2 + x2^2).",
    },
    "copysign": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Copy sign of x2 to magnitude of x1 element-wise.",
    },
    "nextafter": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Return next float after x1 toward x2 element-wise.",
    },
    "ldexp": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Return x1 * 2**x2 element-wise.",
    },
    "heaviside": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Heaviside step function element-wise.",
    },
    "greater": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 > x2.",
    },
    "greater_equal": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 >= x2.",
    },
    "less": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 < x2.",
    },
    "less_equal": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 <= x2.",
    },
    "equal": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 == x2.",
    },
    "not_equal": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise x1 != x2.",
    },
    "logical_and": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise logical AND.",
    },
    "logical_or": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise logical OR.",
    },
    "logical_xor": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise logical XOR.",
    },
    "bitwise_and": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise bitwise AND.",
    },
    "bitwise_or": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise bitwise OR.",
    },
    "bitwise_xor": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise bitwise XOR.",
    },
    "bitwise_left_shift": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise left bit shift.",
    },
    "bitwise_right_shift": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise right bit shift.",
    },
    "left_shift": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise left bit shift (legacy name).",
    },
    "right_shift": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise right bit shift (legacy name).",
    },
    "gcd": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise greatest common divisor.",
    },
    "lcm": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise least common multiple.",
    },
    "divmod": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise (quotient, remainder) tuple.",
    },
    "vecdot": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Vector dot product along last axis.",
    },
    # ------------------------------------------------------------------
    # counted_reduction — implemented in _pointwise.py
    # ------------------------------------------------------------------
    "sum": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Sum of array elements.",
    },
    "prod": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Product of array elements.",
    },
    "max": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Maximum value of array.",
    },
    "min": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Minimum value of array.",
    },
    "amax": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Maximum value of array (alias for max/numpy.amax).",
    },
    "amin": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Minimum value of array (alias for min/numpy.amin).",
    },
    "mean": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Arithmetic mean of array elements.",
    },
    "std": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Standard deviation; cost_multiplier=2 (two passes).",
    },
    "var": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Variance; cost_multiplier=2 (two passes).",
    },
    "argmax": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Index of maximum value.",
    },
    "argmin": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Index of minimum value.",
    },
    "cumsum": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative sum of array elements.",
    },
    "cumprod": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative product of array elements.",
    },
    "any": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Test whether any array element is true.",
    },
    "all": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Test whether all array elements are true.",
    },
    "count_nonzero": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Count non-zero elements.",
    },
    "nansum": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Sum ignoring NaNs.",
    },
    "nanprod": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Product ignoring NaNs.",
    },
    "nanmax": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Maximum ignoring NaNs.",
    },
    "nanmin": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Minimum ignoring NaNs.",
    },
    "nanmean": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Mean ignoring NaNs.",
    },
    "nanstd": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Standard deviation ignoring NaNs.",
    },
    "nanvar": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Variance ignoring NaNs.",
    },
    "nanargmax": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Index of maximum ignoring NaNs.",
    },
    "nanargmin": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Index of minimum ignoring NaNs.",
    },
    "nancumsum": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative sum ignoring NaNs.",
    },
    "nancumprod": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative product ignoring NaNs.",
    },
    "median": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Median of array elements (sorts internally).",
    },
    "nanmedian": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Median ignoring NaNs.",
    },
    "percentile": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "q-th percentile of array elements.",
    },
    "nanpercentile": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "q-th percentile ignoring NaNs.",
    },
    "quantile": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "q-th quantile of array elements.",
    },
    "nanquantile": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "q-th quantile ignoring NaNs.",
    },
    "ptp": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Peak-to-peak (max - min) range of array.",
    },
    "average": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Weighted average of array elements.",
    },
    "cumulative_sum": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative sum (NumPy 2.x array API).",
    },
    "cumulative_prod": {
        "category": "counted_reduction",
        "module": "numpy",
        "notes": "Cumulative product (NumPy 2.x array API).",
    },
    # ------------------------------------------------------------------
    # counted_custom — implemented in _pointwise.py / _einsum.py
    # ------------------------------------------------------------------
    "dot": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Dot product; cost = 2*M*N*K for matrix multiply.",
    },
    "matmul": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Matrix multiplication; cost = 2*M*N*K.",
    },
    "einsum": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Generalized Einstein summation.",
    },
    "einsum_path": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Optimize einsum contraction path (no numeric output).",
    },
    "clip": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Clip array to [a_min, a_max] element-wise.",
    },
    "inner": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Inner product; cost = 2*N for 1-D, 2*N*M for n-D.",
    },
    "outer": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Outer product of two vectors; cost = M*N.",
    },
    "tensordot": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Tensor dot product along specified axes.",
    },
    "vdot": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Dot product with conjugation; cost = 2*N.",
    },
    "kron": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Kronecker product; cost proportional to output size.",
    },
    "cross": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Cross product of two 3-D vectors.",
    },
    "diff": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "n-th discrete difference along axis.",
    },
    "gradient": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Gradient using central differences.",
    },
    "ediff1d": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Differences between consecutive elements.",
    },
    "convolve": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "1-D discrete convolution.",
    },
    "correlate": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "1-D cross-correlation.",
    },
    "corrcoef": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Pearson correlation coefficients.",
    },
    "cov": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Covariance matrix.",
    },
    "trapezoid": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Integrate using the trapezoidal rule.",
    },
    "trapz": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Alias for trapezoid (deprecated).",
    },
    "interp": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "1-D linear interpolation.",
    },
    # ------------------------------------------------------------------
    # linalg — svd + decompositions implemented; rest blacklisted
    # ------------------------------------------------------------------
    "linalg.svd": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Singular value decomposition; cost ~ O(min(m,n)*m*n).",
    },
    "linalg.cholesky": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Cholesky decomposition. Cost: $n^3/3$ (Golub & Van Loan §4.2).",
    },
    "linalg.cond": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Condition number. Cost: m*n*min(m,n) (via SVD).",
    },
    "linalg.cross": {
        "category": "free",
        "module": "numpy.linalg",
        "notes": "Alias for numpy.cross — delegates to mechestim.cross.",
    },
    "linalg.det": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Determinant. Cost: $n^3$ (LU factorization).",
    },
    "linalg.diagonal": {
        "category": "free",
        "module": "numpy.linalg",
        "notes": "View of diagonal — delegates to mechestim.diagonal. Cost: 0 FLOPs.",
    },
    "linalg.eig": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Eigendecomposition. Cost: $10n^3$ (Francis QR, Golub & Van Loan §7.5).",
    },
    "linalg.eigh": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Symmetric eigendecomposition. Cost: $(4/3)n^3$ (Golub & Van Loan §8.3).",
    },
    "linalg.eigvals": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Eigenvalues only. Cost: $10n^3$ (same as eig).",
    },
    "linalg.eigvalsh": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Symmetric eigenvalues. Cost: $(4/3)n^3$ (same as eigh).",
    },
    "linalg.inv": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix inverse. Cost: $n^3$ (LU + solve).",
    },
    "linalg.lstsq": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Least squares. Cost: m*n*min(m,n) (LAPACK gelsd/SVD).",
    },
    "linalg.matmul": {
        "category": "free",
        "module": "numpy.linalg",
        "notes": "Alias for numpy.matmul — delegates to mechestim.matmul.",
    },
    "linalg.matrix_norm": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix norm. Cost depends on ord: 2*numel for Frobenius, m*n*min(m,n) for ord=2.",
    },
    "linalg.matrix_power": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix power. Cost: $(\\lfloor\\log_2 k\\rfloor + \\text{popcount}(k) - 1) \\cdot n^3$ (exponentiation by squaring).",
    },
    "linalg.matrix_rank": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix rank. Cost: m*n*min(m,n) (via SVD).",
    },
    "linalg.matrix_transpose": {
        "category": "free",
        "module": "numpy.linalg",
        "notes": "Transpose view — delegates to mechestim.matrix_transpose. Cost: 0 FLOPs.",
    },
    "linalg.multi_dot": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Chain matmul. Cost: sum of optimal chain matmul costs (CLRS §15.2).",
    },
    "linalg.norm": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Norm. Cost depends on ord: numel for L1/inf, 2*numel for Frobenius, m*n*min(m,n) for ord=2.",
    },
    "linalg.outer": {
        "category": "free",
        "module": "numpy.linalg",
        "notes": "Alias for numpy.outer — delegates to mechestim.outer.",
    },
    "linalg.pinv": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Pseudoinverse. Cost: m*n*min(m,n) (via SVD).",
    },
    "linalg.qr": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "QR decomposition. Cost: $2mn^2 - (2/3)n^3$ (Golub & Van Loan §5.2).",
    },
    "linalg.slogdet": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Sign + log determinant. Cost: $n^3$ (LU factorization).",
    },
    "linalg.solve": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Solve Ax=b. Cost: $n^3$ (LU factorization).",
    },
    "linalg.svdvals": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Singular values only. Cost: m*n*min(m,n) (Golub-Reinsch).",
    },
    "linalg.tensordot": {
        "category": "free",
        "module": "numpy.linalg",
        "notes": "Alias for numpy.tensordot — delegates to mechestim.tensordot.",
    },
    "linalg.tensorinv": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Tensor inverse. Cost: $n^3$ after reshape (delegates to inv).",
    },
    "linalg.tensorsolve": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Tensor solve. Cost: $n^3$ after reshape (delegates to solve).",
    },
    "linalg.trace": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Matrix trace. Cost: n (sum of diagonal elements).",
    },
    "linalg.vecdot": {
        "category": "free",
        "module": "numpy.linalg",
        "notes": "Alias for numpy.vecdot — delegates to mechestim.vecdot.",
    },
    "linalg.vector_norm": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Vector norm. Cost: numel (or 2*numel for general p-norm).",
    },
    # ------------------------------------------------------------------
    # fft — counted_custom (14 transforms) + free (4 utility ops)
    # ------------------------------------------------------------------
    "fft.fft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.fft2": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.fftn": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.fftfreq": {
        "category": "free",
        "module": "numpy.fft",
        "notes": "FFT sample frequencies. No arithmetic; returns index array.",
    },
    "fft.fftshift": {
        "category": "free",
        "module": "numpy.fft",
        "notes": "Shift zero-frequency component to center. No arithmetic; index reordering only.",
    },
    "fft.hfft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "FFT of Hermitian-symmetric signal. Cost: 5*n_out*ceil(log2(n_out)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.ifft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse 1-D complex FFT. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.ifft2": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse 2-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.ifftn": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse N-D complex FFT. Cost: 5*N*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.ifftshift": {
        "category": "free",
        "module": "numpy.fft",
        "notes": "Inverse of fftshift. No arithmetic; index reordering only.",
    },
    "fft.ihfft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse FFT of Hermitian signal. Cost: 5*n*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.irfft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse 1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.irfft2": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse 2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.irfftn": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "Inverse N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.rfft": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "1-D real FFT. Cost: 5*(n//2)*ceil(log2(n)) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.rfft2": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "2-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    "fft.rfftfreq": {
        "category": "free",
        "module": "numpy.fft",
        "notes": "Real FFT sample frequencies. No arithmetic; returns index array.",
    },
    "fft.rfftn": {
        "category": "counted_custom",
        "module": "numpy.fft",
        "notes": "N-D real FFT. Cost: 5*(N//2)*ceil(log2(N)), N=prod(s) (Cooley-Tukey radix-2; Van Loan 1992 §1.4).",
    },
    # ------------------------------------------------------------------
    # free — implemented in _free_ops.py
    # ------------------------------------------------------------------
    "array": {
        "category": "free",
        "module": "numpy",
        "notes": "Create array from data.",
    },
    "zeros": {
        "category": "free",
        "module": "numpy",
        "notes": "Create zero-filled array.",
    },
    "ones": {
        "category": "free",
        "module": "numpy",
        "notes": "Create one-filled array.",
    },
    "full": {
        "category": "free",
        "module": "numpy",
        "notes": "Create array filled with scalar value.",
    },
    "eye": {
        "category": "free",
        "module": "numpy",
        "notes": "Create identity matrix.",
    },
    "identity": {
        "category": "free",
        "module": "numpy",
        "notes": "Create square identity matrix.",
    },
    "diag": {
        "category": "free",
        "module": "numpy",
        "notes": "Extract diagonal or construct diagonal array.",
    },
    "arange": {
        "category": "free",
        "module": "numpy",
        "notes": "Return evenly spaced values in given interval.",
    },
    "linspace": {
        "category": "free",
        "module": "numpy",
        "notes": "Return evenly spaced numbers over interval.",
    },
    "zeros_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Array of zeros with same shape/type as input.",
    },
    "ones_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Array of ones with same shape/type as input.",
    },
    "full_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Array filled with scalar, same shape/type as input.",
    },
    "empty": {
        "category": "free",
        "module": "numpy",
        "notes": "Uninitialized array allocation.",
    },
    "empty_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Uninitialized array with same shape/type as input.",
    },
    "reshape": {
        "category": "free",
        "module": "numpy",
        "notes": "Reshape array without copying.",
    },
    "transpose": {
        "category": "free",
        "module": "numpy",
        "notes": "Permute array dimensions.",
    },
    "swapaxes": {
        "category": "free",
        "module": "numpy",
        "notes": "Interchange two axes of an array.",
    },
    "moveaxis": {
        "category": "free",
        "module": "numpy",
        "notes": "Move axes to new positions.",
    },
    "concatenate": {
        "category": "free",
        "module": "numpy",
        "notes": "Join arrays along axis.",
    },
    "stack": {
        "category": "free",
        "module": "numpy",
        "notes": "Join arrays along new axis.",
    },
    "vstack": {
        "category": "free",
        "module": "numpy",
        "notes": "Stack arrays vertically.",
    },
    "hstack": {
        "category": "free",
        "module": "numpy",
        "notes": "Stack arrays horizontally.",
    },
    "split": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into sub-arrays.",
    },
    "hsplit": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into columns.",
    },
    "vsplit": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into rows.",
    },
    "squeeze": {
        "category": "free",
        "module": "numpy",
        "notes": "Remove size-1 dimensions.",
    },
    "expand_dims": {
        "category": "free",
        "module": "numpy",
        "notes": "Insert new size-1 axis.",
    },
    "ravel": {
        "category": "free",
        "module": "numpy",
        "notes": "Return contiguous flattened array.",
    },
    "copy": {
        "category": "free",
        "module": "numpy",
        "notes": "Return array copy.",
    },
    "where": {
        "category": "free",
        "module": "numpy",
        "notes": "Select elements based on condition.",
    },
    "tile": {
        "category": "free",
        "module": "numpy",
        "notes": "Repeat array by tiling.",
    },
    "repeat": {
        "category": "free",
        "module": "numpy",
        "notes": "Repeat elements of an array.",
    },
    "flip": {
        "category": "free",
        "module": "numpy",
        "notes": "Reverse order of elements along axis.",
    },
    "roll": {
        "category": "free",
        "module": "numpy",
        "notes": "Roll array elements along axis.",
    },
    "sort": {
        "category": "free",
        "module": "numpy",
        "notes": "Return sorted array.",
    },
    "argsort": {
        "category": "free",
        "module": "numpy",
        "notes": "Indirect sort returning indices.",
    },
    "searchsorted": {
        "category": "free",
        "module": "numpy",
        "notes": "Find insertion points to maintain sorted order.",
    },
    "unique": {
        "category": "free",
        "module": "numpy",
        "notes": "Find unique elements of array.",
    },
    "pad": {
        "category": "free",
        "module": "numpy",
        "notes": "Pad array.",
    },
    "triu": {
        "category": "free",
        "module": "numpy",
        "notes": "Upper triangle of array.",
    },
    "tril": {
        "category": "free",
        "module": "numpy",
        "notes": "Lower triangle of array.",
    },
    "diagonal": {
        "category": "free",
        "module": "numpy",
        "notes": "Return specified diagonals.",
    },
    "trace": {
        "category": "free",
        "module": "numpy",
        "notes": "Sum along diagonals.",
    },
    "broadcast_to": {
        "category": "free",
        "module": "numpy",
        "notes": "Broadcast array to new shape.",
    },
    "meshgrid": {
        "category": "free",
        "module": "numpy",
        "notes": "Coordinate matrices from coordinate vectors.",
    },
    "astype": {
        "category": "free",
        "module": "numpy",
        "notes": "Cast array to specified type.",
    },
    "asarray": {
        "category": "free",
        "module": "numpy",
        "notes": "Convert input to array.",
    },
    "isnan": {
        "category": "free",
        "module": "numpy",
        "notes": "Test for NaN element-wise.",
    },
    "isinf": {
        "category": "free",
        "module": "numpy",
        "notes": "Test for infinity element-wise.",
    },
    "isfinite": {
        "category": "free",
        "module": "numpy",
        "notes": "Test for finite values element-wise.",
    },
    "allclose": {
        "category": "free",
        "module": "numpy",
        "notes": "Test whether two arrays are element-wise equal within tolerance.",
    },
    # Additional free ops
    "rot90": {
        "category": "free",
        "module": "numpy",
        "notes": "Rotate array 90 degrees.",
    },
    "fliplr": {
        "category": "free",
        "module": "numpy",
        "notes": "Flip array left-right.",
    },
    "flipud": {
        "category": "free",
        "module": "numpy",
        "notes": "Flip array up-down.",
    },
    "atleast_1d": {
        "category": "free",
        "module": "numpy",
        "notes": "View inputs as arrays with at least one dimension.",
    },
    "atleast_2d": {
        "category": "free",
        "module": "numpy",
        "notes": "View inputs as arrays with at least two dimensions.",
    },
    "atleast_3d": {
        "category": "free",
        "module": "numpy",
        "notes": "View inputs as arrays with at least three dimensions.",
    },
    "column_stack": {
        "category": "free",
        "module": "numpy",
        "notes": "Stack 1-D arrays as columns into 2-D array.",
    },
    "dstack": {
        "category": "free",
        "module": "numpy",
        "notes": "Stack arrays depth-wise (along third axis).",
    },
    "row_stack": {
        "category": "free",
        "module": "numpy",
        "notes": "Stack arrays vertically (alias for vstack).",
    },
    "flatnonzero": {
        "category": "free",
        "module": "numpy",
        "notes": "Return indices of non-zero elements in flattened array.",
    },
    "nonzero": {
        "category": "free",
        "module": "numpy",
        "notes": "Return indices of non-zero elements.",
    },
    "argwhere": {
        "category": "free",
        "module": "numpy",
        "notes": "Find indices of non-zero elements.",
    },
    "isin": {
        "category": "free",
        "module": "numpy",
        "notes": "Test element-wise membership in array.",
    },
    "in1d": {
        "category": "free",
        "module": "numpy",
        "notes": "Test whether each element of 1-D array is in another.",
    },
    "select": {
        "category": "free",
        "module": "numpy",
        "notes": "Return array from list of choices based on conditions.",
    },
    "extract": {
        "category": "free",
        "module": "numpy",
        "notes": "Return elements satisfying condition.",
    },
    "place": {
        "category": "free",
        "module": "numpy",
        "notes": "Change elements satisfying condition.",
    },
    "put": {
        "category": "free",
        "module": "numpy",
        "notes": "Replace elements at given flat indices.",
    },
    "put_along_axis": {
        "category": "free",
        "module": "numpy",
        "notes": "Put values into destination array using indices.",
    },
    "putmask": {
        "category": "free",
        "module": "numpy",
        "notes": "Change elements of array based on condition and input values.",
    },
    "take": {
        "category": "free",
        "module": "numpy",
        "notes": "Take elements from array along axis.",
    },
    "take_along_axis": {
        "category": "free",
        "module": "numpy",
        "notes": "Take values from input array by matching 1-D index.",
    },
    "choose": {
        "category": "free",
        "module": "numpy",
        "notes": "Construct array from index array and choices.",
    },
    "compress": {
        "category": "free",
        "module": "numpy",
        "notes": "Return selected slices along axis.",
    },
    "array_equal": {
        "category": "free",
        "module": "numpy",
        "notes": "True if two arrays have same shape and elements.",
    },
    "array_equiv": {
        "category": "free",
        "module": "numpy",
        "notes": "True if two arrays are shape-consistent and element-equal.",
    },
    "shape": {
        "category": "free",
        "module": "numpy",
        "notes": "Return shape of array.",
    },
    "size": {
        "category": "free",
        "module": "numpy",
        "notes": "Return number of elements in array.",
    },
    "ndim": {
        "category": "free",
        "module": "numpy",
        "notes": "Return number of dimensions of array.",
    },
    "dsplit": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into multiple sub-arrays depth-wise.",
    },
    "array_split": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array into sub-arrays (possibly unequal).",
    },
    "trim_zeros": {
        "category": "free",
        "module": "numpy",
        "notes": "Trim leading/trailing zeros from 1-D array.",
    },
    "resize": {
        "category": "free",
        "module": "numpy",
        "notes": "Return new array with given shape by repeating.",
    },
    "broadcast_shapes": {
        "category": "free",
        "module": "numpy",
        "notes": "Compute broadcast shape from input shapes.",
    },
    "broadcast_arrays": {
        "category": "free",
        "module": "numpy",
        "notes": "Broadcast arrays against each other.",
    },
    "result_type": {
        "category": "free",
        "module": "numpy",
        "notes": "Return type that results from applying NumPy type promotion.",
    },
    "can_cast": {
        "category": "free",
        "module": "numpy",
        "notes": "Returns True if cast is safe.",
    },
    "common_type": {
        "category": "free",
        "module": "numpy",
        "notes": "Return scalar type common to all input arrays.",
    },
    "min_scalar_type": {
        "category": "free",
        "module": "numpy",
        "notes": "Return the minimum scalar type for a value.",
    },
    "promote_types": {
        "category": "free",
        "module": "numpy",
        "notes": "Return smallest type to which both types may be safely cast.",
    },
    "shares_memory": {
        "category": "free",
        "module": "numpy",
        "notes": "Determine if two arrays share memory.",
    },
    "may_share_memory": {
        "category": "free",
        "module": "numpy",
        "notes": "Determine if two arrays might share memory.",
    },
    "packbits": {
        "category": "free",
        "module": "numpy",
        "notes": "Pack elements of array into bits.",
    },
    "unpackbits": {
        "category": "free",
        "module": "numpy",
        "notes": "Unpack elements of array into bits.",
    },
    "fromfunction": {
        "category": "free",
        "module": "numpy",
        "notes": "Construct array by executing function over each coordinate.",
    },
    "fromiter": {
        "category": "free",
        "module": "numpy",
        "notes": "Create array from an iterable.",
    },
    "frombuffer": {
        "category": "free",
        "module": "numpy",
        "notes": "Interpret buffer as 1-D array.",
    },
    "fromstring": {
        "category": "free",
        "module": "numpy",
        "notes": "Create 1-D array from string data.",
    },
    "fromfile": {
        "category": "free",
        "module": "numpy",
        "notes": "Construct array from binary/text file.",
    },
    "fromregex": {
        "category": "free",
        "module": "numpy",
        "notes": "Construct array from text file using regex.",
    },
    "from_dlpack": {
        "category": "free",
        "module": "numpy",
        "notes": "Create ndarray from DLPack object (zero-copy).",
    },
    "block": {
        "category": "free",
        "module": "numpy",
        "notes": "Assemble ndarray from nested list of blocks.",
    },
    "bmat": {
        "category": "free",
        "module": "numpy",
        "notes": "Build matrix from nested list of matrices.",
    },
    "lexsort": {
        "category": "free",
        "module": "numpy",
        "notes": "Indirect stable sort using sequence of keys.",
    },
    "partition": {
        "category": "free",
        "module": "numpy",
        "notes": "Partial sort of array.",
    },
    "argpartition": {
        "category": "free",
        "module": "numpy",
        "notes": "Indirect partial sort returning indices.",
    },
    "union1d": {
        "category": "free",
        "module": "numpy",
        "notes": "Unique union of two arrays.",
    },
    "intersect1d": {
        "category": "free",
        "module": "numpy",
        "notes": "Intersection of two arrays.",
    },
    "setdiff1d": {
        "category": "free",
        "module": "numpy",
        "notes": "Set difference of two arrays.",
    },
    "setxor1d": {
        "category": "free",
        "module": "numpy",
        "notes": "Exclusive-or (symmetric difference) of two arrays.",
    },
    "histogram": {
        "category": "free",
        "module": "numpy",
        "notes": "Compute histogram of dataset.",
    },
    "histogram2d": {
        "category": "free",
        "module": "numpy",
        "notes": "Compute 2-D histogram.",
    },
    "histogramdd": {
        "category": "free",
        "module": "numpy",
        "notes": "Compute multi-dimensional histogram.",
    },
    "histogram_bin_edges": {
        "category": "free",
        "module": "numpy",
        "notes": "Compute bin edges for histogram.",
    },
    "bincount": {
        "category": "free",
        "module": "numpy",
        "notes": "Count occurrences of non-negative integers.",
    },
    "digitize": {
        "category": "free",
        "module": "numpy",
        "notes": "Return indices of bins to which values belong.",
    },
    "unravel_index": {
        "category": "free",
        "module": "numpy",
        "notes": "Convert flat index to multi-dimensional index.",
    },
    "ravel_multi_index": {
        "category": "free",
        "module": "numpy",
        "notes": "Convert multi-dimensional index to flat index.",
    },
    "indices": {
        "category": "free",
        "module": "numpy",
        "notes": "Return array representing indices of a grid.",
    },
    "diag_indices": {
        "category": "free",
        "module": "numpy",
        "notes": "Return indices to access main diagonal of n-D array.",
    },
    "diag_indices_from": {
        "category": "free",
        "module": "numpy",
        "notes": "Return indices to access main diagonal of given array.",
    },
    "diagflat": {
        "category": "free",
        "module": "numpy",
        "notes": "Create diagonal array from flattened input.",
    },
    "mask_indices": {
        "category": "free",
        "module": "numpy",
        "notes": "Return indices of mask for n x n array.",
    },
    "tril_indices": {
        "category": "free",
        "module": "numpy",
        "notes": "Return lower-triangle indices for n x n array.",
    },
    "tril_indices_from": {
        "category": "free",
        "module": "numpy",
        "notes": "Return lower-triangle indices for given array.",
    },
    "triu_indices": {
        "category": "free",
        "module": "numpy",
        "notes": "Return upper-triangle indices for n x n array.",
    },
    "triu_indices_from": {
        "category": "free",
        "module": "numpy",
        "notes": "Return upper-triangle indices for given array.",
    },
    "fill_diagonal": {
        "category": "free",
        "module": "numpy",
        "notes": "Fill main diagonal of given array.",
    },
    "tri": {
        "category": "free",
        "module": "numpy",
        "notes": "Array with ones at and below given diagonal.",
    },
    "geomspace": {
        "category": "free",
        "module": "numpy",
        "notes": "Return numbers spaced evenly on log scale (geometric).",
    },
    "logspace": {
        "category": "free",
        "module": "numpy",
        "notes": "Return numbers spaced evenly on log scale.",
    },
    "concat": {
        "category": "free",
        "module": "numpy",
        "notes": "Join arrays along axis (NumPy 2.x array API alias for concatenate).",
    },
    "vander": {
        "category": "free",
        "module": "numpy",
        "notes": "Generate Vandermonde matrix.",
    },
    "ix_": {
        "category": "free",
        "module": "numpy",
        "notes": "Construct open mesh from multiple sequences.",
    },
    "rollaxis": {
        "category": "free",
        "module": "numpy",
        "notes": "Roll specified axis backwards.",
    },
    "permute_dims": {
        "category": "free",
        "module": "numpy",
        "notes": "Permute dimensions (NumPy 2.x array API).",
    },
    "matrix_transpose": {
        "category": "free",
        "module": "numpy",
        "notes": "Transpose last two dimensions (NumPy 2.x array API).",
    },
    "unstack": {
        "category": "free",
        "module": "numpy",
        "notes": "Unstack array along axis into tuple of arrays (NumPy 2.x).",
    },
    "delete": {
        "category": "free",
        "module": "numpy",
        "notes": "Return array with sub-arrays deleted along axis.",
    },
    "insert": {
        "category": "free",
        "module": "numpy",
        "notes": "Insert values along axis before given indices.",
    },
    "append": {
        "category": "free",
        "module": "numpy",
        "notes": "Append values to end of array.",
    },
    "copyto": {
        "category": "free",
        "module": "numpy",
        "notes": "Copy values from src to dst array.",
    },
    "unique_all": {
        "category": "free",
        "module": "numpy",
        "notes": "Return unique values, indices, inverse, and counts (NumPy 2.x).",
    },
    "unique_counts": {
        "category": "free",
        "module": "numpy",
        "notes": "Return unique values and their counts (NumPy 2.x).",
    },
    "unique_inverse": {
        "category": "free",
        "module": "numpy",
        "notes": "Return unique values and inverse indices (NumPy 2.x).",
    },
    "unique_values": {
        "category": "free",
        "module": "numpy",
        "notes": "Return unique values without extra info (NumPy 2.x).",
    },
    "asarray_chkfinite": {
        "category": "free",
        "module": "numpy",
        "notes": "Convert to array, raising if NaN or inf.",
    },
    "require": {
        "category": "free",
        "module": "numpy",
        "notes": "Return array that satisfies requirements.",
    },
    "issubdtype": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if first argument is lower in type hierarchy.",
    },
    "isdtype": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if array or dtype is of specified kind (NumPy 2.x).",
    },
    "isscalar": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if input is a scalar.",
    },
    "isfortran": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if array is Fortran contiguous.",
    },
    "iterable": {
        "category": "free",
        "module": "numpy",
        "notes": "Return True if object is iterable.",
    },
    "typename": {
        "category": "free",
        "module": "numpy",
        "notes": "Return description of given data type code.",
    },
    "mintypecode": {
        "category": "free",
        "module": "numpy",
        "notes": "Return minimum data type character that can satisfy all given types.",
    },
    "base_repr": {
        "category": "free",
        "module": "numpy",
        "notes": "Return string representation of number in given base.",
    },
    "binary_repr": {
        "category": "free",
        "module": "numpy",
        "notes": "Return binary string representation of the input number.",
    },
    # ------------------------------------------------------------------
    # random — passthrough, category=free
    # ------------------------------------------------------------------
    "random.beta": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Beta distribution.",
    },
    "random.binomial": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from binomial distribution.",
    },
    "random.bytes": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Return random bytes.",
    },
    "random.chisquare": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from chi-square distribution.",
    },
    "random.choice": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Random sample from given 1-D array.",
    },
    "random.default_rng": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Construct a new Generator with default BitGenerator.",
    },
    "random.dirichlet": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Dirichlet distribution.",
    },
    "random.exponential": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from exponential distribution.",
    },
    "random.f": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from F distribution.",
    },
    "random.gamma": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Gamma distribution.",
    },
    "random.geometric": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from geometric distribution.",
    },
    "random.get_state": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Return tuple representing internal state of generator.",
    },
    "random.gumbel": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Gumbel distribution.",
    },
    "random.hypergeometric": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from hypergeometric distribution.",
    },
    "random.laplace": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Laplace distribution.",
    },
    "random.logistic": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from logistic distribution.",
    },
    "random.lognormal": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from log-normal distribution.",
    },
    "random.logseries": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from logarithmic series distribution.",
    },
    "random.multinomial": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from multinomial distribution.",
    },
    "random.multivariate_normal": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from multivariate normal distribution.",
    },
    "random.negative_binomial": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from negative binomial distribution.",
    },
    "random.noncentral_chisquare": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from noncentral chi-square distribution.",
    },
    "random.noncentral_f": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from noncentral F distribution.",
    },
    "random.normal": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from normal (Gaussian) distribution.",
    },
    "random.pareto": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Pareto distribution.",
    },
    "random.permutation": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Randomly permute sequence or return permuted range.",
    },
    "random.poisson": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Poisson distribution.",
    },
    "random.power": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from power distribution with positive exponent.",
    },
    "random.rand": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Random values in [0, 1).",
    },
    "random.randint": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Random integers from low (inclusive) to high (exclusive).",
    },
    "random.randn": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Sample from standard normal distribution.",
    },
    "random.random": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Random floats in [0.0, 1.0).",
    },
    "random.random_integers": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Random integers from low to high (inclusive, deprecated).",
    },
    "random.random_sample": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Random floats in [0.0, 1.0) (alias for random.random).",
    },
    "random.ranf": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Random floats in [0.0, 1.0) (alias).",
    },
    "random.rayleigh": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Rayleigh distribution.",
    },
    "random.sample": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Random floats in [0.0, 1.0) (alias).",
    },
    "random.seed": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Seed random number generator.",
    },
    "random.set_state": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Set internal state of generator.",
    },
    "random.shuffle": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Modify sequence in-place by shuffling.",
    },
    "random.standard_cauchy": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from standard Cauchy distribution.",
    },
    "random.standard_exponential": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from standard exponential distribution.",
    },
    "random.standard_gamma": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from standard Gamma distribution.",
    },
    "random.standard_normal": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from standard normal distribution.",
    },
    "random.standard_t": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from standard Student's t distribution.",
    },
    "random.triangular": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from triangular distribution.",
    },
    "random.uniform": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from uniform distribution.",
    },
    "random.vonmises": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from von Mises distribution.",
    },
    "random.wald": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Wald (inverse Gaussian) distribution.",
    },
    "random.weibull": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Weibull distribution.",
    },
    "random.zipf": {
        "category": "free",
        "module": "numpy.random",
        "notes": "Draw samples from Zipf distribution.",
    },
    # ------------------------------------------------------------------
    # blacklisted — poly functions
    # ------------------------------------------------------------------
    "poly": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Polynomial from roots. Cost: $n^2$ FLOPs.",
    },
    "roots": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Return roots of polynomial with given coefficients. Cost: $10n^3$ FLOPs (companion matrix eig).",
    },
    "polyadd": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Add two polynomials. Cost: max(n1, n2) FLOPs.",
    },
    "polyder": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Differentiate polynomial. Cost: n FLOPs.",
    },
    "polydiv": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Divide one polynomial by another. Cost: n1 * n2 FLOPs.",
    },
    "polyfit": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Least squares polynomial fit. Cost: 2 * m * (deg+1)^2 FLOPs.",
    },
    "polyint": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Integrate polynomial. Cost: n FLOPs.",
    },
    "polymul": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Multiply polynomials. Cost: n1 * n2 FLOPs.",
    },
    "polysub": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Difference (subtraction) of two polynomials. Cost: max(n1, n2) FLOPs.",
    },
    "polyval": {
        "category": "counted_custom",
        "module": "mechestim._polynomial",
        "notes": "Evaluate polynomial at given points. Cost: 2 * m * deg FLOPs (Horner's method).",
    },
    # counted_custom — window functions
    "bartlett": {
        "category": "counted_custom",
        "module": "mechestim._window",
        "notes": "Bartlett window. Cost: n (one linear eval per sample).",
    },
    "blackman": {
        "category": "counted_custom",
        "module": "mechestim._window",
        "notes": "Blackman window. Cost: 3*n (three cosine terms per sample).",
    },
    "hamming": {
        "category": "counted_custom",
        "module": "mechestim._window",
        "notes": "Hamming window. Cost: n (one cosine per sample).",
    },
    "hanning": {
        "category": "counted_custom",
        "module": "mechestim._window",
        "notes": "Hanning window. Cost: n (one cosine per sample).",
    },
    "kaiser": {
        "category": "counted_custom",
        "module": "mechestim._window",
        "notes": "Kaiser window. Cost: 3*n (Bessel function eval per sample).",
    },
    # blacklisted — IO
    "genfromtxt": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Load data from text file with missing values. Not supported.",
    },
    "loadtxt": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Load data from text file. Not supported.",
    },
    "load": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Load arrays from .npy/.npz files. Not supported.",
    },
    "save": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Save array to .npy file. Not supported.",
    },
    "savetxt": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Save array to text file. Not supported.",
    },
    "savez": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Save multiple arrays to .npz file. Not supported.",
    },
    "savez_compressed": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Save multiple arrays to compressed .npz file. Not supported.",
    },
    # blacklisted — config / runtime
    "show_config": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Show NumPy build configuration. Not supported.",
    },
    "show_runtime": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Show runtime info. Not supported.",
    },
    "get_include": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return directory containing NumPy C header files. Not supported.",
    },
    "getbufsize": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return size of buffer used in ufuncs. Not supported.",
    },
    "setbufsize": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Set size of buffer used in ufuncs. Not supported.",
    },
    "geterr": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Get current way of handling floating-point errors. Not supported.",
    },
    "seterr": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Set how floating-point errors are handled. Not supported.",
    },
    "geterrcall": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return current callback function for floating-point errors. Not supported.",
    },
    "seterrcall": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Set callback function for floating-point errors. Not supported.",
    },
    # blacklisted — advanced/meta
    "asmatrix": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Interpret input as matrix (deprecated). Not supported.",
    },
    "nested_iters": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Create nested iterators for multi-index broadcasting. Not supported.",
    },
    "frompyfunc": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Take arbitrary Python function and return NumPy ufunc. Not supported.",
    },
    "piecewise": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Evaluate piecewise-defined function. Not supported.",
    },
    "apply_along_axis": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Apply function along axis. Not supported.",
    },
    "apply_over_axes": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Apply function over multiple axes. Not supported.",
    },
    # blacklisted — datetime
    "datetime_as_string": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Convert datetime array to string representation. Not supported.",
    },
    "datetime_data": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Get information about the step size of datetime dtype. Not supported.",
    },
    "busday_count": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Count valid days between begindate and enddate. Not supported.",
    },
    "busday_offset": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Apply offset to dates subject to valid day rules. Not supported.",
    },
    "is_busday": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Calculates which of given dates are valid days. Not supported.",
    },
    # blacklisted — print/string formatting
    "array2string": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return string representation of array. Not supported.",
    },
    "array_repr": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return string representation of array. Not supported.",
    },
    "array_str": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Return string representation of data in array. Not supported.",
    },
    "format_float_positional": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Format floating point scalar as decimal string. Not supported.",
    },
    "format_float_scientific": {
        "category": "blacklisted",
        "module": "numpy",
        "notes": "Format floating point scalar as scientific notation. Not supported.",
    },
    "unwrap": {
        "category": "counted_custom",
        "module": "mechestim._unwrap",
        "notes": "Phase unwrap. Cost: $\\text{numel}(\\text{input})$ (diff + conditional adjustment).",
    },
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def make_module_getattr(module_prefix: str, module_label: str):
    """Create a __getattr__ that consults the registry."""
    def __getattr__(name: str):
        qualified = f"{module_prefix}{name}" if module_prefix else name
        if qualified in REGISTRY:
            entry = REGISTRY[qualified]
            cat = entry["category"]
            notes = entry.get("notes", "")
            if cat == "blacklisted":
                raise AttributeError(
                    f"{module_label} does not support '{name}' (blacklisted). {notes}"
                )
            if cat == "unclassified":
                raise AttributeError(
                    f"{module_label} has not yet classified '{name}'. "
                    f"Please report this at https://github.com/AIcrowd/mechestim/issues"
                )
            raise AttributeError(
                f"'{name}' is registered but not yet implemented in {module_label}. {notes}"
            )
        raise AttributeError(
            f"{module_label} does not provide '{name}'. "
            f"See https://github.com/AIcrowd/mechestim for supported operations."
        )
    return __getattr__
