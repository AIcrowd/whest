"""
Exhaustive smoke test: calls EVERY non-blacklisted function in the whest
registry through the client-server proxy to verify it works end-to-end.

Runs INSIDE the participant container (no numpy).
"""

import sys

import whest as we

# ── Results tracking ──────────────────────────────────────────────────────

results = {"pass": [], "fail": [], "skip": []}


def test(name, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        results["pass"].append(name)
        print(f"  PASS: {name}")
        return result
    except Exception as e:
        results["fail"].append((name, str(e)))
        print(f"  FAIL: {name} -- {e}")
        return None


def skip(name, reason="not testable in this context"):
    results["skip"].append((name, reason))
    print(f"  SKIP: {name} -- {reason}")


# ── Main test run ─────────────────────────────────────────────────────────

print("=" * 70)
print("  Exhaustive Whest Smoke Test")
print("=" * 70)

with we.BudgetContext(flop_budget=10**9) as budget:
    # ── Test data ──────────────────────────────────────────────────────
    SMALL_POS = we.array([0.5, 1.0, 1.5, 2.0])
    SMALL_UNIT = we.array([0.1, 0.3, 0.5, 0.7])
    SMALL_NEG = we.array([-2.0, -1.0, 1.0, 2.0])
    SMALL_INT = we.array([1.0, 2.0, 3.0, 4.0])
    PAIR_A = we.array([1.0, 2.0, 3.0])
    PAIR_B = we.array([4.0, 5.0, 6.0])
    MATRIX = we.array([[1.0, 2.0], [3.0, 4.0]])
    MATRIX3x2 = we.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    VEC2 = we.array([1.0, 2.0])
    VEC3 = we.array([1.0, 2.0, 3.0])
    BOOL_ARR = we.greater(PAIR_A, we.array([2.0, 2.0, 2.0]))
    SMALL_GE1 = we.array([1.0, 1.5, 2.0, 3.0])  # for arccosh
    INT_ARR = we.array([1, 2, 3, 4], dtype="int64")
    INT_PAIR_A = we.array([6, 12, 15], dtype="int64")
    INT_PAIR_B = we.array([4, 8, 10], dtype="int64")
    COMPLEX_ARR = we.array([1.0 + 2.0j, 3.0 - 1.0j], dtype="complex128")
    ZERO_TRIMMED = we.array([0.0, 1.0, 2.0, 0.0])

    # ── 1. COUNTED UNARY ops ──────────────────────────────────────────
    print("\n--- Counted Unary Ops ---")

    # Standard unary (work on general positive floats)
    for name in [
        "exp",
        "exp2",
        "expm1",
        "sqrt",
        "square",
        "cbrt",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "arctan",
        "arcsinh",
        "sign",
        "ceil",
        "floor",
        "abs",
        "absolute",
        "fabs",
        "negative",
        "positive",
        "rint",
        "round",
        "around",
        "fix",
        "trunc",
        "deg2rad",
        "degrees",
        "rad2deg",
        "radians",
        "log",
        "log2",
        "log10",
        "log1p",
        "reciprocal",
        "signbit",
        "spacing",
        "sinc",
        "i0",
        "nan_to_num",
        "real",
        "imag",
        "conj",
        "conjugate",
        "iscomplex",
        "isreal",
        "real_if_close",
        "logical_not",
    ]:
        test(name, getattr(we, name), SMALL_POS)

    # arcsin, arccos, asin, acos need [-1,1]
    for name in ["arcsin", "arccos", "asin", "acos"]:
        test(name, getattr(we, name), SMALL_UNIT)

    # arctanh, atanh need (-1,1)
    for name in ["arctanh", "atanh", "atan"]:
        test(name, getattr(we, name), SMALL_UNIT)

    # acosh, arccosh need >= 1
    for name in ["acosh", "arccosh"]:
        test(name, getattr(we, name), SMALL_GE1)

    # Multi-output unary
    test("modf", we.modf, SMALL_POS)
    test("frexp", we.frexp, SMALL_POS)

    # sort_complex
    test("sort_complex", we.sort_complex, COMPLEX_ARR)

    # isclose (binary signature but counted_unary category)
    test("isclose", we.isclose, PAIR_A, PAIR_B)

    # angle (works on complex)
    test("angle", we.angle, COMPLEX_ARR)

    # iscomplexobj, isrealobj work on any array
    test("iscomplexobj", we.iscomplexobj, SMALL_POS)
    test("isrealobj", we.isrealobj, SMALL_POS)

    # bitwise unary on int arrays
    test("bitwise_invert", we.bitwise_invert, INT_ARR)
    test("bitwise_not", we.bitwise_not, INT_ARR)
    test("bitwise_count", we.bitwise_count, INT_ARR)
    test("invert", we.invert, INT_ARR)

    # isneginf, isposinf
    test("isneginf", we.isneginf, SMALL_NEG)
    test("isposinf", we.isposinf, SMALL_POS)

    # isnat needs datetime - skip or test with float (will produce all False)
    skip("isnat", "requires datetime array")

    # ── 2. COUNTED BINARY ops ─────────────────────────────────────────
    print("\n--- Counted Binary Ops ---")

    for name in [
        "add",
        "subtract",
        "multiply",
        "divide",
        "true_divide",
        "floor_divide",
        "power",
        "pow",
        "float_power",
        "mod",
        "remainder",
        "fmod",
        "maximum",
        "minimum",
        "fmax",
        "fmin",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "equal",
        "not_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "logaddexp",
        "logaddexp2",
        "arctan2",
        "atan2",
        "hypot",
        "copysign",
        "nextafter",
        "heaviside",
    ]:
        test(name, getattr(we, name), PAIR_A, PAIR_B)

    # ldexp: x * 2^i
    test("ldexp", we.ldexp, PAIR_A, we.array([1, 2, 3], dtype="int64"))

    # bitwise binary on int arrays
    for name in [
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "left_shift",
        "right_shift",
    ]:
        test(name, getattr(we, name), INT_PAIR_A, we.array([1, 2, 1], dtype="int64"))

    # gcd, lcm
    test("gcd", we.gcd, INT_PAIR_A, INT_PAIR_B)
    test("lcm", we.lcm, INT_PAIR_A, INT_PAIR_B)

    # divmod (multi-output binary)
    test("divmod", we.divmod, PAIR_A, PAIR_B)

    # vecdot
    test("vecdot", we.vecdot, PAIR_A, PAIR_B)

    # ── 3. COUNTED REDUCTION ops ──────────────────────────────────────
    print("\n--- Counted Reduction Ops ---")

    for name in [
        "sum",
        "prod",
        "mean",
        "std",
        "var",
        "max",
        "min",
        "amax",
        "amin",
        "all",
        "any",
        "argmax",
        "argmin",
        "cumsum",
        "cumprod",
        "count_nonzero",
        "median",
        "nansum",
        "nanprod",
        "nanmean",
        "nanstd",
        "nanvar",
        "nanmax",
        "nanmin",
        "nanmedian",
        "nanargmax",
        "nanargmin",
        "nancumprod",
        "nancumsum",
        "ptp",
    ]:
        test(name, getattr(we, name), SMALL_INT)

    # average
    test("average", we.average, SMALL_INT)

    # percentile / quantile
    test("percentile", we.percentile, SMALL_INT, 50)
    test("quantile", we.quantile, SMALL_INT, 0.5)
    test("nanpercentile", we.nanpercentile, SMALL_INT, 50)
    test("nanquantile", we.nanquantile, SMALL_INT, 0.5)

    # cumulative_sum, cumulative_prod
    test("cumulative_sum", we.cumulative_sum, SMALL_INT, axis=0)
    test("cumulative_prod", we.cumulative_prod, SMALL_INT, axis=0)

    # ── 4. COUNTED CUSTOM ops ─────────────────────────────────────────
    print("\n--- Counted Custom Ops ---")

    # clip
    test("clip", we.clip, SMALL_INT, 1.5, 3.5)

    # dot, matmul
    test("dot", we.dot, PAIR_A, PAIR_B)
    test("matmul", we.matmul, MATRIX, VEC2)

    # inner, outer, vdot
    test("inner", we.inner, PAIR_A, PAIR_B)
    test("outer", we.outer, PAIR_A, PAIR_B)
    test("vdot", we.vdot, PAIR_A, PAIR_B)

    # tensordot
    test("tensordot", we.tensordot, MATRIX, MATRIX, 1)

    # kron
    test("kron", we.kron, VEC2, VEC3)

    # cross (3-d vectors)
    test("cross", we.cross, VEC3, we.array([4.0, 5.0, 6.0]))

    # diff, ediff1d
    test("diff", we.diff, SMALL_INT)
    test("ediff1d", we.ediff1d, SMALL_INT)

    # gradient
    test("gradient", we.gradient, SMALL_INT)

    # convolve, correlate
    test("convolve", we.convolve, PAIR_A, VEC2)
    test("correlate", we.correlate, PAIR_A, VEC2)

    # corrcoef, cov
    test("corrcoef", we.corrcoef, SMALL_INT)
    test("cov", we.cov, SMALL_INT)

    # einsum
    test("einsum", we.einsum, "ij,jk->ik", MATRIX, MATRIX)

    # einsum_path
    test("einsum_path", we.einsum_path, "ij,jk->ik", MATRIX, MATRIX)

    # trapezoid, trapz
    test("trapezoid", we.trapezoid, SMALL_INT)
    test("trapz", we.trapz, SMALL_INT)

    # interp
    test(
        "interp",
        we.interp,
        we.array([1.5, 2.5]),
        we.array([1.0, 2.0, 3.0]),
        we.array([10.0, 20.0, 30.0]),
    )

    # linalg.svd
    test("linalg.svd", we.linalg.svd, MATRIX3x2)

    # ── 5. FREE ops ───────────────────────────────────────────────────
    print("\n--- Free Ops: Tensor Creation ---")

    test("array", we.array, [1.0, 2.0, 3.0])
    test("zeros", we.zeros, (3, 3))
    test("ones", we.ones, (3, 3))
    test("full", we.full, (3, 3), 7.0)
    test("empty", we.empty, (3, 3))
    test("eye", we.eye, 3)
    test("identity", we.identity, 3)
    test("arange", we.arange, 0, 10, 2)
    test("linspace", we.linspace, 0, 1, 5)
    test("logspace", we.logspace, 0, 2, 5)
    test("geomspace", we.geomspace, 1, 100, 5)
    test("zeros_like", we.zeros_like, SMALL_INT)
    test("ones_like", we.ones_like, SMALL_INT)
    test("full_like", we.full_like, SMALL_INT, 9.0)
    test("empty_like", we.empty_like, SMALL_INT)
    test("diag", we.diag, VEC3)
    test("diagflat", we.diagflat, VEC2)
    test("tri", we.tri, 3, 3)
    test("tril", we.tril, MATRIX)
    test("triu", we.triu, MATRIX)
    test("vander", we.vander, PAIR_A, 3)

    print("\n--- Free Ops: Tensor Manipulation ---")

    test("reshape", we.reshape, SMALL_INT, (2, 2))
    test("transpose", we.transpose, MATRIX)
    test("swapaxes", we.swapaxes, MATRIX, 0, 1)
    test("moveaxis", we.moveaxis, MATRIX, 0, 1)
    test("concatenate", we.concatenate, [PAIR_A, PAIR_B])
    test("stack", we.stack, [PAIR_A, PAIR_B])
    test("vstack", we.vstack, [PAIR_A, PAIR_B])
    test("hstack", we.hstack, [PAIR_A, PAIR_B])
    test("column_stack", we.column_stack, [PAIR_A, PAIR_B])
    test("row_stack", we.row_stack, [PAIR_A, PAIR_B])
    test("dstack", we.dstack, [PAIR_A, PAIR_B])
    test("split", we.split, we.array([1.0, 2.0, 3.0, 4.0]), 2)
    test("hsplit", we.hsplit, we.array([1.0, 2.0, 3.0, 4.0]), 2)
    test("vsplit", we.vsplit, we.array([[1.0, 2.0], [3.0, 4.0]]), 2)
    test("dsplit", we.dsplit, we.array([[[1.0, 2.0], [3.0, 4.0]]]), 2)
    test("array_split", we.array_split, SMALL_INT, 2)
    test("squeeze", we.squeeze, we.array([[[1.0, 2.0]]]))
    test("expand_dims", we.expand_dims, PAIR_A, 0)
    test("ravel", we.ravel, MATRIX)
    test("copy", we.copy, PAIR_A)
    test("flip", we.flip, PAIR_A)
    test("fliplr", we.fliplr, MATRIX)
    test("flipud", we.flipud, MATRIX)
    test("rot90", we.rot90, MATRIX)
    test("roll", we.roll, PAIR_A, 1)
    test("rollaxis", we.rollaxis, MATRIX, 1)
    test("tile", we.tile, PAIR_A, 2)
    test("repeat", we.repeat, PAIR_A, 2)
    test("resize", we.resize, PAIR_A, (2, 3))
    test("append", we.append, PAIR_A, PAIR_B)
    test("insert", we.insert, PAIR_A, 1, 99.0)
    test("delete", we.delete, PAIR_A, 1)
    test("unique", we.unique, we.array([3.0, 1.0, 2.0, 1.0]))
    test("trim_zeros", we.trim_zeros, ZERO_TRIMMED)
    test("sort", we.sort, we.array([3.0, 1.0, 2.0]))
    test("argsort", we.argsort, we.array([3.0, 1.0, 2.0]))
    test("partition", we.partition, we.array([3.0, 1.0, 2.0, 4.0]), 2)
    test("argpartition", we.argpartition, we.array([3.0, 1.0, 2.0, 4.0]), 2)
    test("take", we.take, PAIR_A, [0, 2])
    test(
        "take_along_axis",
        we.take_along_axis,
        we.array([[1.0, 2.0], [3.0, 4.0]]),
        we.array([[0, 1], [1, 0]], dtype="int64"),
        axis=1,
    )
    test("compress", we.compress, [True, False, True], PAIR_A)
    test("extract", we.extract, BOOL_ARR, PAIR_A)
    test("diagonal", we.diagonal, MATRIX)
    test("trace", we.trace, MATRIX)
    test("pad", we.pad, PAIR_A, 1)
    test("searchsorted", we.searchsorted, we.array([1.0, 3.0, 5.0]), 2.0)

    print("\n--- Free Ops: Where / Select / Place ---")

    test("where", we.where, BOOL_ARR, PAIR_A, PAIR_B)
    test("select", we.select, [BOOL_ARR], [PAIR_A])
    test("nonzero", we.nonzero, we.array([0.0, 1.0, 0.0, 2.0]))
    test("argwhere", we.argwhere, we.array([0.0, 1.0, 0.0, 2.0]))
    test("flatnonzero", we.flatnonzero, we.array([0.0, 1.0, 0.0, 2.0]))
    test("isin", we.isin, PAIR_A, [1.0, 3.0])
    test("in1d", we.in1d, PAIR_A, we.array([1.0, 3.0]))

    print("\n--- Free Ops: Set ops ---")

    test("intersect1d", we.intersect1d, PAIR_A, we.array([2.0, 3.0, 7.0]))
    test("union1d", we.union1d, PAIR_A, we.array([2.0, 5.0]))
    test("setdiff1d", we.setdiff1d, PAIR_A, we.array([2.0]))
    test("setxor1d", we.setxor1d, PAIR_A, we.array([2.0, 5.0]))

    print("\n--- Free Ops: Comparison / Type ---")

    test("allclose", we.allclose, PAIR_A, PAIR_A)
    test("array_equal", we.array_equal, PAIR_A, PAIR_A)
    test("array_equiv", we.array_equiv, PAIR_A, PAIR_A)
    test("isfinite", we.isfinite, SMALL_POS)
    test("isinf", we.isinf, SMALL_POS)
    test("isnan", we.isnan, SMALL_POS)
    test("isscalar", we.isscalar, 5.0)

    print("\n--- Free Ops: Shape / Type Introspection ---")

    test("shape", we.shape, MATRIX)
    test("ndim", we.ndim, MATRIX)
    test("size", we.size, MATRIX)

    print("\n--- Free Ops: Broadcast ---")

    test("broadcast_to", we.broadcast_to, PAIR_A, (2, 3))
    test("broadcast_arrays", we.broadcast_arrays, PAIR_A, PAIR_B)
    test("broadcast_shapes", we.broadcast_shapes, (3,), (3,))

    print("\n--- Free Ops: Histogram ---")

    test("histogram", we.histogram, SMALL_INT, 3)
    test("histogram_bin_edges", we.histogram_bin_edges, SMALL_INT, 3)

    print("\n--- Free Ops: Type Conversion / Casting ---")

    test("asarray", we.asarray, PAIR_A)
    test("asarray_chkfinite", we.asarray_chkfinite, PAIR_A)
    test("astype", we.astype, PAIR_A, "float32")
    test("can_cast", we.can_cast, "float32", "float64")
    test("result_type", we.result_type, "float32", "float64")
    test("promote_types", we.promote_types, "float32", "float64")

    print("\n--- Free Ops: Atleast ---")

    test("atleast_1d", we.atleast_1d, PAIR_A)
    test("atleast_2d", we.atleast_2d, PAIR_A)
    test("atleast_3d", we.atleast_3d, PAIR_A)

    print("\n--- Free Ops: Index Helpers ---")

    test("diag_indices", we.diag_indices, 3)
    test("diag_indices_from", we.diag_indices_from, MATRIX)
    test("tril_indices", we.tril_indices, 3)
    test("triu_indices", we.triu_indices, 3)
    test("tril_indices_from", we.tril_indices_from, MATRIX)
    test("triu_indices_from", we.triu_indices_from, MATRIX)
    test("indices", we.indices, (2, 3))
    test(
        "ix_", we.ix_, we.array([0, 1], dtype="int64"), we.array([0, 1], dtype="int64")
    )
    test("unravel_index", we.unravel_index, 5, (3, 3))
    test(
        "ravel_multi_index",
        we.ravel_multi_index,
        (we.array([0, 1, 2], dtype="int64"), we.array([0, 1, 2], dtype="int64")),
        (3, 3),
    )
    test("mask_indices", we.mask_indices, 3, we.triu)

    print("\n--- Free Ops: Mesh / Grid ---")

    test("meshgrid", we.meshgrid, we.array([1.0, 2.0]), we.array([3.0, 4.0]))

    print("\n--- Free Ops: Misc ---")

    test("lexsort", we.lexsort, (we.array([1.0, 2.0, 1.0]), we.array([3.0, 1.0, 2.0])))
    test("digitize", we.digitize, we.array([0.5, 1.5, 2.5]), we.array([1.0, 2.0, 3.0]))
    test("bincount", we.bincount, we.array([0, 1, 1, 2, 3, 3, 3], dtype="int64"))
    test("packbits", we.packbits, we.array([1, 0, 1, 1, 0, 0, 0, 1], dtype="uint8"))
    test("unpackbits", we.unpackbits, we.array([177], dtype="uint8"))

    test("block", we.block, [[PAIR_A, PAIR_B]])
    test("concat", we.concat, [PAIR_A, PAIR_B])

    print("\n--- Free Ops: Misc Introspection ---")

    test("iterable", we.iterable, PAIR_A)
    test("isfortran", we.isfortran, MATRIX)
    test("typename", we.typename, "float64")
    test("mintypecode", we.mintypecode, ["f", "d"])
    test("base_repr", we.base_repr, 10, 2)
    test("binary_repr", we.binary_repr, 10)

    print("\n--- Free Ops: Put / Place / Copyto ---")

    # put, place, putmask, fill_diagonal, copyto mutate arrays
    arr_put = we.array([1.0, 2.0, 3.0, 4.0])
    test("put", we.put, arr_put, [0, 2], [99.0, 88.0])

    arr_place = we.array([1.0, 2.0, 3.0, 4.0])
    test(
        "place",
        we.place,
        arr_place,
        we.array([True, False, True, False], dtype="bool"),
        [99.0, 88.0],
    )

    arr_pm = we.array([1.0, 2.0, 3.0, 4.0])
    test(
        "putmask",
        we.putmask,
        arr_pm,
        we.array([True, False, True, False], dtype="bool"),
        0.0,
    )

    mat_fd = we.array([[1.0, 2.0], [3.0, 4.0]])
    test("fill_diagonal", we.fill_diagonal, mat_fd, 0.0)

    arr_ct = we.array([1.0, 2.0, 3.0])
    test("copyto", we.copyto, arr_ct, we.array([9.0, 8.0, 7.0]))

    print("\n--- Free Ops: Choose / Require ---")

    test(
        "choose",
        we.choose,
        we.array([0, 1, 0], dtype="int64"),
        [we.array([10.0, 20.0, 30.0]), we.array([40.0, 50.0, 60.0])],
    )
    test("require", we.require, PAIR_A)

    print("\n--- Free Ops: Matrix Transpose / Permute ---")

    test("matrix_transpose", we.matrix_transpose, we.array([[[1.0, 2.0], [3.0, 4.0]]]))
    test("permute_dims", we.permute_dims, MATRIX, (1, 0))

    print("\n--- Free Ops: Unique Variants ---")

    test("unique_all", we.unique_all, we.array([3.0, 1.0, 2.0, 1.0]))
    test("unique_counts", we.unique_counts, we.array([3.0, 1.0, 2.0, 1.0]))
    test("unique_inverse", we.unique_inverse, we.array([3.0, 1.0, 2.0, 1.0]))
    test("unique_values", we.unique_values, we.array([3.0, 1.0, 2.0, 1.0]))

    print("\n--- Free Ops: Unstack ---")

    test("unstack", we.unstack, MATRIX)

    print("\n--- Free Ops: Memory Overlap ---")

    test("shares_memory", we.shares_memory, PAIR_A, PAIR_B)
    test("may_share_memory", we.may_share_memory, PAIR_A, PAIR_B)

    print("\n--- Free Ops: Misc Type ---")

    test("min_scalar_type", we.min_scalar_type, 10)
    test("issubdtype", we.issubdtype, "float64", "float64")
    test("common_type", we.common_type, PAIR_A)

    print("\n--- Free Ops: Put Along Axis ---")

    test(
        "put_along_axis",
        we.put_along_axis,
        we.array([[1.0, 2.0], [3.0, 4.0]]),
        we.array([[0], [1]], dtype="int64"),
        we.array([[99.0], [88.0]]),
        axis=1,
    )

    print("\n--- Free Ops: Histogram 2D / DD ---")

    test(
        "histogram2d",
        we.histogram2d,
        we.array([1.0, 2.0, 3.0]),
        we.array([4.0, 5.0, 6.0]),
        3,
    )
    test(
        "histogramdd", we.histogramdd, we.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 3
    )

    # ── 6. RANDOM submodule ───────────────────────────────────────────
    print("\n--- Random Submodule ---")

    test("random.rand", we.random.rand, 3, 3)
    test("random.randn", we.random.randn, 3, 3)
    test("random.random", we.random.random, (3,))
    test("random.randint", we.random.randint, 0, 10, (3,))
    test("random.uniform", we.random.uniform, 0.0, 1.0, (3,))
    test("random.normal", we.random.normal, 0.0, 1.0, (3,))
    test("random.seed", we.random.seed, 42)
    test("random.choice", we.random.choice, 10, 3)
    test("random.permutation", we.random.permutation, 5)
    test("random.beta", we.random.beta, 2.0, 5.0, (3,))
    test("random.binomial", we.random.binomial, 10, 0.5, (3,))
    test("random.chisquare", we.random.chisquare, 2, (3,))
    test("random.exponential", we.random.exponential, 1.0, (3,))
    test("random.gamma", we.random.gamma, 2.0, 1.0, (3,))
    test("random.geometric", we.random.geometric, 0.5, (3,))
    test("random.gumbel", we.random.gumbel, 0.0, 1.0, (3,))
    test("random.laplace", we.random.laplace, 0.0, 1.0, (3,))
    test("random.logistic", we.random.logistic, 0.0, 1.0, (3,))
    test("random.lognormal", we.random.lognormal, 0.0, 1.0, (3,))
    test("random.logseries", we.random.logseries, 0.9, (3,))
    test("random.multinomial", we.random.multinomial, 10, [0.5, 0.3, 0.2])
    test("random.negative_binomial", we.random.negative_binomial, 5, 0.5, (3,))
    test("random.noncentral_chisquare", we.random.noncentral_chisquare, 2, 1.0, (3,))
    test("random.noncentral_f", we.random.noncentral_f, 5, 10, 1.0, (3,))
    test("random.pareto", we.random.pareto, 2.0, (3,))
    test("random.poisson", we.random.poisson, 5.0, (3,))
    test("random.power", we.random.power, 2.0, (3,))
    test("random.rayleigh", we.random.rayleigh, 1.0, (3,))
    test("random.standard_cauchy", we.random.standard_cauchy, (3,))
    test("random.standard_exponential", we.random.standard_exponential, (3,))
    test("random.standard_gamma", we.random.standard_gamma, 2.0, (3,))
    test("random.standard_normal", we.random.standard_normal, (3,))
    test("random.standard_t", we.random.standard_t, 5.0, (3,))
    test("random.triangular", we.random.triangular, 0.0, 0.5, 1.0, (3,))
    test("random.vonmises", we.random.vonmises, 0.0, 1.0, (3,))
    test("random.wald", we.random.wald, 1.0, 1.0, (3,))
    test("random.weibull", we.random.weibull, 2.0, (3,))
    test("random.zipf", we.random.zipf, 2.0, (3,))
    test("random.dirichlet", we.random.dirichlet, [1.0, 1.0, 1.0])
    test(
        "random.multivariate_normal",
        we.random.multivariate_normal,
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    test("random.f", we.random.f, 5, 10, (3,))
    test("random.hypergeometric", we.random.hypergeometric, 10, 5, 7, (3,))
    test("random.random_sample", we.random.random_sample, (3,))
    test("random.ranf", we.random.ranf, (3,))
    test("random.sample", we.random.sample, (3,))

    # random state ops
    test("random.get_state", we.random.get_state)

    # shuffle mutates in place
    arr_shuf = we.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test("random.shuffle", we.random.shuffle, arr_shuf)

    # Skip: random.set_state (needs valid state), random.bytes (returns bytes),
    #        random.random_integers (deprecated), random.default_rng
    skip("random.set_state", "needs valid state object")
    skip("random.bytes", "returns raw bytes, not array")
    skip("random.random_integers", "deprecated in numpy")
    skip("random.default_rng", "returns generator object, not array")

    # ── 7. Skipped functions ──────────────────────────────────────────
    print("\n--- Skipped Functions ---")

    # Functions that read/write files or need special environments
    for name in [
        "fromfile",
        "fromstring",
        "frombuffer",
        "fromfunction",
        "fromiter",
        "fromregex",
        "from_dlpack",
        "bmat",
        "isdtype",
    ]:
        skip(name, "requires special input type or file I/O")

    # These are hard to test generically
    skip("histogram2d (already tested)", "already tested above")
    skip("select (already tested)", "already tested above")

# ── Summary ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  EXHAUSTIVE TEST SUMMARY")
print("=" * 70)
print(f"  Passed:  {len(results['pass'])}")
print(f"  Failed:  {len(results['fail'])}")
print(f"  Skipped: {len(results['skip'])}")
print()

if results["fail"]:
    print("  FAILURES:")
    for name, err in results["fail"]:
        print(f"    {name}: {err}")
    print()

if results["skip"]:
    print("  SKIPPED:")
    for name, reason in results["skip"]:
        print(f"    {name}: {reason}")
    print()

total = len(results["pass"]) + len(results["fail"])
print(f"  Total functions called: {total}")
if results["fail"]:
    print(f"  RESULT: {len(results['fail'])} FAILURES")
    sys.exit(1)
else:
    print("  RESULT: ALL PASSED")
    sys.exit(0)
