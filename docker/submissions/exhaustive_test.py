"""
Exhaustive smoke test: calls EVERY non-blacklisted function in the mechestim
registry through the client-server proxy to verify it works end-to-end.

Runs INSIDE the participant container (no numpy).
"""
import mechestim as me
import sys
import traceback

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
print("  Exhaustive Mechestim Smoke Test")
print("=" * 70)

with me.BudgetContext(flop_budget=10**9) as budget:

    # ── Test data ──────────────────────────────────────────────────────
    SMALL_POS = me.array([0.5, 1.0, 1.5, 2.0])
    SMALL_UNIT = me.array([0.1, 0.3, 0.5, 0.7])
    SMALL_NEG = me.array([-2.0, -1.0, 1.0, 2.0])
    SMALL_INT = me.array([1.0, 2.0, 3.0, 4.0])
    PAIR_A = me.array([1.0, 2.0, 3.0])
    PAIR_B = me.array([4.0, 5.0, 6.0])
    MATRIX = me.array([[1.0, 2.0], [3.0, 4.0]])
    MATRIX3x2 = me.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    VEC2 = me.array([1.0, 2.0])
    VEC3 = me.array([1.0, 2.0, 3.0])
    BOOL_ARR = me.greater(PAIR_A, me.array([2.0, 2.0, 2.0]))
    SMALL_GE1 = me.array([1.0, 1.5, 2.0, 3.0])  # for arccosh
    INT_ARR = me.array([1, 2, 3, 4], dtype="int64")
    INT_PAIR_A = me.array([6, 12, 15], dtype="int64")
    INT_PAIR_B = me.array([4, 8, 10], dtype="int64")
    COMPLEX_ARR = me.array([1.0 + 2.0j, 3.0 - 1.0j], dtype="complex128")
    ZERO_TRIMMED = me.array([0.0, 1.0, 2.0, 0.0])

    # ── 1. COUNTED UNARY ops ──────────────────────────────────────────
    print("\n--- Counted Unary Ops ---")

    # Standard unary (work on general positive floats)
    for name in ["exp", "exp2", "expm1", "sqrt", "square", "cbrt",
                  "sin", "cos", "tan", "sinh", "cosh", "tanh",
                  "arctan", "arcsinh",
                  "sign", "ceil", "floor", "abs", "absolute", "fabs",
                  "negative", "positive", "rint", "round", "around",
                  "fix", "trunc", "deg2rad", "degrees", "rad2deg", "radians",
                  "log", "log2", "log10", "log1p",
                  "reciprocal", "signbit", "spacing",
                  "sinc", "i0", "nan_to_num", "real", "imag",
                  "conj", "conjugate", "iscomplex", "isreal",
                  "real_if_close", "logical_not"]:
        test(name, getattr(me, name), SMALL_POS)

    # arcsin, arccos, asin, acos need [-1,1]
    for name in ["arcsin", "arccos", "asin", "acos"]:
        test(name, getattr(me, name), SMALL_UNIT)

    # arctanh, atanh need (-1,1)
    for name in ["arctanh", "atanh", "atan"]:
        test(name, getattr(me, name), SMALL_UNIT)

    # acosh, arccosh need >= 1
    for name in ["acosh", "arccosh"]:
        test(name, getattr(me, name), SMALL_GE1)

    # Multi-output unary
    test("modf", me.modf, SMALL_POS)
    test("frexp", me.frexp, SMALL_POS)

    # sort_complex
    test("sort_complex", me.sort_complex, COMPLEX_ARR)

    # isclose (binary signature but counted_unary category)
    test("isclose", me.isclose, PAIR_A, PAIR_B)

    # angle (works on complex)
    test("angle", me.angle, COMPLEX_ARR)

    # iscomplexobj, isrealobj work on any array
    test("iscomplexobj", me.iscomplexobj, SMALL_POS)
    test("isrealobj", me.isrealobj, SMALL_POS)

    # bitwise unary on int arrays
    test("bitwise_invert", me.bitwise_invert, INT_ARR)
    test("bitwise_not", me.bitwise_not, INT_ARR)
    test("bitwise_count", me.bitwise_count, INT_ARR)
    test("invert", me.invert, INT_ARR)

    # isneginf, isposinf
    test("isneginf", me.isneginf, SMALL_NEG)
    test("isposinf", me.isposinf, SMALL_POS)

    # isnat needs datetime - skip or test with float (will produce all False)
    skip("isnat", "requires datetime array")

    # ── 2. COUNTED BINARY ops ─────────────────────────────────────────
    print("\n--- Counted Binary Ops ---")

    for name in ["add", "subtract", "multiply", "divide", "true_divide",
                  "floor_divide", "power", "pow", "float_power",
                  "mod", "remainder", "fmod",
                  "maximum", "minimum", "fmax", "fmin",
                  "greater", "greater_equal", "less", "less_equal",
                  "equal", "not_equal",
                  "logical_and", "logical_or", "logical_xor",
                  "logaddexp", "logaddexp2",
                  "arctan2", "atan2", "hypot", "copysign",
                  "nextafter", "heaviside"]:
        test(name, getattr(me, name), PAIR_A, PAIR_B)

    # ldexp: x * 2^i
    test("ldexp", me.ldexp, PAIR_A, me.array([1, 2, 3], dtype="int64"))

    # bitwise binary on int arrays
    for name in ["bitwise_and", "bitwise_or", "bitwise_xor",
                  "bitwise_left_shift", "bitwise_right_shift",
                  "left_shift", "right_shift"]:
        test(name, getattr(me, name), INT_PAIR_A, me.array([1, 2, 1], dtype="int64"))

    # gcd, lcm
    test("gcd", me.gcd, INT_PAIR_A, INT_PAIR_B)
    test("lcm", me.lcm, INT_PAIR_A, INT_PAIR_B)

    # divmod (multi-output binary)
    test("divmod", me.divmod, PAIR_A, PAIR_B)

    # vecdot
    test("vecdot", me.vecdot, PAIR_A, PAIR_B)

    # ── 3. COUNTED REDUCTION ops ──────────────────────────────────────
    print("\n--- Counted Reduction Ops ---")

    for name in ["sum", "prod", "mean", "std", "var", "max", "min",
                  "amax", "amin", "all", "any",
                  "argmax", "argmin", "cumsum", "cumprod",
                  "count_nonzero", "median",
                  "nansum", "nanprod", "nanmean", "nanstd", "nanvar",
                  "nanmax", "nanmin", "nanmedian",
                  "nanargmax", "nanargmin",
                  "nancumprod", "nancumsum",
                  "ptp"]:
        test(name, getattr(me, name), SMALL_INT)

    # average
    test("average", me.average, SMALL_INT)

    # percentile / quantile
    test("percentile", me.percentile, SMALL_INT, 50)
    test("quantile", me.quantile, SMALL_INT, 0.5)
    test("nanpercentile", me.nanpercentile, SMALL_INT, 50)
    test("nanquantile", me.nanquantile, SMALL_INT, 0.5)

    # cumulative_sum, cumulative_prod
    test("cumulative_sum", me.cumulative_sum, SMALL_INT, axis=0)
    test("cumulative_prod", me.cumulative_prod, SMALL_INT, axis=0)

    # ── 4. COUNTED CUSTOM ops ─────────────────────────────────────────
    print("\n--- Counted Custom Ops ---")

    # clip
    test("clip", me.clip, SMALL_INT, 1.5, 3.5)

    # dot, matmul
    test("dot", me.dot, PAIR_A, PAIR_B)
    test("matmul", me.matmul, MATRIX, VEC2)

    # inner, outer, vdot
    test("inner", me.inner, PAIR_A, PAIR_B)
    test("outer", me.outer, PAIR_A, PAIR_B)
    test("vdot", me.vdot, PAIR_A, PAIR_B)

    # tensordot
    test("tensordot", me.tensordot, MATRIX, MATRIX, 1)

    # kron
    test("kron", me.kron, VEC2, VEC3)

    # cross (3-d vectors)
    test("cross", me.cross, VEC3, me.array([4.0, 5.0, 6.0]))

    # diff, ediff1d
    test("diff", me.diff, SMALL_INT)
    test("ediff1d", me.ediff1d, SMALL_INT)

    # gradient
    test("gradient", me.gradient, SMALL_INT)

    # convolve, correlate
    test("convolve", me.convolve, PAIR_A, VEC2)
    test("correlate", me.correlate, PAIR_A, VEC2)

    # corrcoef, cov
    test("corrcoef", me.corrcoef, SMALL_INT)
    test("cov", me.cov, SMALL_INT)

    # einsum
    test("einsum", me.einsum, "ij,jk->ik", MATRIX, MATRIX)

    # einsum_path
    test("einsum_path", me.einsum_path, "ij,jk->ik", MATRIX, MATRIX)

    # trapezoid, trapz
    test("trapezoid", me.trapezoid, SMALL_INT)
    test("trapz", me.trapz, SMALL_INT)

    # interp
    test("interp", me.interp,
         me.array([1.5, 2.5]),
         me.array([1.0, 2.0, 3.0]),
         me.array([10.0, 20.0, 30.0]))

    # linalg.svd
    test("linalg.svd", me.linalg.svd, MATRIX3x2)

    # ── 5. FREE ops ───────────────────────────────────────────────────
    print("\n--- Free Ops: Tensor Creation ---")

    test("array", me.array, [1.0, 2.0, 3.0])
    test("zeros", me.zeros, (3, 3))
    test("ones", me.ones, (3, 3))
    test("full", me.full, (3, 3), 7.0)
    test("empty", me.empty, (3, 3))
    test("eye", me.eye, 3)
    test("identity", me.identity, 3)
    test("arange", me.arange, 0, 10, 2)
    test("linspace", me.linspace, 0, 1, 5)
    test("logspace", me.logspace, 0, 2, 5)
    test("geomspace", me.geomspace, 1, 100, 5)
    test("zeros_like", me.zeros_like, SMALL_INT)
    test("ones_like", me.ones_like, SMALL_INT)
    test("full_like", me.full_like, SMALL_INT, 9.0)
    test("empty_like", me.empty_like, SMALL_INT)
    test("diag", me.diag, VEC3)
    test("diagflat", me.diagflat, VEC2)
    test("tri", me.tri, 3, 3)
    test("tril", me.tril, MATRIX)
    test("triu", me.triu, MATRIX)
    test("vander", me.vander, PAIR_A, 3)

    print("\n--- Free Ops: Tensor Manipulation ---")

    test("reshape", me.reshape, SMALL_INT, (2, 2))
    test("transpose", me.transpose, MATRIX)
    test("swapaxes", me.swapaxes, MATRIX, 0, 1)
    test("moveaxis", me.moveaxis, MATRIX, 0, 1)
    test("concatenate", me.concatenate, [PAIR_A, PAIR_B])
    test("stack", me.stack, [PAIR_A, PAIR_B])
    test("vstack", me.vstack, [PAIR_A, PAIR_B])
    test("hstack", me.hstack, [PAIR_A, PAIR_B])
    test("column_stack", me.column_stack, [PAIR_A, PAIR_B])
    test("row_stack", me.row_stack, [PAIR_A, PAIR_B])
    test("dstack", me.dstack, [PAIR_A, PAIR_B])
    test("split", me.split, me.array([1.0, 2.0, 3.0, 4.0]), 2)
    test("hsplit", me.hsplit, me.array([1.0, 2.0, 3.0, 4.0]), 2)
    test("vsplit", me.vsplit, me.array([[1.0, 2.0], [3.0, 4.0]]), 2)
    test("dsplit", me.dsplit, me.array([[[1.0, 2.0], [3.0, 4.0]]]), 2)
    test("array_split", me.array_split, SMALL_INT, 2)
    test("squeeze", me.squeeze, me.array([[[1.0, 2.0]]]))
    test("expand_dims", me.expand_dims, PAIR_A, 0)
    test("ravel", me.ravel, MATRIX)
    test("copy", me.copy, PAIR_A)
    test("flip", me.flip, PAIR_A)
    test("fliplr", me.fliplr, MATRIX)
    test("flipud", me.flipud, MATRIX)
    test("rot90", me.rot90, MATRIX)
    test("roll", me.roll, PAIR_A, 1)
    test("rollaxis", me.rollaxis, MATRIX, 1)
    test("tile", me.tile, PAIR_A, 2)
    test("repeat", me.repeat, PAIR_A, 2)
    test("resize", me.resize, PAIR_A, (2, 3))
    test("append", me.append, PAIR_A, PAIR_B)
    test("insert", me.insert, PAIR_A, 1, 99.0)
    test("delete", me.delete, PAIR_A, 1)
    test("unique", me.unique, me.array([3.0, 1.0, 2.0, 1.0]))
    test("trim_zeros", me.trim_zeros, ZERO_TRIMMED)
    test("sort", me.sort, me.array([3.0, 1.0, 2.0]))
    test("argsort", me.argsort, me.array([3.0, 1.0, 2.0]))
    test("partition", me.partition, me.array([3.0, 1.0, 2.0, 4.0]), 2)
    test("argpartition", me.argpartition, me.array([3.0, 1.0, 2.0, 4.0]), 2)
    test("take", me.take, PAIR_A, [0, 2])
    test("take_along_axis", me.take_along_axis,
         me.array([[1.0, 2.0], [3.0, 4.0]]),
         me.array([[0, 1], [1, 0]], dtype="int64"), axis=1)
    test("compress", me.compress, [True, False, True], PAIR_A)
    test("extract", me.extract, BOOL_ARR, PAIR_A)
    test("diagonal", me.diagonal, MATRIX)
    test("trace", me.trace, MATRIX)
    test("pad", me.pad, PAIR_A, 1)
    test("searchsorted", me.searchsorted, me.array([1.0, 3.0, 5.0]), 2.0)

    print("\n--- Free Ops: Where / Select / Place ---")

    test("where", me.where, BOOL_ARR, PAIR_A, PAIR_B)
    test("select", me.select, [BOOL_ARR], [PAIR_A])
    test("nonzero", me.nonzero, me.array([0.0, 1.0, 0.0, 2.0]))
    test("argwhere", me.argwhere, me.array([0.0, 1.0, 0.0, 2.0]))
    test("flatnonzero", me.flatnonzero, me.array([0.0, 1.0, 0.0, 2.0]))
    test("isin", me.isin, PAIR_A, [1.0, 3.0])
    test("in1d", me.in1d, PAIR_A, me.array([1.0, 3.0]))

    print("\n--- Free Ops: Set ops ---")

    test("intersect1d", me.intersect1d, PAIR_A, me.array([2.0, 3.0, 7.0]))
    test("union1d", me.union1d, PAIR_A, me.array([2.0, 5.0]))
    test("setdiff1d", me.setdiff1d, PAIR_A, me.array([2.0]))
    test("setxor1d", me.setxor1d, PAIR_A, me.array([2.0, 5.0]))

    print("\n--- Free Ops: Comparison / Type ---")

    test("allclose", me.allclose, PAIR_A, PAIR_A)
    test("array_equal", me.array_equal, PAIR_A, PAIR_A)
    test("array_equiv", me.array_equiv, PAIR_A, PAIR_A)
    test("isfinite", me.isfinite, SMALL_POS)
    test("isinf", me.isinf, SMALL_POS)
    test("isnan", me.isnan, SMALL_POS)
    test("isscalar", me.isscalar, 5.0)

    print("\n--- Free Ops: Shape / Type Introspection ---")

    test("shape", me.shape, MATRIX)
    test("ndim", me.ndim, MATRIX)
    test("size", me.size, MATRIX)

    print("\n--- Free Ops: Broadcast ---")

    test("broadcast_to", me.broadcast_to, PAIR_A, (2, 3))
    test("broadcast_arrays", me.broadcast_arrays, PAIR_A, PAIR_B)
    test("broadcast_shapes", me.broadcast_shapes, (3,), (3,))

    print("\n--- Free Ops: Histogram ---")

    test("histogram", me.histogram, SMALL_INT, 3)
    test("histogram_bin_edges", me.histogram_bin_edges, SMALL_INT, 3)

    print("\n--- Free Ops: Type Conversion / Casting ---")

    test("asarray", me.asarray, PAIR_A)
    test("asarray_chkfinite", me.asarray_chkfinite, PAIR_A)
    test("astype", me.astype, PAIR_A, "float32")
    test("can_cast", me.can_cast, "float32", "float64")
    test("result_type", me.result_type, "float32", "float64")
    test("promote_types", me.promote_types, "float32", "float64")

    print("\n--- Free Ops: Atleast ---")

    test("atleast_1d", me.atleast_1d, PAIR_A)
    test("atleast_2d", me.atleast_2d, PAIR_A)
    test("atleast_3d", me.atleast_3d, PAIR_A)

    print("\n--- Free Ops: Index Helpers ---")

    test("diag_indices", me.diag_indices, 3)
    test("diag_indices_from", me.diag_indices_from, MATRIX)
    test("tril_indices", me.tril_indices, 3)
    test("triu_indices", me.triu_indices, 3)
    test("tril_indices_from", me.tril_indices_from, MATRIX)
    test("triu_indices_from", me.triu_indices_from, MATRIX)
    test("indices", me.indices, (2, 3))
    test("ix_", me.ix_, me.array([0, 1], dtype="int64"), me.array([0, 1], dtype="int64"))
    test("unravel_index", me.unravel_index, 5, (3, 3))
    test("ravel_multi_index", me.ravel_multi_index,
         (me.array([0, 1, 2], dtype="int64"), me.array([0, 1, 2], dtype="int64")), (3, 3))
    test("mask_indices", me.mask_indices, 3, me.triu)

    print("\n--- Free Ops: Mesh / Grid ---")

    test("meshgrid", me.meshgrid, me.array([1.0, 2.0]), me.array([3.0, 4.0]))

    print("\n--- Free Ops: Misc ---")

    test("lexsort", me.lexsort, (me.array([1.0, 2.0, 1.0]), me.array([3.0, 1.0, 2.0])))
    test("digitize", me.digitize, me.array([0.5, 1.5, 2.5]), me.array([1.0, 2.0, 3.0]))
    test("bincount", me.bincount, me.array([0, 1, 1, 2, 3, 3, 3], dtype="int64"))
    test("packbits", me.packbits, me.array([1, 0, 1, 1, 0, 0, 0, 1], dtype="uint8"))
    test("unpackbits", me.unpackbits, me.array([177], dtype="uint8"))

    test("block", me.block, [[PAIR_A, PAIR_B]])
    test("concat", me.concat, [PAIR_A, PAIR_B])

    print("\n--- Free Ops: Misc Introspection ---")

    test("iterable", me.iterable, PAIR_A)
    test("isfortran", me.isfortran, MATRIX)
    test("typename", me.typename, "float64")
    test("mintypecode", me.mintypecode, ["f", "d"])
    test("base_repr", me.base_repr, 10, 2)
    test("binary_repr", me.binary_repr, 10)

    print("\n--- Free Ops: Put / Place / Copyto ---")

    # put, place, putmask, fill_diagonal, copyto mutate arrays
    arr_put = me.array([1.0, 2.0, 3.0, 4.0])
    test("put", me.put, arr_put, [0, 2], [99.0, 88.0])

    arr_place = me.array([1.0, 2.0, 3.0, 4.0])
    test("place", me.place, arr_place,
         me.array([True, False, True, False], dtype="bool"), [99.0, 88.0])

    arr_pm = me.array([1.0, 2.0, 3.0, 4.0])
    test("putmask", me.putmask, arr_pm,
         me.array([True, False, True, False], dtype="bool"), 0.0)

    mat_fd = me.array([[1.0, 2.0], [3.0, 4.0]])
    test("fill_diagonal", me.fill_diagonal, mat_fd, 0.0)

    arr_ct = me.array([1.0, 2.0, 3.0])
    test("copyto", me.copyto, arr_ct, me.array([9.0, 8.0, 7.0]))

    print("\n--- Free Ops: Choose / Require ---")

    test("choose", me.choose, me.array([0, 1, 0], dtype="int64"),
         [me.array([10.0, 20.0, 30.0]), me.array([40.0, 50.0, 60.0])])
    test("require", me.require, PAIR_A)

    print("\n--- Free Ops: Matrix Transpose / Permute ---")

    test("matrix_transpose", me.matrix_transpose,
         me.array([[[1.0, 2.0], [3.0, 4.0]]]))
    test("permute_dims", me.permute_dims, MATRIX, (1, 0))

    print("\n--- Free Ops: Unique Variants ---")

    test("unique_all", me.unique_all, me.array([3.0, 1.0, 2.0, 1.0]))
    test("unique_counts", me.unique_counts, me.array([3.0, 1.0, 2.0, 1.0]))
    test("unique_inverse", me.unique_inverse, me.array([3.0, 1.0, 2.0, 1.0]))
    test("unique_values", me.unique_values, me.array([3.0, 1.0, 2.0, 1.0]))

    print("\n--- Free Ops: Unstack ---")

    test("unstack", me.unstack, MATRIX)

    print("\n--- Free Ops: Memory Overlap ---")

    test("shares_memory", me.shares_memory, PAIR_A, PAIR_B)
    test("may_share_memory", me.may_share_memory, PAIR_A, PAIR_B)

    print("\n--- Free Ops: Misc Type ---")

    test("min_scalar_type", me.min_scalar_type, 10)
    test("issubdtype", me.issubdtype, "float64", "float64")
    test("common_type", me.common_type, PAIR_A)

    print("\n--- Free Ops: Put Along Axis ---")

    test("put_along_axis", me.put_along_axis,
         me.array([[1.0, 2.0], [3.0, 4.0]]),
         me.array([[0], [1]], dtype="int64"),
         me.array([[99.0], [88.0]]), axis=1)

    print("\n--- Free Ops: Histogram 2D / DD ---")

    test("histogram2d", me.histogram2d,
         me.array([1.0, 2.0, 3.0]), me.array([4.0, 5.0, 6.0]), 3)
    test("histogramdd", me.histogramdd,
         me.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 3)

    # ── 6. RANDOM submodule ───────────────────────────────────────────
    print("\n--- Random Submodule ---")

    test("random.rand", me.random.rand, 3, 3)
    test("random.randn", me.random.randn, 3, 3)
    test("random.random", me.random.random, (3,))
    test("random.randint", me.random.randint, 0, 10, (3,))
    test("random.uniform", me.random.uniform, 0.0, 1.0, (3,))
    test("random.normal", me.random.normal, 0.0, 1.0, (3,))
    test("random.seed", me.random.seed, 42)
    test("random.choice", me.random.choice, 10, 3)
    test("random.permutation", me.random.permutation, 5)
    test("random.beta", me.random.beta, 2.0, 5.0, (3,))
    test("random.binomial", me.random.binomial, 10, 0.5, (3,))
    test("random.chisquare", me.random.chisquare, 2, (3,))
    test("random.exponential", me.random.exponential, 1.0, (3,))
    test("random.gamma", me.random.gamma, 2.0, 1.0, (3,))
    test("random.geometric", me.random.geometric, 0.5, (3,))
    test("random.gumbel", me.random.gumbel, 0.0, 1.0, (3,))
    test("random.laplace", me.random.laplace, 0.0, 1.0, (3,))
    test("random.logistic", me.random.logistic, 0.0, 1.0, (3,))
    test("random.lognormal", me.random.lognormal, 0.0, 1.0, (3,))
    test("random.logseries", me.random.logseries, 0.9, (3,))
    test("random.multinomial", me.random.multinomial, 10, [0.5, 0.3, 0.2])
    test("random.negative_binomial", me.random.negative_binomial, 5, 0.5, (3,))
    test("random.noncentral_chisquare", me.random.noncentral_chisquare, 2, 1.0, (3,))
    test("random.noncentral_f", me.random.noncentral_f, 5, 10, 1.0, (3,))
    test("random.pareto", me.random.pareto, 2.0, (3,))
    test("random.poisson", me.random.poisson, 5.0, (3,))
    test("random.power", me.random.power, 2.0, (3,))
    test("random.rayleigh", me.random.rayleigh, 1.0, (3,))
    test("random.standard_cauchy", me.random.standard_cauchy, (3,))
    test("random.standard_exponential", me.random.standard_exponential, (3,))
    test("random.standard_gamma", me.random.standard_gamma, 2.0, (3,))
    test("random.standard_normal", me.random.standard_normal, (3,))
    test("random.standard_t", me.random.standard_t, 5.0, (3,))
    test("random.triangular", me.random.triangular, 0.0, 0.5, 1.0, (3,))
    test("random.vonmises", me.random.vonmises, 0.0, 1.0, (3,))
    test("random.wald", me.random.wald, 1.0, 1.0, (3,))
    test("random.weibull", me.random.weibull, 2.0, (3,))
    test("random.zipf", me.random.zipf, 2.0, (3,))
    test("random.dirichlet", me.random.dirichlet, [1.0, 1.0, 1.0])
    test("random.multivariate_normal", me.random.multivariate_normal,
         [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    test("random.f", me.random.f, 5, 10, (3,))
    test("random.hypergeometric", me.random.hypergeometric, 10, 5, 7, (3,))
    test("random.random_sample", me.random.random_sample, (3,))
    test("random.ranf", me.random.ranf, (3,))
    test("random.sample", me.random.sample, (3,))

    # random state ops
    test("random.get_state", me.random.get_state)

    # shuffle mutates in place
    arr_shuf = me.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test("random.shuffle", me.random.shuffle, arr_shuf)

    # Skip: random.set_state (needs valid state), random.bytes (returns bytes),
    #        random.random_integers (deprecated), random.default_rng
    skip("random.set_state", "needs valid state object")
    skip("random.bytes", "returns raw bytes, not array")
    skip("random.random_integers", "deprecated in numpy")
    skip("random.default_rng", "returns generator object, not array")

    # ── 7. Skipped functions ──────────────────────────────────────────
    print("\n--- Skipped Functions ---")

    # Functions that read/write files or need special environments
    for name in ["fromfile", "fromstring", "frombuffer", "fromfunction",
                 "fromiter", "fromregex", "from_dlpack",
                 "bmat", "isdtype"]:
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
