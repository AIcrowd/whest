"""
Exhaustive smoke test: calls EVERY non-blacklisted function in the flopscope
registry through the client-server proxy to verify it works end-to-end.

Runs INSIDE the participant container (no numpy).
"""

import sys

import flopscope as flops
import flopscope.numpy as fnp

fnp = fnp  # backwards-compat local alias for this test
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
print("  Exhaustive Flopscope Smoke Test")
print("=" * 70)

with flops.BudgetContext(flop_budget=10**9) as budget:
    # ── Test data ──────────────────────────────────────────────────────
    SMALL_POS = fnp.array([0.5, 1.0, 1.5, 2.0])
    SMALL_UNIT = fnp.array([0.1, 0.3, 0.5, 0.7])
    SMALL_NEG = fnp.array([-2.0, -1.0, 1.0, 2.0])
    SMALL_INT = fnp.array([1.0, 2.0, 3.0, 4.0])
    PAIR_A = fnp.array([1.0, 2.0, 3.0])
    PAIR_B = fnp.array([4.0, 5.0, 6.0])
    MATRIX = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    MATRIX3x2 = fnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    VEC2 = fnp.array([1.0, 2.0])
    VEC3 = fnp.array([1.0, 2.0, 3.0])
    BOOL_ARR = fnp.greater(PAIR_A, fnp.array([2.0, 2.0, 2.0]))
    SMALL_GE1 = fnp.array([1.0, 1.5, 2.0, 3.0])  # for arccosh
    INT_ARR = fnp.array([1, 2, 3, 4], dtype="int64")
    INT_PAIR_A = fnp.array([6, 12, 15], dtype="int64")
    INT_PAIR_B = fnp.array([4, 8, 10], dtype="int64")
    COMPLEX_ARR = fnp.array([1.0 + 2.0j, 3.0 - 1.0j], dtype="complex128")
    ZERO_TRIMMED = fnp.array([0.0, 1.0, 2.0, 0.0])

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
        test(name, getattr(fnp, name), SMALL_POS)

    # arcsin, arccos, asin, acos need [-1,1]
    for name in ["arcsin", "arccos", "asin", "acos"]:
        test(name, getattr(fnp, name), SMALL_UNIT)

    # arctanh, atanh need (-1,1)
    for name in ["arctanh", "atanh", "atan"]:
        test(name, getattr(fnp, name), SMALL_UNIT)

    # acosh, arccosh need >= 1
    for name in ["acosh", "arccosh"]:
        test(name, getattr(fnp, name), SMALL_GE1)

    # Multi-output unary
    test("modf", fnp.modf, SMALL_POS)
    test("frexp", fnp.frexp, SMALL_POS)

    # sort_complex
    test("sort_complex", fnp.sort_complex, COMPLEX_ARR)

    # isclose (binary signature but counted_unary category)
    test("isclose", fnp.isclose, PAIR_A, PAIR_B)

    # angle (works on complex)
    test("angle", fnp.angle, COMPLEX_ARR)

    # iscomplexobj, isrealobj work on any array
    test("iscomplexobj", fnp.iscomplexobj, SMALL_POS)
    test("isrealobj", fnp.isrealobj, SMALL_POS)

    # bitwise unary on int arrays
    test("bitwise_invert", fnp.bitwise_invert, INT_ARR)
    test("bitwise_not", fnp.bitwise_not, INT_ARR)
    test("bitwise_count", fnp.bitwise_count, INT_ARR)
    test("invert", fnp.invert, INT_ARR)

    # isneginf, isposinf
    test("isneginf", fnp.isneginf, SMALL_NEG)
    test("isposinf", fnp.isposinf, SMALL_POS)

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
        test(name, getattr(fnp, name), PAIR_A, PAIR_B)

    # ldexp: x * 2^i
    test("ldexp", fnp.ldexp, PAIR_A, fnp.array([1, 2, 3], dtype="int64"))

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
        test(name, getattr(fnp, name), INT_PAIR_A, fnp.array([1, 2, 1], dtype="int64"))

    # gcd, lcm
    test("gcd", fnp.gcd, INT_PAIR_A, INT_PAIR_B)
    test("lcm", fnp.lcm, INT_PAIR_A, INT_PAIR_B)

    # divmod (multi-output binary)
    test("divmod", fnp.divmod, PAIR_A, PAIR_B)

    # vecdot
    test("vecdot", fnp.vecdot, PAIR_A, PAIR_B)

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
        test(name, getattr(fnp, name), SMALL_INT)

    # average
    test("average", fnp.average, SMALL_INT)

    # percentile / quantile
    test("percentile", fnp.percentile, SMALL_INT, 50)
    test("quantile", fnp.quantile, SMALL_INT, 0.5)
    test("nanpercentile", fnp.nanpercentile, SMALL_INT, 50)
    test("nanquantile", fnp.nanquantile, SMALL_INT, 0.5)

    # cumulative_sum, cumulative_prod
    test("cumulative_sum", fnp.cumulative_sum, SMALL_INT, axis=0)
    test("cumulative_prod", fnp.cumulative_prod, SMALL_INT, axis=0)

    # ── 4. COUNTED CUSTOM ops ─────────────────────────────────────────
    print("\n--- Counted Custom Ops ---")

    # clip
    test("clip", fnp.clip, SMALL_INT, 1.5, 3.5)

    # dot, matmul
    test("dot", fnp.dot, PAIR_A, PAIR_B)
    test("matmul", fnp.matmul, MATRIX, VEC2)

    # inner, outer, vdot
    test("inner", fnp.inner, PAIR_A, PAIR_B)
    test("outer", fnp.outer, PAIR_A, PAIR_B)
    test("vdot", fnp.vdot, PAIR_A, PAIR_B)

    # tensordot
    test("tensordot", fnp.tensordot, MATRIX, MATRIX, 1)

    # kron
    test("kron", fnp.kron, VEC2, VEC3)

    # cross (3-d vectors)
    test("cross", fnp.cross, VEC3, fnp.array([4.0, 5.0, 6.0]))

    # diff, ediff1d
    test("diff", fnp.diff, SMALL_INT)
    test("ediff1d", fnp.ediff1d, SMALL_INT)

    # gradient
    test("gradient", fnp.gradient, SMALL_INT)

    # convolve, correlate
    test("convolve", fnp.convolve, PAIR_A, VEC2)
    test("correlate", fnp.correlate, PAIR_A, VEC2)

    # corrcoef, cov
    test("corrcoef", fnp.corrcoef, SMALL_INT)
    test("cov", fnp.cov, SMALL_INT)

    # einsum
    test("einsum", fnp.einsum, "ij,jk->ik", MATRIX, MATRIX)

    # einsum_path
    test("einsum_path", fnp.einsum_path, "ij,jk->ik", MATRIX, MATRIX)

    # trapezoid, trapz
    test("trapezoid", fnp.trapezoid, SMALL_INT)
    test("trapz", fnp.trapz, SMALL_INT)

    # interp
    test(
        "interp",
        fnp.interp,
        fnp.array([1.5, 2.5]),
        fnp.array([1.0, 2.0, 3.0]),
        fnp.array([10.0, 20.0, 30.0]),
    )

    # linalg.svd
    test("linalg.svd", fnp.linalg.svd, MATRIX3x2)

    # ── 5. FREE ops ───────────────────────────────────────────────────
    print("\n--- Free Ops: Tensor Creation ---")

    test("array", fnp.array, [1.0, 2.0, 3.0])
    test("zeros", fnp.zeros, (3, 3))
    test("ones", fnp.ones, (3, 3))
    test("full", fnp.full, (3, 3), 7.0)
    test("empty", fnp.empty, (3, 3))
    test("eye", fnp.eye, 3)
    test("identity", fnp.identity, 3)
    test("arange", fnp.arange, 0, 10, 2)
    test("linspace", fnp.linspace, 0, 1, 5)
    test("logspace", fnp.logspace, 0, 2, 5)
    test("geomspace", fnp.geomspace, 1, 100, 5)
    test("zeros_like", fnp.zeros_like, SMALL_INT)
    test("ones_like", fnp.ones_like, SMALL_INT)
    test("full_like", fnp.full_like, SMALL_INT, 9.0)
    test("empty_like", fnp.empty_like, SMALL_INT)
    test("diag", fnp.diag, VEC3)
    test("diagflat", fnp.diagflat, VEC2)
    test("tri", fnp.tri, 3, 3)
    test("tril", fnp.tril, MATRIX)
    test("triu", fnp.triu, MATRIX)
    test("vander", fnp.vander, PAIR_A, 3)

    print("\n--- Free Ops: Tensor Manipulation ---")

    test("reshape", fnp.reshape, SMALL_INT, (2, 2))
    test("transpose", fnp.transpose, MATRIX)
    test("swapaxes", fnp.swapaxes, MATRIX, 0, 1)
    test("moveaxis", fnp.moveaxis, MATRIX, 0, 1)
    test("concatenate", fnp.concatenate, [PAIR_A, PAIR_B])
    test("stack", fnp.stack, [PAIR_A, PAIR_B])
    test("vstack", fnp.vstack, [PAIR_A, PAIR_B])
    test("hstack", fnp.hstack, [PAIR_A, PAIR_B])
    test("column_stack", fnp.column_stack, [PAIR_A, PAIR_B])
    test("row_stack", fnp.row_stack, [PAIR_A, PAIR_B])
    test("dstack", fnp.dstack, [PAIR_A, PAIR_B])
    test("split", fnp.split, fnp.array([1.0, 2.0, 3.0, 4.0]), 2)
    test("hsplit", fnp.hsplit, fnp.array([1.0, 2.0, 3.0, 4.0]), 2)
    test("vsplit", fnp.vsplit, fnp.array([[1.0, 2.0], [3.0, 4.0]]), 2)
    test("dsplit", fnp.dsplit, fnp.array([[[1.0, 2.0], [3.0, 4.0]]]), 2)
    test("array_split", fnp.array_split, SMALL_INT, 2)
    test("squeeze", fnp.squeeze, fnp.array([[[1.0, 2.0]]]))
    test("expand_dims", fnp.expand_dims, PAIR_A, 0)
    test("ravel", fnp.ravel, MATRIX)
    test("copy", fnp.copy, PAIR_A)
    test("flip", fnp.flip, PAIR_A)
    test("fliplr", fnp.fliplr, MATRIX)
    test("flipud", fnp.flipud, MATRIX)
    test("rot90", fnp.rot90, MATRIX)
    test("roll", fnp.roll, PAIR_A, 1)
    test("rollaxis", fnp.rollaxis, MATRIX, 1)
    test("tile", fnp.tile, PAIR_A, 2)
    test("repeat", fnp.repeat, PAIR_A, 2)
    test("resize", fnp.resize, PAIR_A, (2, 3))
    test("append", fnp.append, PAIR_A, PAIR_B)
    test("insert", fnp.insert, PAIR_A, 1, 99.0)
    test("delete", fnp.delete, PAIR_A, 1)
    test("unique", fnp.unique, fnp.array([3.0, 1.0, 2.0, 1.0]))
    test("trim_zeros", fnp.trim_zeros, ZERO_TRIMMED)
    test("sort", fnp.sort, fnp.array([3.0, 1.0, 2.0]))
    test("argsort", fnp.argsort, fnp.array([3.0, 1.0, 2.0]))
    test("partition", fnp.partition, fnp.array([3.0, 1.0, 2.0, 4.0]), 2)
    test("argpartition", fnp.argpartition, fnp.array([3.0, 1.0, 2.0, 4.0]), 2)
    test("take", fnp.take, PAIR_A, [0, 2])
    test(
        "take_along_axis",
        fnp.take_along_axis,
        fnp.array([[1.0, 2.0], [3.0, 4.0]]),
        fnp.array([[0, 1], [1, 0]], dtype="int64"),
        axis=1,
    )
    test("compress", fnp.compress, [True, False, True], PAIR_A)
    test("extract", fnp.extract, BOOL_ARR, PAIR_A)
    test("diagonal", fnp.diagonal, MATRIX)
    test("trace", fnp.trace, MATRIX)
    test("pad", fnp.pad, PAIR_A, 1)
    test("searchsorted", fnp.searchsorted, fnp.array([1.0, 3.0, 5.0]), 2.0)

    print("\n--- Free Ops: Where / Select / Place ---")

    test("where", fnp.where, BOOL_ARR, PAIR_A, PAIR_B)
    test("select", fnp.select, [BOOL_ARR], [PAIR_A])
    test("nonzero", fnp.nonzero, fnp.array([0.0, 1.0, 0.0, 2.0]))
    test("argwhere", fnp.argwhere, fnp.array([0.0, 1.0, 0.0, 2.0]))
    test("flatnonzero", fnp.flatnonzero, fnp.array([0.0, 1.0, 0.0, 2.0]))
    test("isin", fnp.isin, PAIR_A, [1.0, 3.0])
    test("in1d", fnp.in1d, PAIR_A, fnp.array([1.0, 3.0]))

    print("\n--- Free Ops: Set ops ---")

    test("intersect1d", fnp.intersect1d, PAIR_A, fnp.array([2.0, 3.0, 7.0]))
    test("union1d", fnp.union1d, PAIR_A, fnp.array([2.0, 5.0]))
    test("setdiff1d", fnp.setdiff1d, PAIR_A, fnp.array([2.0]))
    test("setxor1d", fnp.setxor1d, PAIR_A, fnp.array([2.0, 5.0]))

    print("\n--- Free Ops: Comparison / Type ---")

    test("allclose", fnp.allclose, PAIR_A, PAIR_A)
    test("array_equal", fnp.array_equal, PAIR_A, PAIR_A)
    test("array_equiv", fnp.array_equiv, PAIR_A, PAIR_A)
    test("isfinite", fnp.isfinite, SMALL_POS)
    test("isinf", fnp.isinf, SMALL_POS)
    test("isnan", fnp.isnan, SMALL_POS)
    test("isscalar", fnp.isscalar, 5.0)

    print("\n--- Free Ops: Shape / Type Introspection ---")

    test("shape", fnp.shape, MATRIX)
    test("ndim", fnp.ndim, MATRIX)
    test("size", fnp.size, MATRIX)

    print("\n--- Free Ops: Broadcast ---")

    test("broadcast_to", fnp.broadcast_to, PAIR_A, (2, 3))
    test("broadcast_arrays", fnp.broadcast_arrays, PAIR_A, PAIR_B)
    test("broadcast_shapes", fnp.broadcast_shapes, (3,), (3,))

    print("\n--- Free Ops: Histogram ---")

    test("histogram", fnp.histogram, SMALL_INT, 3)
    test("histogram_bin_edges", fnp.histogram_bin_edges, SMALL_INT, 3)

    print("\n--- Free Ops: Type Conversion / Casting ---")

    test("asarray", fnp.asarray, PAIR_A)
    test("asarray_chkfinite", fnp.asarray_chkfinite, PAIR_A)
    test("astype", fnp.astype, PAIR_A, "float32")
    test("can_cast", fnp.can_cast, "float32", "float64")
    test("result_type", fnp.result_type, "float32", "float64")
    test("promote_types", fnp.promote_types, "float32", "float64")

    print("\n--- Free Ops: Atleast ---")

    test("atleast_1d", fnp.atleast_1d, PAIR_A)
    test("atleast_2d", fnp.atleast_2d, PAIR_A)
    test("atleast_3d", fnp.atleast_3d, PAIR_A)

    print("\n--- Free Ops: Index Helpers ---")

    test("diag_indices", fnp.diag_indices, 3)
    test("diag_indices_from", fnp.diag_indices_from, MATRIX)
    test("tril_indices", fnp.tril_indices, 3)
    test("triu_indices", fnp.triu_indices, 3)
    test("tril_indices_from", fnp.tril_indices_from, MATRIX)
    test("triu_indices_from", fnp.triu_indices_from, MATRIX)
    test("indices", fnp.indices, (2, 3))
    test(
        "ix_", fnp.ix_, fnp.array([0, 1], dtype="int64"), fnp.array([0, 1], dtype="int64")
    )
    test("unravel_index", fnp.unravel_index, 5, (3, 3))
    test(
        "ravel_multi_index",
        fnp.ravel_multi_index,
        (fnp.array([0, 1, 2], dtype="int64"), fnp.array([0, 1, 2], dtype="int64")),
        (3, 3),
    )
    test("mask_indices", fnp.mask_indices, 3, fnp.triu)

    print("\n--- Free Ops: Mesh / Grid ---")

    test("meshgrid", fnp.meshgrid, fnp.array([1.0, 2.0]), fnp.array([3.0, 4.0]))

    print("\n--- Free Ops: Misc ---")

    test("lexsort", fnp.lexsort, (fnp.array([1.0, 2.0, 1.0]), fnp.array([3.0, 1.0, 2.0])))
    test("digitize", fnp.digitize, fnp.array([0.5, 1.5, 2.5]), fnp.array([1.0, 2.0, 3.0]))
    test("bincount", fnp.bincount, fnp.array([0, 1, 1, 2, 3, 3, 3], dtype="int64"))
    test("packbits", fnp.packbits, fnp.array([1, 0, 1, 1, 0, 0, 0, 1], dtype="uint8"))
    test("unpackbits", fnp.unpackbits, fnp.array([177], dtype="uint8"))

    test("block", fnp.block, [[PAIR_A, PAIR_B]])
    test("concat", fnp.concat, [PAIR_A, PAIR_B])

    print("\n--- Free Ops: Misc Introspection ---")

    test("iterable", fnp.iterable, PAIR_A)
    test("isfortran", fnp.isfortran, MATRIX)
    test("typename", fnp.typename, "float64")
    test("mintypecode", fnp.mintypecode, ["f", "d"])
    test("base_repr", fnp.base_repr, 10, 2)
    test("binary_repr", fnp.binary_repr, 10)

    print("\n--- Free Ops: Put / Place / Copyto ---")

    # put, place, putmask, fill_diagonal, copyto mutate arrays
    arr_put = fnp.array([1.0, 2.0, 3.0, 4.0])
    test("put", fnp.put, arr_put, [0, 2], [99.0, 88.0])

    arr_place = fnp.array([1.0, 2.0, 3.0, 4.0])
    test(
        "place",
        fnp.place,
        arr_place,
        fnp.array([True, False, True, False], dtype="bool"),
        [99.0, 88.0],
    )

    arr_pm = fnp.array([1.0, 2.0, 3.0, 4.0])
    test(
        "putmask",
        fnp.putmask,
        arr_pm,
        fnp.array([True, False, True, False], dtype="bool"),
        0.0,
    )

    mat_fd = fnp.array([[1.0, 2.0], [3.0, 4.0]])
    test("fill_diagonal", fnp.fill_diagonal, mat_fd, 0.0)

    arr_ct = fnp.array([1.0, 2.0, 3.0])
    test("copyto", fnp.copyto, arr_ct, fnp.array([9.0, 8.0, 7.0]))

    print("\n--- Free Ops: Choose / Require ---")

    test(
        "choose",
        fnp.choose,
        fnp.array([0, 1, 0], dtype="int64"),
        [fnp.array([10.0, 20.0, 30.0]), fnp.array([40.0, 50.0, 60.0])],
    )
    test("require", fnp.require, PAIR_A)

    print("\n--- Free Ops: Matrix Transpose / Permute ---")

    test("matrix_transpose", fnp.matrix_transpose, fnp.array([[[1.0, 2.0], [3.0, 4.0]]]))
    test("permute_dims", fnp.permute_dims, MATRIX, (1, 0))

    print("\n--- Free Ops: Unique Variants ---")

    test("unique_all", fnp.unique_all, fnp.array([3.0, 1.0, 2.0, 1.0]))
    test("unique_counts", fnp.unique_counts, fnp.array([3.0, 1.0, 2.0, 1.0]))
    test("unique_inverse", fnp.unique_inverse, fnp.array([3.0, 1.0, 2.0, 1.0]))
    test("unique_values", fnp.unique_values, fnp.array([3.0, 1.0, 2.0, 1.0]))

    print("\n--- Free Ops: Unstack ---")

    test("unstack", fnp.unstack, MATRIX)

    print("\n--- Free Ops: Memory Overlap ---")

    test("shares_memory", fnp.shares_memory, PAIR_A, PAIR_B)
    test("may_share_memory", fnp.may_share_memory, PAIR_A, PAIR_B)

    print("\n--- Free Ops: Misc Type ---")

    test("min_scalar_type", fnp.min_scalar_type, 10)
    test("issubdtype", fnp.issubdtype, "float64", "float64")
    test("common_type", fnp.common_type, PAIR_A)

    print("\n--- Free Ops: Put Along Axis ---")

    test(
        "put_along_axis",
        fnp.put_along_axis,
        fnp.array([[1.0, 2.0], [3.0, 4.0]]),
        fnp.array([[0], [1]], dtype="int64"),
        fnp.array([[99.0], [88.0]]),
        axis=1,
    )

    print("\n--- Free Ops: Histogram 2D / DD ---")

    test(
        "histogram2d",
        fnp.histogram2d,
        fnp.array([1.0, 2.0, 3.0]),
        fnp.array([4.0, 5.0, 6.0]),
        3,
    )
    test(
        "histogramdd", fnp.histogramdd, fnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 3
    )

    # ── 6. RANDOM submodule ───────────────────────────────────────────
    print("\n--- Random Submodule ---")

    test("random.rand", fnp.random.rand, 3, 3)
    test("random.randn", fnp.random.randn, 3, 3)
    test("random.random", fnp.random.random, (3,))
    test("random.randint", fnp.random.randint, 0, 10, (3,))
    test("random.uniform", fnp.random.uniform, 0.0, 1.0, (3,))
    test("random.normal", fnp.random.normal, 0.0, 1.0, (3,))
    test("random.seed", fnp.random.seed, 42)
    test("random.choice", fnp.random.choice, 10, 3)
    test("random.permutation", fnp.random.permutation, 5)
    test("random.beta", fnp.random.beta, 2.0, 5.0, (3,))
    test("random.binomial", fnp.random.binomial, 10, 0.5, (3,))
    test("random.chisquare", fnp.random.chisquare, 2, (3,))
    test("random.exponential", fnp.random.exponential, 1.0, (3,))
    test("random.gamma", fnp.random.gamma, 2.0, 1.0, (3,))
    test("random.geometric", fnp.random.geometric, 0.5, (3,))
    test("random.gumbel", fnp.random.gumbel, 0.0, 1.0, (3,))
    test("random.laplace", fnp.random.laplace, 0.0, 1.0, (3,))
    test("random.logistic", fnp.random.logistic, 0.0, 1.0, (3,))
    test("random.lognormal", fnp.random.lognormal, 0.0, 1.0, (3,))
    test("random.logseries", fnp.random.logseries, 0.9, (3,))
    test("random.multinomial", fnp.random.multinomial, 10, [0.5, 0.3, 0.2])
    test("random.negative_binomial", fnp.random.negative_binomial, 5, 0.5, (3,))
    test("random.noncentral_chisquare", fnp.random.noncentral_chisquare, 2, 1.0, (3,))
    test("random.noncentral_f", fnp.random.noncentral_f, 5, 10, 1.0, (3,))
    test("random.pareto", fnp.random.pareto, 2.0, (3,))
    test("random.poisson", fnp.random.poisson, 5.0, (3,))
    test("random.power", fnp.random.power, 2.0, (3,))
    test("random.rayleigh", fnp.random.rayleigh, 1.0, (3,))
    test("random.standard_cauchy", fnp.random.standard_cauchy, (3,))
    test("random.standard_exponential", fnp.random.standard_exponential, (3,))
    test("random.standard_gamma", fnp.random.standard_gamma, 2.0, (3,))
    test("random.standard_normal", fnp.random.standard_normal, (3,))
    test("random.standard_t", fnp.random.standard_t, 5.0, (3,))
    test("random.triangular", fnp.random.triangular, 0.0, 0.5, 1.0, (3,))
    test("random.vonmises", fnp.random.vonmises, 0.0, 1.0, (3,))
    test("random.wald", fnp.random.wald, 1.0, 1.0, (3,))
    test("random.weibull", fnp.random.weibull, 2.0, (3,))
    test("random.zipf", fnp.random.zipf, 2.0, (3,))
    test("random.dirichlet", fnp.random.dirichlet, [1.0, 1.0, 1.0])
    test(
        "random.multivariate_normal",
        fnp.random.multivariate_normal,
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    test("random.f", fnp.random.f, 5, 10, (3,))
    test("random.hypergeometric", fnp.random.hypergeometric, 10, 5, 7, (3,))
    test("random.random_sample", fnp.random.random_sample, (3,))
    test("random.ranf", fnp.random.ranf, (3,))
    test("random.sample", fnp.random.sample, (3,))

    # random state ops
    test("random.get_state", fnp.random.get_state)

    # shuffle mutates in place
    arr_shuf = fnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test("random.shuffle", fnp.random.shuffle, arr_shuf)

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
