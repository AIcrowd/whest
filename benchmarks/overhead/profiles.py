"""Input profile materialization for overhead benchmark cases."""

from __future__ import annotations

from typing import Any

import numpy as np

_VECTOR_SIZES = {"tiny": 8, "medium": 256}
_MATRIX_SIZES = {"tiny": 8, "medium": 48}
_FFT_SIZES = {"tiny": 16, "medium": 128}
_RANDOM_SIZES = {"tiny": 8, "medium": 128}

_UNIT_INTERVAL_UNARY = {"arccos", "arcsin"}
_OPEN_UNIT_INTERVAL_UNARY = {"arctanh"}
_AT_LEAST_ONE_UNARY = {"arccosh"}
_POSITIVE_UNARY = {"log", "log10", "log2", "sqrt", "reciprocal"}
_GT_NEGATIVE_ONE_UNARY = {"log1p"}
_NONZERO_DENOMINATOR_BINARY = {
    "divide",
    "true_divide",
    "floor_divide",
    "fmod",
    "mod",
    "remainder",
}
_POSITIVE_BASE_BINARY = {"power", "float_power"}
_BITWISE_UNARY = {"bitwise_not", "bitwise_invert", "invert", "bitwise_count"}
_BITWISE_BINARY = {"bitwise_and", "bitwise_or", "bitwise_xor", "gcd", "lcm"}
_BITWISE_SHIFT = {
    "bitwise_left_shift",
    "bitwise_right_shift",
    "left_shift",
    "right_shift",
}

_RANDOM_EXTRA_ARGS: dict[str, tuple[object, ...]] = {
    "uniform": (0.0, 1.0),
    "standard_gamma": (1.0,),
    "standard_t": (3,),
    "poisson": (5.0,),
    "binomial": (10, 0.5),
    "beta": (2.0, 5.0),
    "chisquare": (2,),
    "exponential": (1.0,),
    "f": (5, 10),
    "gamma": (2.0, 1.0),
    "geometric": (0.5,),
    "gumbel": (0.0, 1.0),
    "hypergeometric": (100, 50, 20),
    "laplace": (0.0, 1.0),
    "logistic": (0.0, 1.0),
    "lognormal": (0.0, 1.0),
    "logseries": (0.5,),
    "negative_binomial": (10, 0.5),
    "noncentral_chisquare": (2, 1.0),
    "noncentral_f": (5, 10, 1.0),
    "normal": (0.0, 1.0),
    "pareto": (3.0,),
    "power": (5.0,),
    "rayleigh": (1.0,),
    "triangular": (-1.0, 0.0, 1.0),
    "vonmises": (0.0, 1.0),
    "wald": (1.0, 1.0),
    "weibull": (2.0,),
    "zipf": (2.0,),
    "randint": (0, 1000),
}

_REDUCTION_SPECIAL_ARGS: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {
    "percentile": ((), {"q": 50}),
    "nanpercentile": ((), {"q": 50}),
    "quantile": ((), {"q": 0.5}),
    "nanquantile": ((), {"q": 0.5}),
}


def _vector(size_name: str, dtype: str, *, offset: int = 0) -> np.ndarray:
    size = _VECTOR_SIZES[size_name]
    values = np.arange(size, dtype=np.float64)
    values = ((values + offset) % 29 - 14) / 8.0
    return values.astype(np.dtype(dtype), copy=False)


def _positive_vector(size_name: str, dtype: str, *, offset: int = 0) -> np.ndarray:
    return np.abs(_vector(size_name, dtype, offset=offset)) + np.asarray(
        0.25, dtype=np.dtype(dtype)
    )


def _int_vector(size_name: str, dtype: str = "int64", *, offset: int = 0) -> np.ndarray:
    size = _VECTOR_SIZES[size_name]
    values = (np.arange(size, dtype=np.int64) + offset) % 31
    return (values + 1).astype(np.dtype(dtype), copy=False)


def _complex_vector(
    size_name: str, dtype: str = "complex128", *, offset: int = 0
) -> np.ndarray:
    real = _vector(size_name, "float64", offset=offset)
    imag = _vector(size_name, "float64", offset=offset + 11)
    return (real + (1j * imag)).astype(np.dtype(dtype), copy=False)


def _linspace_vector(
    size_name: str, dtype: str, start: float, stop: float
) -> np.ndarray:
    return np.linspace(
        start,
        stop,
        _VECTOR_SIZES[size_name],
        dtype=np.dtype(dtype),
    )


def _matrix(size_name: str, dtype: str, *, offset: int = 0) -> np.ndarray:
    side = _MATRIX_SIZES[size_name]
    size = side * side
    values = np.arange(size, dtype=np.float64)
    values = ((values + offset) % 37 - 18) / 10.0
    return values.reshape(side, side).astype(np.dtype(dtype), copy=False)


def _spd_matrix(size_name: str, dtype: str) -> np.ndarray:
    matrix = _matrix(size_name, dtype, offset=11)
    side = matrix.shape[0]
    with np.errstate(all="ignore"):
        return matrix @ matrix.T + side * np.eye(side, dtype=np.dtype(dtype))


def _fft_vector(size_name: str, dtype: str) -> np.ndarray:
    size = _FFT_SIZES[size_name]
    grid = np.linspace(0.0, 2.0 * np.pi, size, dtype=np.dtype(dtype))
    return np.sin(3 * grid) + 0.5 * np.cos(5 * grid)


def _fft_matrix(size_name: str, dtype: str) -> np.ndarray:
    side = int(np.sqrt(_FFT_SIZES[size_name]))
    grid = np.linspace(0.0, 2.0 * np.pi, side * side, dtype=np.dtype(dtype))
    values = np.sin(3 * grid) + 0.25 * np.cos(7 * grid)
    return values.reshape(side, side)


def _array_data(size_name: str) -> list[list[float]]:
    side = 2 if size_name == "tiny" else 12
    return [
        [float((row * side + col) % 7) / 3.0 for col in range(side)]
        for row in range(side)
    ]


def _sort_vector(size_name: str, dtype: str, *, reverse: bool = False) -> np.ndarray:
    values = np.sort(_vector(size_name, dtype, offset=23))
    if reverse:
        return values[::-1].copy()
    return values


def _set_vectors(size_name: str, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    return _vector(size_name, dtype, offset=0), _vector(size_name, dtype, offset=9)


def _choice_pool(size_name: str, dtype: str) -> np.ndarray:
    size = _RANDOM_SIZES[size_name]
    return np.arange(size * 2, dtype=np.dtype(dtype))


def _cube(size_name: str, dtype: str) -> np.ndarray:
    depth = 4 if size_name == "tiny" else 8
    values = np.arange(2 * 3 * depth, dtype=np.float64)
    values = ((values % 19) - 9) / 4.0
    return values.reshape(2, 3, depth).astype(np.dtype(dtype), copy=False)


def _small_kernel(dtype: str) -> np.ndarray:
    return np.array([0.25, -0.5, 0.25], dtype=np.dtype(dtype))


def _matrix_samples(size_name: str, dtype: str, *, rows: int = 4) -> np.ndarray:
    cols = 12 if size_name == "tiny" else 48
    values = np.linspace(-1.5, 1.5, rows * cols, dtype=np.float64).reshape(rows, cols)
    return values.astype(np.dtype(dtype), copy=False)


def _tensor_matrix(size_name: str, dtype: str) -> tuple[np.ndarray, int]:
    side = 4 if size_name == "tiny" else 9
    matrix = _spd_matrix(
        "tiny" if size_name == "tiny" else "medium", dtype=np.dtype(dtype).name
    )
    matrix = matrix[:side, :side]
    return matrix, int(round(np.sqrt(side)))


def _misc_args(
    op_name: str, size_name: str, dtype: str
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if op_name in {"diff", "ediff1d"}:
        return (_vector(size_name, dtype),), {}
    if op_name in {"gradient", "unwrap"}:
        return (_vector(size_name, dtype),), {}
    if op_name in {"convolve", "correlate"}:
        return (_vector(size_name, dtype), _small_kernel(dtype)), {}
    if op_name in {"corrcoef", "cov"}:
        return (_matrix_samples(size_name, dtype),), {}
    if op_name == "cross":
        a = np.stack([_vector(size_name, dtype, offset=i) for i in range(3)], axis=-1)
        b = np.stack(
            [_vector(size_name, dtype, offset=i + 7) for i in range(3)], axis=-1
        )
        return (a, b), {}
    if op_name == "histogram":
        return (_vector(size_name, dtype),), {"bins": 8}
    if op_name == "histogram2d":
        return (_vector(size_name, dtype), _vector(size_name, dtype, offset=5)), {
            "bins": 8
        }
    if op_name == "histogramdd":
        samples = np.stack(
            [_vector(size_name, dtype, offset=i * 5) for i in range(3)],
            axis=-1,
        )
        return (samples,), {"bins": 8}
    if op_name == "histogram_bin_edges":
        return (_vector(size_name, dtype),), {"bins": 8}
    if op_name == "digitize":
        bins = np.linspace(-2.0, 2.0, 9, dtype=np.dtype(dtype))
        return (_vector(size_name, dtype), bins), {}
    if op_name == "bincount":
        return (_int_vector(size_name),), {}
    if op_name == "interp":
        xp = np.linspace(0.0, 1.0, 8, dtype=np.dtype(dtype))
        fp = np.linspace(-1.0, 1.0, 8, dtype=np.dtype(dtype))
        return (
            np.linspace(0.0, 1.0, _VECTOR_SIZES[size_name], dtype=np.dtype(dtype)),
            xp,
            fp,
        ), {}
    if op_name == "trapezoid":
        return (_vector(size_name, dtype),), {}
    if op_name == "vander":
        return (_vector(size_name, dtype),), {"N": 6}
    raise ValueError(f"unsupported misc op: {op_name!r}")


def _polynomial_args(
    op_name: str, size_name: str, dtype: str
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    coeffs = np.linspace(-1.0, 1.0, 6, dtype=np.dtype(dtype))
    coeffs_b = np.linspace(0.5, 1.5, 6, dtype=np.dtype(dtype))
    x = np.linspace(-1.0, 1.0, _VECTOR_SIZES[size_name], dtype=np.dtype(dtype))
    roots = np.linspace(-1.0, 1.0, 5, dtype=np.dtype(dtype))
    if op_name == "polyval":
        return (coeffs, x), {}
    if op_name == "polyfit":
        y = np.polyval(coeffs, x)
        return (x, y, 3), {}
    if op_name == "poly":
        return (roots,), {}
    if op_name == "roots":
        return (coeffs,), {}
    if op_name in {"polyadd", "polysub", "polymul", "polydiv"}:
        return (coeffs, coeffs_b), {}
    if op_name in {"polyder", "polyint"}:
        return (coeffs,), {}
    raise ValueError(f"unsupported polynomial op: {op_name!r}")


def _linalg_delegate_args(
    op_name: str, size_name: str, dtype: str
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if op_name in {"cond", "matrix_rank", "matrix_norm"}:
        return (_spd_matrix(size_name, dtype),), {}
    if op_name == "matrix_power":
        return (_spd_matrix(size_name, dtype), 3), {}
    if op_name == "norm":
        return (_vector(size_name, dtype),), {}
    if op_name == "vector_norm":
        return (_vector(size_name, dtype),), {}
    if op_name == "outer":
        return (_vector(size_name, dtype), _vector(size_name, dtype, offset=7)), {}
    if op_name == "cross":
        a = np.stack([_vector(size_name, dtype, offset=i) for i in range(3)], axis=-1)
        b = np.stack(
            [_vector(size_name, dtype, offset=i + 7) for i in range(3)], axis=-1
        )
        return (a, b), {}
    if op_name == "multi_dot":
        a = _matrix(size_name, dtype)[:, :12]
        b = _matrix(size_name, dtype, offset=7)[:12, :12]
        c = _matrix(size_name, dtype, offset=14)[:12, :]
        return ([a, b, c],), {}
    if op_name == "tensordot":
        a = _matrix(size_name, dtype).reshape(4, 4, -1)
        b = _matrix(size_name, dtype, offset=7).reshape(-1, 4, 4)
        return (a, b), {"axes": 1}
    if op_name == "tensorinv":
        matrix, side = _tensor_matrix(size_name, dtype)
        return (matrix.reshape(side, side, side, side),), {}
    if op_name == "tensorsolve":
        matrix, side = _tensor_matrix(size_name, dtype)
        tensor = matrix.reshape(side, side, side, side)
        rhs = np.ones((side, side), dtype=np.dtype(dtype))
        return (tensor, rhs), {}
    if op_name == "trace":
        return (_matrix(size_name, dtype),), {}
    if op_name == "matmul":
        return (_matrix(size_name, dtype), _matrix(size_name, dtype, offset=17)), {}
    if op_name == "vecdot":
        return (_vector(size_name, dtype), _vector(size_name, dtype, offset=7)), {}
    raise ValueError(f"unsupported linalg delegate op: {op_name!r}")


def _bitwise_args(
    op_name: str, size_name: str
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if op_name in _BITWISE_UNARY:
        return (_int_vector(size_name),), {}
    if op_name in _BITWISE_BINARY:
        return (_int_vector(size_name), _int_vector(size_name, offset=5)), {}
    if op_name in _BITWISE_SHIFT:
        return (
            _int_vector(size_name),
            np.full(_VECTOR_SIZES[size_name], 2, dtype=np.int64),
        ), {}
    raise ValueError(f"unsupported bitwise op: {op_name!r}")


def _complex_args(
    op_name: str, size_name: str, dtype: str
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    values = _complex_vector(size_name, dtype)
    if op_name == "sort_complex":
        return (values,), {}
    return (values,), {}


def _free_args(
    op_name: str, size_name: str, dtype: str
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    matrix = _matrix(size_name, dtype)
    cube = _cube(size_name, dtype)
    if op_name in {"asarray"}:
        return (_array_data(size_name),), {}
    if op_name in {
        "copy",
        "require",
        "ravel",
        "flip",
        "fliplr",
        "flipud",
        "rot90",
        "squeeze",
    }:
        if op_name == "squeeze":
            return (matrix.reshape(1, *matrix.shape),), {}
        return (matrix,), {}
    if op_name == "astype":
        return (matrix, "float32"), {}
    if op_name in {"atleast_1d", "atleast_2d", "atleast_3d"}:
        return (_vector(size_name, dtype),), {}
    if op_name == "broadcast_arrays":
        return (matrix[:1, :], matrix[0, :]), {}
    if op_name == "broadcast_shapes":
        width = matrix.shape[1]
        return ((1, width), (matrix.shape[0], width)), {}
    if op_name == "broadcast_to":
        return (matrix[:1, :], matrix.shape), {}
    if op_name in {"column_stack", "hstack"}:
        return ((_vector(size_name, dtype), _vector(size_name, dtype, offset=7)),), {}
    if op_name in {"array_split", "split"}:
        return (_vector(size_name, dtype), 2), {}
    if op_name == "hsplit":
        return (matrix, 2), {}
    if op_name == "vsplit":
        return (matrix, 2), {}
    if op_name == "dsplit":
        return (cube, 2), {}
    if op_name in {"empty", "ones", "zeros"}:
        return (matrix.shape,), {}
    if op_name in {"empty_like", "ones_like", "zeros_like"}:
        return (matrix,), {}
    if op_name == "expand_dims":
        return (_vector(size_name, dtype),), {"axis": 0}
    if op_name in {"eye", "identity"}:
        return (matrix.shape[0],), {}
    if op_name in {"isdtype"}:
        return (np.dtype(dtype), "real floating"), {}
    if op_name in {"isfortran", "ndim", "shape", "size"}:
        return (matrix,), {}
    if op_name == "isscalar":
        return (3.14,), {}
    if op_name == "iterable":
        return ([1, 2, 3],), {}
    if op_name in {"issubdtype"}:
        return (np.float64, np.floating), {}
    if op_name in {"min_scalar_type"}:
        return (3.14,), {}
    if op_name in {"mintypecode"}:
        return (["d", "f"],), {}
    if op_name in {"can_cast"}:
        return (np.int32, np.float64), {}
    if op_name in {"common_type"}:
        return (_vector(size_name, "float32"), _vector(size_name, "float64")), {}
    if op_name in {"promote_types"}:
        return (np.int32, np.float64), {}
    if op_name in {"result_type"}:
        return (_vector(size_name, "float32"), _vector(size_name, "float64")), {}
    if op_name == "diag_indices":
        return (matrix.shape[0],), {}
    if op_name == "diag_indices_from":
        return (matrix,), {}
    if op_name == "diagonal":
        return (matrix,), {}
    if op_name == "matrix_transpose":
        return (matrix,), {}
    if op_name in {"may_share_memory", "shares_memory"}:
        return (matrix, matrix[:, :]), {}
    if op_name in {"moveaxis"}:
        return (cube, 0, 2), {}
    if op_name in {"permute_dims"}:
        return (cube, (1, 0, 2)), {}
    if op_name in {"rollaxis"}:
        return (cube, 2, 0), {}
    if op_name in {"swapaxes"}:
        return (cube, 0, 2), {}
    if op_name in {"transpose"}:
        return (cube,), {}
    if op_name == "reshape":
        return (matrix, (matrix.shape[0], -1)), {}
    if op_name == "ravel_multi_index":
        dims = (4, 5)
        return ((np.array([1, 2]), np.array([3, 4])), dims), {}
    if op_name == "tri":
        return (matrix.shape[0], matrix.shape[1]), {}
    if op_name in {"tril", "triu"}:
        return (matrix,), {}
    if op_name in {"tril_indices", "triu_indices"}:
        return (matrix.shape[0],), {}
    if op_name in {"tril_indices_from", "triu_indices_from"}:
        return (matrix,), {}
    if op_name == "unravel_index":
        return (np.array([1, 5]), (3, 4)), {}
    raise ValueError(f"unsupported free op: {op_name!r}")


def _unary_vector(op_name: str, size_name: str, dtype: str) -> np.ndarray:
    if op_name in _UNIT_INTERVAL_UNARY:
        return _linspace_vector(size_name, dtype, -0.9, 0.9)
    if op_name in _OPEN_UNIT_INTERVAL_UNARY:
        return _linspace_vector(size_name, dtype, -0.9, 0.9)
    if op_name in _AT_LEAST_ONE_UNARY:
        return _linspace_vector(size_name, dtype, 1.25, 3.0)
    if op_name in _POSITIVE_UNARY:
        return _linspace_vector(size_name, dtype, 0.25, 3.0)
    if op_name in _GT_NEGATIVE_ONE_UNARY:
        return _linspace_vector(size_name, dtype, -0.75, 2.0)
    return _vector(size_name, dtype)


def _binary_inputs(
    op_name: str, size_name: str, dtype: str
) -> tuple[np.ndarray, np.ndarray]:
    if op_name in _NONZERO_DENOMINATOR_BINARY:
        return _vector(size_name, dtype), _positive_vector(size_name, dtype, offset=13)
    if op_name in _POSITIVE_BASE_BINARY:
        return _positive_vector(size_name, dtype), _vector(size_name, dtype, offset=13)
    if op_name == "ldexp":
        exponents = np.resize(
            np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32),
            _VECTOR_SIZES[size_name],
        )
        return _vector(size_name, dtype), exponents
    return _vector(size_name, dtype), _vector(size_name, dtype, offset=13)


def materialize_case_inputs(
    case_payload: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Materialize positional/keyword call inputs for one benchmark case."""

    profile_kind = case_payload.get("profile_kind")
    if not profile_kind:
        operand_shapes = case_payload.get("operand_shapes", [])
        dtype = case_payload["dtype"]
        args = []
        for index, shape in enumerate(operand_shapes):
            size = int(np.prod(shape, dtype=int)) if shape else 1
            values = np.arange(size, dtype=np.float64)
            values = ((values + (index * 17)) % 31 - 15) / 16.0
            args.append(values.reshape(tuple(shape)).astype(dtype, copy=False))
        return tuple(args), {}

    size_name = case_payload["size_name"]
    dtype = case_payload["dtype"]
    params = case_payload.get("profile_params", {})
    op_name = str(params.get("op_name", case_payload.get("op_name", "")))

    if profile_kind == "vector_unary":
        return (_unary_vector(op_name, size_name, dtype),), {}
    if profile_kind == "vector_binary":
        return _binary_inputs(op_name, size_name, dtype), {}
    if profile_kind == "vector_reduction":
        args, kwargs = _REDUCTION_SPECIAL_ARGS.get(op_name, ((), {}))
        return (_vector(size_name, dtype), *args), dict(kwargs)
    if profile_kind == "matmul":
        return (_matrix(size_name, dtype), _matrix(size_name, dtype, offset=19)), {}
    if profile_kind == "allclose":
        a = _vector(size_name, dtype)
        b = a + np.asarray(1e-6, dtype=np.dtype(dtype))
        return (a, b), {}
    if profile_kind == "array_compare":
        return _set_vectors(size_name, dtype), {}
    if profile_kind == "clip":
        return (_vector(size_name, dtype), -1.0, 1.0), {}
    if profile_kind == "array_constructor":
        return (_array_data(size_name),), {}
    if profile_kind == "arange":
        stop = _VECTOR_SIZES[size_name]
        return (0, stop, 1), {}
    if profile_kind == "linspace":
        size = _VECTOR_SIZES[size_name]
        return (0.0, 1.0, size), {}
    if profile_kind == "logspace":
        size = _VECTOR_SIZES[size_name]
        return (0.0, 2.0, size), {}
    if profile_kind == "geomspace":
        size = _VECTOR_SIZES[size_name]
        return (1.0, float(size + 1), size), {}
    if profile_kind == "sort_like":
        return (_vector(size_name, dtype, offset=31),), {}
    if profile_kind == "partition_like":
        values = _vector(size_name, dtype, offset=31)
        return (values, len(values) // 2), {}
    if profile_kind == "searchsorted":
        haystack = _sort_vector(size_name, dtype)
        queries = _vector(size_name, dtype, offset=5)
        return (haystack, queries), {}
    if profile_kind == "set_binary":
        return _set_vectors(size_name, dtype), {}
    if profile_kind == "trace":
        return (_matrix(size_name, dtype),), {}
    if profile_kind == "linalg_matrix":
        return (_spd_matrix(size_name, dtype),), {}
    if profile_kind == "linalg_matrix_rhs":
        matrix = _spd_matrix(size_name, dtype)
        rhs = np.ones(matrix.shape[0], dtype=np.dtype(dtype))
        return (matrix, rhs), {}
    if profile_kind == "fft_vector":
        return (_fft_vector(size_name, dtype),), {}
    if profile_kind == "fft_matrix":
        return (_fft_matrix(size_name, dtype),), {}
    if profile_kind == "random_call":
        size = _RANDOM_SIZES[size_name]
        random_name = str(params["random_name"])
        extra_args = _RANDOM_EXTRA_ARGS.get(random_name, ())
        if random_name == "bytes":
            return (size,), {}
        if random_name == "random_integers":
            return (0, 100, size), {}
        if random_name == "shuffle":
            return (_choice_pool(size_name, dtype).copy(),), {}
        if random_name == "permutation":
            return (size,), {}
        if random_name == "choice":
            return (_choice_pool(size_name, dtype), size), {}
        if random_name == "dirichlet":
            return ([1.0, 2.0, 3.0], size), {}
        if random_name == "multinomial":
            return (10, [0.2, 0.3, 0.5], size), {}
        if random_name == "multivariate_normal":
            return (np.zeros(4), np.eye(4), size), {}
        return (*extra_args, size), {}
    if profile_kind in {
        "diff_like",
        "gradient",
        "convolve_like",
        "corrcoef_like",
        "cross_like",
        "histogram",
        "histogram2d",
        "histogramdd",
        "histogram_bin_edges",
        "digitize",
        "bincount",
        "interp",
        "trapezoid",
        "vander",
    }:
        return _misc_args(op_name, size_name, dtype)
    if profile_kind == "polynomial_call":
        return _polynomial_args(op_name, size_name, dtype)
    if profile_kind == "linalg_delegate":
        return _linalg_delegate_args(op_name, size_name, dtype)
    if profile_kind == "fft_freqs":
        return (_FFT_SIZES[size_name],), {"d": 0.5}
    if profile_kind == "fft_shift":
        return (_fft_vector(size_name, dtype),), {}
    if profile_kind in {"bitwise_unary", "bitwise_binary", "bitwise_shift"}:
        return _bitwise_args(op_name, size_name)
    if profile_kind in {"complex_unary", "complex_sort"}:
        return _complex_args(op_name, size_name, dtype)
    if profile_kind in {
        "array_unary_free",
        "astype_free",
        "atleast_free",
        "broadcast_arrays_free",
        "broadcast_shapes_free",
        "broadcast_to_free",
        "stack_free",
        "split_array",
        "shape_constructor_free",
        "like_constructor_free",
        "expand_dims_free",
        "eye_free",
        "dtype_relation_free",
        "array_predicate_free",
        "scalar_predicate_free",
        "memory_relation_free",
        "diag_indices_free",
        "diag_indices_from_free",
        "diagonal_free",
        "moveaxis_free",
        "reshape_free",
        "ravel_multi_index_free",
        "transpose_free",
        "tri_free",
        "triangular_matrix_free",
        "tri_indices_free",
        "tri_indices_from_free",
        "unravel_index_free",
    }:
        return _free_args(op_name, size_name, dtype)
    if profile_kind == "window":
        size = _VECTOR_SIZES[size_name]
        if op_name == "kaiser":
            return (size, 14.0), {}
        return (size,), {}

    raise ValueError(f"unsupported profile_kind: {profile_kind!r}")
