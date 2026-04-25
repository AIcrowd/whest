"""flopscope.numpy — counted drop-in for numpy (JAX-style).

Usage::

    import flopscope.numpy as fnp

    a = fnp.array([1, 2, 3])
    b = fnp.einsum('ij,j->i', W, x)
    u, s, vt = fnp.linalg.svd(M)

All operations exposed here are counted against the currently active
:class:`~flopscope.BudgetContext`. Accessing an attribute that flopscope
does not implement raises :class:`AttributeError` — there is no
transparent fallback to ``numpy``. This is deliberate: silently exposing
uncounted numpy operations would defeat the purpose of FLOP accounting.
If you need a raw numpy operation, import ``numpy`` directly.
"""

import importlib as _importlib

import numpy as _np

# --- Flopscope array type, re-exposed under the numpy-shaped name ---
from flopscope._ndarray import FlopscopeArray as ndarray  # noqa: F401

# --- Counting, histogram & generation ops (counted) ---
from flopscope._counting_ops import (  # noqa: F401
    allclose,
    apply_along_axis,
    apply_over_axes,
    array_equal,
    array_equiv,
    bincount,
    geomspace,
    histogram,
    histogram2d,
    histogram_bin_edges,
    histogramdd,
    logspace,
    piecewise,
    trace,
    vander,
)

# --- Einsum ---
from flopscope._einsum import (  # noqa: F401
    clear_einsum_cache,
    einsum,
    einsum_cache_info,
    einsum_path,
)

# --- Free ops ---
from flopscope._free_ops import (  # noqa: F401
    append,
    arange,
    argwhere,
    array,
    array_split,
    asarray,
    asarray_chkfinite,
    astype,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    base_repr,
    binary_repr,
    block,
    bmat,
    broadcast_arrays,
    broadcast_shapes,
    broadcast_to,
    can_cast,
    choose,
    column_stack,
    common_type,
    compress,
    concat,
    concatenate,
    copy,
    copyto,
    delete,
    diag,
    diag_indices,
    diag_indices_from,
    diagflat,
    diagonal,
    dsplit,
    dstack,
    empty,
    empty_like,
    expand_dims,
    extract,
    eye,
    fill_diagonal,
    flatnonzero,
    flip,
    fliplr,
    flipud,
    from_dlpack,
    frombuffer,
    fromfile,
    fromfunction,
    fromiter,
    fromregex,
    fromstring,
    full,
    full_like,
    hsplit,
    hstack,
    identity,
    indices,
    insert,
    isdtype,
    isfinite,
    isfortran,
    isinf,
    isnan,
    isscalar,
    issubdtype,
    iterable,
    ix_,
    linspace,
    mask_indices,
    matrix_transpose,
    may_share_memory,
    meshgrid,
    min_scalar_type,
    mintypecode,
    moveaxis,
    ndim,
    nonzero,
    ones,
    ones_like,
    packbits,
    pad,
    permute_dims,
    place,
    promote_types,
    put,
    put_along_axis,
    putmask,
    ravel,
    ravel_multi_index,
    repeat,
    require,
    reshape,
    resize,
    result_type,
    roll,
    rollaxis,
    rot90,
    row_stack,
    select,
    shape,
    shares_memory,
    size,
    split,
    squeeze,
    stack,
    swapaxes,
    take,
    take_along_axis,
    tile,
    transpose,
    tri,
    tril,
    tril_indices,
    tril_indices_from,
    trim_zeros,
    triu,
    triu_indices,
    triu_indices_from,
    typename,
    unpackbits,
    unravel_index,
    unstack,
    vsplit,
    vstack,
    where,
    zeros,
    zeros_like,
)

# --- Pointwise (counted) ---
from flopscope._pointwise import (  # noqa: F401
    abs,
    absolute,
    acos,
    acosh,
    add,
    all,
    amax,
    amin,
    angle,
    any,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    argmax,
    argmin,
    around,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    average,
    bitwise_and,
    bitwise_count,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_not,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    cbrt,
    ceil,
    clip,
    conj,
    conjugate,
    convolve,
    copysign,
    corrcoef,
    correlate,
    cos,
    cosh,
    count_nonzero,
    cov,
    cross,
    cumprod,
    cumsum,
    cumulative_prod,
    cumulative_sum,
    deg2rad,
    degrees,
    diff,
    divide,
    divmod,
    dot,
    ediff1d,
    equal,
    exp,
    exp2,
    expm1,
    fabs,
    fix,
    float_power,
    floor,
    floor_divide,
    fmax,
    fmin,
    fmod,
    frexp,
    gcd,
    gradient,
    greater,
    greater_equal,
    heaviside,
    hypot,
    i0,
    imag,
    inner,
    interp,
    invert,
    isclose,
    iscomplex,
    iscomplexobj,
    isnat,
    isneginf,
    isposinf,
    isreal,
    isrealobj,
    kron,
    lcm,
    ldexp,
    left_shift,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    matmul,
    matvec,
    max,
    maximum,
    mean,
    median,
    min,
    minimum,
    mod,
    modf,
    multiply,
    nan_to_num,
    nanargmax,
    nanargmin,
    nancumprod,
    nancumsum,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanpercentile,
    nanprod,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
    negative,
    nextafter,
    not_equal,
    outer,
    percentile,
    positive,
    pow,
    power,
    prod,
    ptp,
    quantile,
    rad2deg,
    radians,
    real,
    real_if_close,
    reciprocal,
    remainder,
    right_shift,
    rint,
    round,
    sign,
    signbit,
    sin,
    sinc,
    sinh,
    sort_complex,
    spacing,
    sqrt,
    square,
    std,
    subtract,
    sum,
    tan,
    tanh,
    tensordot,
    trapezoid,
    trapz,
    true_divide,
    trunc,
    var,
    vdot,
    vecdot,
    vecmat,
)

# --- Polynomial (counted) ---
from flopscope._polynomial import (  # noqa: F401
    poly,
    polyadd,
    polyder,
    polydiv,
    polyfit,
    polyint,
    polymul,
    polysub,
    polyval,
    roots,
)

# --- Sorting, search & set ops (counted) ---
from flopscope._sorting_ops import (  # noqa: F401
    argpartition,
    argsort,
    digitize,
    in1d,
    intersect1d,
    isin,
    lexsort,
    partition,
    searchsorted,
    setdiff1d,
    setxor1d,
    sort,
    union1d,
    unique,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
)

# --- Unwrap (counted) ---
from flopscope._unwrap import unwrap  # noqa: F401

# --- Window functions (counted) ---
from flopscope._window import (  # noqa: F401
    bartlett,
    blackman,
    hamming,
    hanning,
    kaiser,
)

# --- dtype utilities ---
from flopscope._dtypes import (  # noqa: F401
    dtype,
    floating,
    integer,
    number,
    uint16,
    uint32,
    uint64,
)
from flopscope._type_info import finfo, iinfo  # noqa: F401
from flopscope._errstate import (  # noqa: F401
    broadcast,
    errstate,
    get_printoptions,
    geterr,
    ndenumerate,
    ndindex,
    nditer,
    printoptions,
    set_printoptions,
    seterr,
)

# --- Numpy constants ---
pi = _np.pi
e = _np.e
inf = _np.inf
nan = _np.nan
newaxis = _np.newaxis

float16 = _np.float16
float32 = _np.float32
float64 = _np.float64
int8 = _np.int8
int16 = _np.int16
int32 = _np.int32
int64 = _np.int64
uint8 = _np.uint8
bool_ = _np.bool_
complex64 = _np.complex64
complex128 = _np.complex128

# --- Registry-aware __getattr__ ---
# Strict policy: any name not explicitly implemented above (and not in the
# registry as a known-classified op) raises AttributeError. We intentionally
# do NOT fall through to ``numpy`` — silently exposing uncounted numpy ops
# would defeat the purpose of FLOP accounting.
from flopscope._registry import make_module_getattr as _make_module_getattr

_LAZY_SUBMODULES = frozenset({"linalg", "fft", "random", "testing", "typing"})
_registry_getattr = _make_module_getattr(module_prefix="", module_label="flopscope.numpy")


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f"flopscope.numpy.{name}")
        globals()[name] = module
        return module
    return _registry_getattr(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)
