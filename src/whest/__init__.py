"""whest: NumPy-compatible math primitives with FLOP counting.

Usage::

    import whest as me

    with me.BudgetContext(flop_budget=1_000_000) as budget:
        W = me.array(weight_matrix)
        h = me.einsum('ij,j->i', W, x)
        h = me.maximum(h, 0)
        print(budget.summary())
"""

import importlib as _importlib

import numpy as _np

from whest._registry import REGISTRY_META as _REGISTRY_META

__version__ = f"0.2.0+np{_np.__version__}"
__numpy_version__ = _np.__version__
__numpy_pinned__ = _REGISTRY_META["numpy_version"]
__numpy_supported__ = _REGISTRY_META.get("numpy_supported", ">=2.0.0,<2.5.0")

from whest._version_check import check_numpy_version as _check_numpy_version

_check_numpy_version(__numpy_supported__)

# --- Budget and diagnostics ---
from whest._budget import (  # noqa: F401
    BudgetContext,
    OpRecord,
    budget,
    budget_reset,
    budget_summary_dict,
    namespace,
)
from whest._config import configure  # noqa: F401

# --- Counting, histogram & generation ops (counted) ---
from whest._counting_ops import (  # noqa: F401
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
from whest._display import budget_live, budget_summary  # noqa: F401

# --- Einsum ---
from whest._einsum import (  # noqa: F401
    clear_einsum_cache,
    einsum,
    einsum_cache_info,
    einsum_path,
)

# --- Free ops ---
from whest._free_ops import (  # noqa: F401
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

# --- Path optimization types ---
from whest._opt_einsum import PathInfo, StepInfo  # noqa: F401

# --- Permutation groups ---
from whest._perm_group import SymmetryGroup  # noqa: F401

# --- Pointwise (counted) ---
from whest._pointwise import (  # noqa: F401
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
from whest._polynomial import (  # noqa: F401
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
from whest._sorting_ops import (  # noqa: F401
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

# --- Symmetric tensor ---
from whest._symmetric import (  # noqa: F401
    SymmetricTensor,
    as_symmetric,
    is_symmetric,
    symmetrize,
)

# --- Unwrap (counted) ---
from whest._unwrap import unwrap  # noqa: F401

# --- Window functions (counted) ---
from whest._window import (  # noqa: F401
    bartlett,
    blackman,
    hamming,
    hanning,
    kaiser,
)

# --- Errors ---
from whest.errors import (  # noqa: F401
    BudgetExhaustedError,
    NoBudgetContextError,
    SymmetryError,
    SymmetryLossWarning,
    TimeExhaustedError,
    UnsupportedFunctionError,
    WhestError,
    WhestWarning,
)

# --- NumPy constants and types ---
# _np already imported above for version attributes

pi = _np.pi
e = _np.e
inf = _np.inf
nan = _np.nan
newaxis = _np.newaxis
from whest._ndarray import WhestArray as ndarray  # noqa: E402, F401

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

# --- Missing dtype types and numpy utilities (re-exported as free) ---
from whest import typing  # noqa: F401, E402
from whest._dtypes import (  # noqa: F401, E402
    dtype,
    floating,
    integer,
    number,
    uint16,
    uint32,
    uint64,
)
from whest._errstate import (  # noqa: F401, E402
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
from whest._registry import make_module_getattr as _make_module_getattr
from whest._type_info import finfo, iinfo  # noqa: F401, E402

_LAZY_SUBMODULES = frozenset({"fft", "flops", "linalg", "random", "stats", "testing"})
_registry_getattr = _make_module_getattr(module_prefix="", module_label="whest")


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f"whest.{name}")
        globals()[name] = module
        return module
    return _registry_getattr(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)
