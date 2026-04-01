"""mechestim: NumPy-compatible math primitives with FLOP counting.

Usage::

    import mechestim as me

    with me.BudgetContext(flop_budget=1_000_000) as budget:
        W = me.array(weight_matrix)
        h = me.einsum('ij,j->i', W, x)
        h = me.maximum(h, 0)
        print(budget.summary())
"""

import numpy as _np
from mechestim._registry import REGISTRY_META as _REGISTRY_META

__version__ = f"0.2.0+np{_np.__version__}"
__numpy_version__ = _np.__version__
__numpy_pinned__ = _REGISTRY_META["numpy_version"]

from mechestim._version_check import check_numpy_version as _check_numpy_version
_check_numpy_version(__numpy_pinned__)

# --- Budget and diagnostics ---
from mechestim._budget import BudgetContext, OpRecord  # noqa: F401

# --- Errors ---
from mechestim.errors import (  # noqa: F401
    BudgetExhaustedError,
    MechEstimError,
    MechEstimWarning,
    NoBudgetContextError,
    SymmetryError,
)

# --- Einsum ---
from mechestim._einsum import einsum  # noqa: F401

# --- Pointwise (counted) ---
from mechestim._pointwise import (  # noqa: F401
    abs,
    add,
    argmax,
    argmin,
    ceil,
    clip,
    cos,
    cumprod,
    cumsum,
    divide,
    dot,
    exp,
    floor,
    log,
    log2,
    log10,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    mod,
    multiply,
    negative,
    power,
    prod,
    sign,
    sin,
    sqrt,
    square,
    std,
    subtract,
    sum,
    tanh,
    var,
)

# --- Free ops ---
from mechestim._free_ops import (  # noqa: F401
    allclose,
    arange,
    argsort,
    array,
    asarray,
    astype,
    broadcast_to,
    concatenate,
    copy,
    diag,
    diagonal,
    empty,
    empty_like,
    expand_dims,
    eye,
    flip,
    full,
    full_like,
    hstack,
    hsplit,
    identity,
    isfinite,
    isinf,
    isnan,
    linspace,
    meshgrid,
    moveaxis,
    ones,
    ones_like,
    pad,
    ravel,
    repeat,
    reshape,
    roll,
    searchsorted,
    sort,
    split,
    squeeze,
    stack,
    swapaxes,
    tile,
    trace,
    transpose,
    tril,
    triu,
    unique,
    vsplit,
    vstack,
    where,
    zeros,
    zeros_like,
)

# --- Submodules ---
from mechestim import fft  # noqa: F401
from mechestim import flops  # noqa: F401
from mechestim import linalg  # noqa: F401
from mechestim import random  # noqa: F401

# --- NumPy constants and types ---
# _np already imported above for version attributes

pi = _np.pi
e = _np.e
inf = _np.inf
nan = _np.nan
newaxis = _np.newaxis
ndarray = _np.ndarray
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


from mechestim._registry import make_module_getattr as _make_module_getattr

__getattr__ = _make_module_getattr(module_prefix="", module_label="mechestim")
