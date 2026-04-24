"""Public FLOP cost query API.

All exported helpers return weighted FLOP costs by default, using the active
operation multipliers from ``whest._weights``. The underlying analytical
formulas remain available to the runtime in the private modules.
"""

from __future__ import annotations

from functools import wraps

from whest._flops import (
    analytical_pointwise_cost as _analytical_pointwise_cost,
)
from whest._flops import (
    analytical_reduction_cost as _analytical_reduction_cost,
)
from whest._flops import (
    einsum_cost as _analytical_einsum_cost,
)
from whest._flops import (
    svd_cost as _analytical_svd_cost,
)
from whest._polynomial import (
    poly_cost as _analytical_poly_cost,
)
from whest._polynomial import (
    polyadd_cost as _analytical_polyadd_cost,
)
from whest._polynomial import (
    polyder_cost as _analytical_polyder_cost,
)
from whest._polynomial import (
    polydiv_cost as _analytical_polydiv_cost,
)
from whest._polynomial import (
    polyfit_cost as _analytical_polyfit_cost,
)
from whest._polynomial import (
    polyint_cost as _analytical_polyint_cost,
)
from whest._polynomial import (
    polymul_cost as _analytical_polymul_cost,
)
from whest._polynomial import (
    polysub_cost as _analytical_polysub_cost,
)
from whest._polynomial import (
    polyval_cost as _analytical_polyval_cost,
)
from whest._polynomial import (
    roots_cost as _analytical_roots_cost,
)
from whest._symmetric import SymmetryInfo
from whest._unwrap import unwrap_cost as _analytical_unwrap_cost
from whest._weights import get_weight
from whest._window import (
    bartlett_cost as _analytical_bartlett_cost,
)
from whest._window import (
    blackman_cost as _analytical_blackman_cost,
)
from whest._window import (
    hamming_cost as _analytical_hamming_cost,
)
from whest._window import (
    hanning_cost as _analytical_hanning_cost,
)
from whest._window import (
    kaiser_cost as _analytical_kaiser_cost,
)
from whest.fft._transforms import (
    fft_cost as _analytical_fft_cost,
)
from whest.fft._transforms import (
    fftn_cost as _analytical_fftn_cost,
)
from whest.fft._transforms import (
    hfft_cost as _analytical_hfft_cost,
)
from whest.fft._transforms import (
    rfft_cost as _analytical_rfft_cost,
)
from whest.fft._transforms import (
    rfftn_cost as _analytical_rfftn_cost,
)
from whest.linalg._compound import (
    matrix_power_cost as _analytical_matrix_power_cost,
)
from whest.linalg._compound import (
    multi_dot_cost as _analytical_multi_dot_cost,
)
from whest.linalg._decompositions import (
    cholesky_cost as _analytical_cholesky_cost,
)
from whest.linalg._decompositions import (
    eig_cost as _analytical_eig_cost,
)
from whest.linalg._decompositions import (
    eigh_cost as _analytical_eigh_cost,
)
from whest.linalg._decompositions import (
    eigvals_cost as _analytical_eigvals_cost,
)
from whest.linalg._decompositions import (
    eigvalsh_cost as _analytical_eigvalsh_cost,
)
from whest.linalg._decompositions import (
    qr_cost as _analytical_qr_cost,
)
from whest.linalg._decompositions import (
    svdvals_cost as _analytical_svdvals_cost,
)
from whest.linalg._properties import (
    cond_cost as _analytical_cond_cost,
)
from whest.linalg._properties import (
    det_cost as _analytical_det_cost,
)
from whest.linalg._properties import (
    matrix_norm_cost as _analytical_matrix_norm_cost,
)
from whest.linalg._properties import (
    matrix_rank_cost as _analytical_matrix_rank_cost,
)
from whest.linalg._properties import (
    norm_cost as _analytical_norm_cost,
)
from whest.linalg._properties import (
    slogdet_cost as _analytical_slogdet_cost,
)
from whest.linalg._properties import (
    trace_cost as _analytical_trace_cost,
)
from whest.linalg._properties import (
    vector_norm_cost as _analytical_vector_norm_cost,
)
from whest.linalg._solvers import (
    inv_cost as _analytical_inv_cost,
)
from whest.linalg._solvers import (
    lstsq_cost as _analytical_lstsq_cost,
)
from whest.linalg._solvers import (
    pinv_cost as _analytical_pinv_cost,
)
from whest.linalg._solvers import (
    solve_cost as _analytical_solve_cost,
)
from whest.linalg._solvers import (
    tensorinv_cost as _analytical_tensorinv_cost,
)
from whest.linalg._solvers import (
    tensorsolve_cost as _analytical_tensorsolve_cost,
)


def _weight_cost(op_name: str, analytical_cost: int) -> int:
    """Convert an analytical FLOP count into the public weighted estimate.

    This intentionally mirrors ``BudgetContext.deduct()`` by flooring via
    ``int(...)`` after applying the weight multiplier, so public estimates and
    runtime accounting stay in sync.
    """
    return int(analytical_cost * get_weight(op_name))


def _make_weighted_cost(op_name: str, analytical_fn):
    @wraps(analytical_fn)
    def wrapper(*args, **kwargs):
        return _weight_cost(op_name, analytical_fn(*args, **kwargs))

    analytical_doc = analytical_fn.__doc__ or ""
    wrapper.__doc__ = (
        f"Weighted FLOP cost estimate for ``{op_name}``.\n\n"
        "This public helper returns the analytical cost formula multiplied by "
        "the active operation weight from ``whest._weights``. The final value "
        "is floored with ``int(...)`` to match runtime budget accounting.\n\n"
        "Analytical formula:\n"
        f"{analytical_doc}"
    )
    return wrapper


def pointwise_cost(
    op_name: str,
    *,
    shape: tuple[int, ...],
    symmetry_info: SymmetryInfo | None = None,
) -> int:
    """Weighted FLOP cost of a pointwise operation.

    Parameters
    ----------
    op_name : str
        Operation name used for weight lookup, e.g. ``"exp"`` or ``"add"``.
    shape : tuple of int
        Output shape of the pointwise operation.
    symmetry_info : SymmetryInfo or None, optional
        If provided, only unique elements are counted analytically before the
        operation weight is applied.

    Returns
    -------
    int
        Weighted public cost estimate, floored to match runtime accounting.
    """
    if not isinstance(op_name, str):
        raise TypeError("pointwise_cost() requires op_name as the first argument")
    return _weight_cost(
        op_name,
        _analytical_pointwise_cost(shape, symmetry_info=symmetry_info),
    )


def reduction_cost(
    op_name: str,
    *,
    input_shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
    symmetry_info: SymmetryInfo | None = None,
) -> int:
    """Weighted FLOP cost of a reduction operation.

    Parameters
    ----------
    op_name : str
        Operation name used for weight lookup, e.g. ``"sum"`` or ``"max"``.
    input_shape : tuple of int
        Shape of the reduction input.
    axis : int, tuple of int, or None, optional
        Reduction axis or axes. When a tuple is given, the cost is computed
        with those axes treated as the reduced set.
    symmetry_info : SymmetryInfo or None, optional
        If provided, symmetry is used to count unique outputs and inputs
        (see ``_flops.analytical_reduction_cost``).

    Returns
    -------
    int
        Weighted public cost estimate, floored to match runtime accounting.
    """
    if not isinstance(op_name, str):
        raise TypeError("reduction_cost() requires op_name as the first argument")
    return _weight_cost(
        op_name,
        _analytical_reduction_cost(
            input_shape,
            axis=axis,
            symmetry_info=symmetry_info,
        ),
    )


einsum_cost = _make_weighted_cost("einsum", _analytical_einsum_cost)
svd_cost = _make_weighted_cost("linalg.svd", _analytical_svd_cost)

polyval_cost = _make_weighted_cost("polyval", _analytical_polyval_cost)
polyadd_cost = _make_weighted_cost("polyadd", _analytical_polyadd_cost)
polysub_cost = _make_weighted_cost("polysub", _analytical_polysub_cost)
polymul_cost = _make_weighted_cost("polymul", _analytical_polymul_cost)
polydiv_cost = _make_weighted_cost("polydiv", _analytical_polydiv_cost)
polyfit_cost = _make_weighted_cost("polyfit", _analytical_polyfit_cost)
poly_cost = _make_weighted_cost("poly", _analytical_poly_cost)
roots_cost = _make_weighted_cost("roots", _analytical_roots_cost)
polyder_cost = _make_weighted_cost("polyder", _analytical_polyder_cost)
polyint_cost = _make_weighted_cost("polyint", _analytical_polyint_cost)

unwrap_cost = _make_weighted_cost("unwrap", _analytical_unwrap_cost)

bartlett_cost = _make_weighted_cost("bartlett", _analytical_bartlett_cost)
blackman_cost = _make_weighted_cost("blackman", _analytical_blackman_cost)
hamming_cost = _make_weighted_cost("hamming", _analytical_hamming_cost)
hanning_cost = _make_weighted_cost("hanning", _analytical_hanning_cost)
kaiser_cost = _make_weighted_cost("kaiser", _analytical_kaiser_cost)

fft_cost = _make_weighted_cost("fft.fft", _analytical_fft_cost)
rfft_cost = _make_weighted_cost("fft.rfft", _analytical_rfft_cost)
fftn_cost = _make_weighted_cost("fft.fftn", _analytical_fftn_cost)
rfftn_cost = _make_weighted_cost("fft.rfftn", _analytical_rfftn_cost)
hfft_cost = _make_weighted_cost("fft.hfft", _analytical_hfft_cost)

multi_dot_cost = _make_weighted_cost("linalg.multi_dot", _analytical_multi_dot_cost)
matrix_power_cost = _make_weighted_cost(
    "linalg.matrix_power",
    _analytical_matrix_power_cost,
)

cholesky_cost = _make_weighted_cost("linalg.cholesky", _analytical_cholesky_cost)
qr_cost = _make_weighted_cost("linalg.qr", _analytical_qr_cost)
eig_cost = _make_weighted_cost("linalg.eig", _analytical_eig_cost)
eigh_cost = _make_weighted_cost("linalg.eigh", _analytical_eigh_cost)
eigvals_cost = _make_weighted_cost("linalg.eigvals", _analytical_eigvals_cost)
eigvalsh_cost = _make_weighted_cost("linalg.eigvalsh", _analytical_eigvalsh_cost)
svdvals_cost = _make_weighted_cost("linalg.svdvals", _analytical_svdvals_cost)

trace_cost = _make_weighted_cost("linalg.trace", _analytical_trace_cost)
det_cost = _make_weighted_cost("linalg.det", _analytical_det_cost)
slogdet_cost = _make_weighted_cost("linalg.slogdet", _analytical_slogdet_cost)
norm_cost = _make_weighted_cost("linalg.norm", _analytical_norm_cost)
vector_norm_cost = _make_weighted_cost(
    "linalg.vector_norm",
    _analytical_vector_norm_cost,
)
matrix_norm_cost = _make_weighted_cost(
    "linalg.matrix_norm",
    _analytical_matrix_norm_cost,
)
cond_cost = _make_weighted_cost("linalg.cond", _analytical_cond_cost)
matrix_rank_cost = _make_weighted_cost(
    "linalg.matrix_rank",
    _analytical_matrix_rank_cost,
)

solve_cost = _make_weighted_cost("linalg.solve", _analytical_solve_cost)
inv_cost = _make_weighted_cost("linalg.inv", _analytical_inv_cost)
lstsq_cost = _make_weighted_cost("linalg.lstsq", _analytical_lstsq_cost)
pinv_cost = _make_weighted_cost("linalg.pinv", _analytical_pinv_cost)
tensorsolve_cost = _make_weighted_cost(
    "linalg.tensorsolve",
    _analytical_tensorsolve_cost,
)
tensorinv_cost = _make_weighted_cost("linalg.tensorinv", _analytical_tensorinv_cost)

__all__ = [
    # Existing
    "einsum_cost",
    "pointwise_cost",
    "reduction_cost",
    "svd_cost",
    # Linalg
    "cholesky_cost",
    "qr_cost",
    "eig_cost",
    "eigh_cost",
    "eigvals_cost",
    "eigvalsh_cost",
    "svdvals_cost",
    "solve_cost",
    "inv_cost",
    "lstsq_cost",
    "pinv_cost",
    "tensorsolve_cost",
    "tensorinv_cost",
    "trace_cost",
    "det_cost",
    "slogdet_cost",
    "norm_cost",
    "vector_norm_cost",
    "matrix_norm_cost",
    "cond_cost",
    "matrix_rank_cost",
    "multi_dot_cost",
    "matrix_power_cost",
    # FFT
    "fft_cost",
    "rfft_cost",
    "fftn_cost",
    "rfftn_cost",
    "hfft_cost",
    # Polynomial
    "polyval_cost",
    "polyadd_cost",
    "polysub_cost",
    "polymul_cost",
    "polydiv_cost",
    "polyfit_cost",
    "poly_cost",
    "roots_cost",
    "polyder_cost",
    "polyint_cost",
    # Window
    "bartlett_cost",
    "blackman_cost",
    "hamming_cost",
    "hanning_cost",
    "kaiser_cost",
    # Other
    "unwrap_cost",
    # Symmetric
    "SymmetryInfo",
]
