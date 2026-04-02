"""Public FLOP cost query API.

All cost functions are pure (shape params) -> int. They can be used for
pre-flight cost estimation without running any computation.
"""
# Existing
from mechestim._flops import einsum_cost, pointwise_cost, reduction_cost, svd_cost  # noqa: F401
from mechestim._symmetric import SymmetryInfo  # noqa: F401

# Linalg — decompositions
from mechestim.linalg._decompositions import (  # noqa: F401
    cholesky_cost, qr_cost, eig_cost, eigh_cost,
    eigvals_cost, eigvalsh_cost, svdvals_cost,
)

# Linalg — solvers
from mechestim.linalg._solvers import (  # noqa: F401
    solve_cost, inv_cost, lstsq_cost, pinv_cost,
    tensorsolve_cost, tensorinv_cost,
)

# Linalg — properties
from mechestim.linalg._properties import (  # noqa: F401
    trace_cost, det_cost, slogdet_cost, norm_cost,
    vector_norm_cost, matrix_norm_cost, cond_cost, matrix_rank_cost,
)

# Linalg — compound
from mechestim.linalg._compound import multi_dot_cost, matrix_power_cost  # noqa: F401

# FFT
from mechestim.fft._transforms import (  # noqa: F401
    fft_cost, rfft_cost, fftn_cost, rfftn_cost, hfft_cost,
)

# Polynomial
from mechestim._polynomial import (  # noqa: F401
    polyval_cost, polyadd_cost, polysub_cost, polymul_cost,
    polydiv_cost, polyfit_cost, poly_cost, roots_cost,
    polyder_cost, polyint_cost,
)

# Window
from mechestim._window import (  # noqa: F401
    bartlett_cost, blackman_cost, hamming_cost, hanning_cost, kaiser_cost,
)

# Other
from mechestim._unwrap import unwrap_cost  # noqa: F401

__all__ = [
    # Existing
    "einsum_cost", "pointwise_cost", "reduction_cost", "svd_cost",
    # Linalg
    "cholesky_cost", "qr_cost", "eig_cost", "eigh_cost",
    "eigvals_cost", "eigvalsh_cost", "svdvals_cost",
    "solve_cost", "inv_cost", "lstsq_cost", "pinv_cost",
    "tensorsolve_cost", "tensorinv_cost",
    "trace_cost", "det_cost", "slogdet_cost", "norm_cost",
    "vector_norm_cost", "matrix_norm_cost", "cond_cost", "matrix_rank_cost",
    "multi_dot_cost", "matrix_power_cost",
    # FFT
    "fft_cost", "rfft_cost", "fftn_cost", "rfftn_cost", "hfft_cost",
    # Polynomial
    "polyval_cost", "polyadd_cost", "polysub_cost", "polymul_cost",
    "polydiv_cost", "polyfit_cost", "poly_cost", "roots_cost",
    "polyder_cost", "polyint_cost",
    # Window
    "bartlett_cost", "blackman_cost", "hamming_cost", "hanning_cost", "kaiser_cost",
    # Other
    "unwrap_cost",
    # Symmetric
    "SymmetryInfo",
]
