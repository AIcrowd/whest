"""Linear algebra submodule for mechestim."""
from mechestim.linalg._svd import svd  # noqa: F401
from mechestim.linalg._aliases import (  # noqa: F401
    matmul, cross, outer, tensordot, vecdot, diagonal, matrix_transpose,
)
from mechestim.linalg._decompositions import (  # noqa: F401
    cholesky, qr, eig, eigh, eigvals, eigvalsh, svdvals,
    cholesky_cost, qr_cost, eig_cost, eigh_cost, eigvals_cost, eigvalsh_cost, svdvals_cost,
)
from mechestim.linalg._solvers import (  # noqa: F401
    solve, inv, lstsq, pinv, tensorsolve, tensorinv,
    solve_cost, inv_cost, lstsq_cost, pinv_cost, tensorsolve_cost, tensorinv_cost,
)
from mechestim.linalg._properties import (  # noqa: F401
    trace, det, slogdet, norm, vector_norm, matrix_norm, cond, matrix_rank,
    trace_cost, det_cost, slogdet_cost, norm_cost, vector_norm_cost,
    matrix_norm_cost, cond_cost, matrix_rank_cost,
)
from mechestim.linalg._compound import (  # noqa: F401
    multi_dot, matrix_power,
    multi_dot_cost, matrix_power_cost,
)
from mechestim._registry import make_module_getattr as _make_module_getattr

__all__ = [
    "svd",
    "matmul", "cross", "outer", "tensordot", "vecdot",
    "diagonal", "matrix_transpose",
    "cholesky", "qr", "eig", "eigh", "eigvals", "eigvalsh", "svdvals",
    "solve", "inv", "lstsq", "pinv", "tensorsolve", "tensorinv",
    "trace", "det", "slogdet", "norm", "vector_norm", "matrix_norm", "cond", "matrix_rank",
    "multi_dot", "matrix_power",
]

__getattr__ = _make_module_getattr(module_prefix="linalg.", module_label="mechestim.linalg")
