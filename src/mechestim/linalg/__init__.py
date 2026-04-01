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
from mechestim._registry import make_module_getattr as _make_module_getattr

__all__ = [
    "svd",
    "matmul", "cross", "outer", "tensordot", "vecdot",
    "diagonal", "matrix_transpose",
    "cholesky", "qr", "eig", "eigh", "eigvals", "eigvalsh", "svdvals",
    "solve", "inv", "lstsq", "pinv", "tensorsolve", "tensorinv",
]

__getattr__ = _make_module_getattr(module_prefix="linalg.", module_label="mechestim.linalg")
