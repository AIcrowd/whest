"""Linear algebra submodule for mechestim."""
from mechestim.linalg._svd import svd  # noqa: F401
from mechestim.linalg._aliases import (  # noqa: F401
    matmul, cross, outer, tensordot, vecdot, diagonal, matrix_transpose,
)
from mechestim._registry import make_module_getattr as _make_module_getattr

__all__ = [
    "svd",
    "matmul", "cross", "outer", "tensordot", "vecdot",
    "diagonal", "matrix_transpose",
]

__getattr__ = _make_module_getattr(module_prefix="linalg.", module_label="mechestim.linalg")
