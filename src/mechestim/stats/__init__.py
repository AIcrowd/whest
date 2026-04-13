"""scipy.stats-compatible distributions with FLOP counting."""

from mechestim._registry import make_module_getattr as _make_module_getattr
from mechestim.stats._cauchy import cauchy  # noqa: F401
from mechestim.stats._expon import expon  # noqa: F401
from mechestim.stats._laplace import laplace  # noqa: F401
from mechestim.stats._logistic import logistic  # noqa: F401
from mechestim.stats._norm import norm  # noqa: F401
from mechestim.stats._uniform import uniform  # noqa: F401

__all__ = [
    "cauchy",
    "expon",
    "laplace",
    "logistic",
    "norm",
    "uniform",
]

__getattr__ = _make_module_getattr(
    module_prefix="stats.", module_label="mechestim.stats"
)
