"""scipy.stats-compatible distributions with FLOP counting."""

from mechestim._registry import make_module_getattr as _make_module_getattr
from mechestim.stats._norm import norm  # noqa: F401

__all__ = ["norm"]

__getattr__ = _make_module_getattr(
    module_prefix="stats.", module_label="mechestim.stats"
)
