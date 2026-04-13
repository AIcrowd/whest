"""scipy.stats-compatible distributions with FLOP counting.

This submodule provides a **subset of scipy.stats** continuous distributions,
each with ``.pdf()``, ``.cdf()``, and ``.ppf()`` methods that match the
scipy API exactly (same function signatures, same numerical results).

Unlike NumPy operations which are direct wrappers, these functions are
**not** part of NumPy. They reproduce the ``scipy.stats`` interface so that
participants can use standard statistical distributions without importing
scipy (which is not available in the sandbox).

Available distributions
-----------------------
===============  ===========================================
``norm``         Normal (Gaussian) distribution
``uniform``      Continuous uniform distribution
``expon``        Exponential distribution
``cauchy``       Cauchy (Lorentz) distribution
``logistic``     Logistic distribution
``laplace``      Laplace (double-exponential) distribution
``lognorm``      Log-normal distribution
``truncnorm``    Truncated normal distribution
===============  ===========================================

Usage
-----
All distributions use the scipy ``loc``/``scale`` parameterisation::

    import mechestim as me

    me.stats.norm.pdf(0.0)                    # standard normal PDF at 0
    me.stats.norm.cdf(1.96, loc=0, scale=1)   # ≈ 0.975
    me.stats.expon.ppf(0.5, scale=2.0)        # median of Exp(rate=0.5)

FLOP costs
----------
Each method deducts a **flat FLOP cost per element** from the active budget.
Costs are documented in each method's docstring. Internal sub-operations
(exp, log, etc.) are computed via raw NumPy and do **not** incur additional
FLOP charges.

Compatibility
-------------
Outputs are verified against ``scipy.stats`` to within 1e-12 relative
tolerance across the full input domain. See ``tests/test_stats_*.py``.
"""

from mechestim._registry import make_module_getattr as _make_module_getattr
from mechestim.stats._cauchy import cauchy  # noqa: F401
from mechestim.stats._expon import expon  # noqa: F401
from mechestim.stats._laplace import laplace  # noqa: F401
from mechestim.stats._logistic import logistic  # noqa: F401
from mechestim.stats._lognorm import lognorm  # noqa: F401
from mechestim.stats._norm import norm  # noqa: F401
from mechestim.stats._truncnorm import truncnorm  # noqa: F401
from mechestim.stats._uniform import uniform  # noqa: F401

__all__ = [
    "cauchy",
    "expon",
    "laplace",
    "logistic",
    "lognorm",
    "norm",
    "truncnorm",
    "uniform",
]

__getattr__ = _make_module_getattr(
    module_prefix="stats.", module_label="mechestim.stats"
)
