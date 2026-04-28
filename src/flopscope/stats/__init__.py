"""Continuous probability distributions with analytic FLOP accounting.

``flopscope.stats`` provides a focused subset of ``scipy.stats`` continuous
distributions. Each exported distribution object exposes ``pdf``, ``cdf``,
and ``ppf`` methods with SciPy-compatible signatures while charging a flat
FLOP cost per output element to the active budget.

Available distributions
-----------------------
norm
    Normal (Gaussian) distribution.
uniform
    Continuous uniform distribution.
expon
    Exponential distribution.
cauchy
    Cauchy (Lorentz) distribution.
logistic
    Logistic distribution.
laplace
    Laplace (double-exponential) distribution.
lognorm
    Log-normal distribution.
truncnorm
    Truncated normal distribution.

Notes
-----
All distributions use SciPy's ``loc``/``scale`` parameterization. Shape
parameters, when present, precede ``loc`` and ``scale`` exactly as they do in
``scipy.stats``. Each public method requires an active
``flopscope.BudgetContext`` and deducts the documented flat FLOP charge before
returning a ``FlopscopeArray`` result.

Examples
--------
>>> import flopscope as flops
>>> with flops.BudgetContext(flop_budget=32) as budget:
...     probs = flops.stats.norm.cdf([0.0, 1.96])
...     summary = budget.summary()
"""

from flopscope._registry import make_module_getattr as _make_module_getattr
from flopscope.stats._cauchy import cauchy  # noqa: F401
from flopscope.stats._expon import expon  # noqa: F401
from flopscope.stats._laplace import laplace  # noqa: F401
from flopscope.stats._logistic import logistic  # noqa: F401
from flopscope.stats._lognorm import lognorm  # noqa: F401
from flopscope.stats._norm import norm  # noqa: F401
from flopscope.stats._truncnorm import truncnorm  # noqa: F401
from flopscope.stats._uniform import uniform  # noqa: F401

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
    module_prefix="stats.", module_label="flopscope.stats"
)
