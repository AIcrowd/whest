"""Normal (Gaussian) distribution with FLOP counting.

Mimics ``scipy.stats.norm`` — see
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
"""

from __future__ import annotations

import numpy as np

from mechestim.stats._base import ContinuousDistribution
from mechestim.stats._erf import _erf
from mechestim.stats._ndtri import _ndtri

_NORM_PDF_COST = 10
_NORM_CDF_COST = 20
_NORM_PPF_COST = 40

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


class NormDistribution(ContinuousDistribution):
    """Normal (Gaussian) continuous random variable.

    Equivalent to ``scipy.stats.norm``.  Uses the standard ``loc`` /
    ``scale`` parameterisation (mean and standard deviation).

    Methods
    -------
    pdf(x, loc=0, scale=1)
        Probability density function.
    cdf(x, loc=0, scale=1)
        Cumulative distribution function.
    ppf(q, loc=0, scale=1)
        Percent-point function (inverse of CDF).
    """

    def __init__(self):
        super().__init__("norm")

    def pdf(self, x, loc=0, scale=1):
        """Probability density function at *x*.

        Equivalent to ``scipy.stats.norm.pdf(x, loc, scale)``.

        FLOP Cost
        ---------
        10 * numel(x) FLOPs

        Parameters
        ----------
        x : array_like
            Quantiles.
        loc : float, optional
            Mean of the distribution (default 0).
        scale : float, optional
            Standard deviation (default 1).

        Returns
        -------
        MechestimArray
            PDF evaluated at *x*.
        """
        return self._deduct_and_call("pdf", _NORM_PDF_COST, x, loc=loc, scale=scale)

    def cdf(self, x, loc=0, scale=1):
        """Cumulative distribution function at *x*.

        Equivalent to ``scipy.stats.norm.cdf(x, loc, scale)``.

        FLOP Cost
        ---------
        20 * numel(x) FLOPs

        Parameters
        ----------
        x : array_like
            Quantiles.
        loc : float, optional
            Mean of the distribution (default 0).
        scale : float, optional
            Standard deviation (default 1).

        Returns
        -------
        MechestimArray
            CDF evaluated at *x*.
        """
        return self._deduct_and_call("cdf", _NORM_CDF_COST, x, loc=loc, scale=scale)

    def ppf(self, q, loc=0, scale=1):
        """Percent-point function (inverse CDF) at *q*.

        Equivalent to ``scipy.stats.norm.ppf(q, loc, scale)``.

        FLOP Cost
        ---------
        40 * numel(q) FLOPs

        Parameters
        ----------
        q : array_like
            Quantiles in [0, 1].
        loc : float, optional
            Mean of the distribution (default 0).
        scale : float, optional
            Standard deviation (default 1).

        Returns
        -------
        MechestimArray
            PPF evaluated at *q*.
        """
        return self._deduct_and_call("ppf", _NORM_PPF_COST, q, loc=loc, scale=scale)

    # --- Pure-NumPy implementations (no budget deduction) ---

    def _compute_pdf(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return _INV_SQRT_2PI / scale * np.exp(-0.5 * z * z)

    def _compute_cdf(self, x, loc=0, scale=1):
        z = (x - loc) / (scale * _SQRT2)
        return 0.5 * (1.0 + _erf(z))

    def _compute_ppf(self, q, loc=0, scale=1):
        return loc + scale * _ndtri(q)


norm = NormDistribution()
