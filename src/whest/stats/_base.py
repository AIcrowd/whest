"""Base class for scipy-compatible continuous distributions."""

from __future__ import annotations

import numpy as _np

from whest._ndarray import _aswhest
from whest._validation import require_budget


class ContinuousDistribution:
    """Base for scipy-compatible continuous distributions with FLOP counting.

    Subclasses implement ``_compute_pdf``, ``_compute_cdf``, ``_compute_ppf``
    as pure-NumPy helpers, then call :meth:`_deduct_and_call` to wrap them
    with budget deduction and WhestArray conversion.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _deduct_and_call(self, method: str, cost_per_elem: int, x, *args, **kwargs):
        """Deduct FLOPs then call the pure-numpy implementation.

        Parameters
        ----------
        method : str
            Method name for budget logging, e.g. ``"pdf"``.
        cost_per_elem : int
            Flat FLOP cost per output element.
        x : array_like
            Primary input array (determines output size).
        *args, **kwargs
            Forwarded to ``_compute_{method}``.
        """
        budget = require_budget()
        x = _np.asarray(x, dtype=_np.float64)
        n = max(x.size, 1)
        op_name = f"stats.{self._name}.{method}"
        budget.deduct(
            op_name,
            flop_cost=cost_per_elem * n,
            subscripts=None,
            shapes=(x.shape,),
        )
        compute_fn = getattr(self, f"_compute_{method}")
        result = compute_fn(x, *args, **kwargs)
        return _aswhest(result)

    def __repr__(self) -> str:
        return f"<whest.stats.{self._name}>"
