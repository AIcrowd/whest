"""Dtype introspection utilities re-exported as free (0-FLOP) helpers.

`finfo` and `iinfo` are metadata queries that return info objects
(eps, max, min, bits, etc.). They perform no computation.
"""

from __future__ import annotations

import numpy as _np

finfo = _np.finfo
iinfo = _np.iinfo
