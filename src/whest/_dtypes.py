"""NumPy dtype types re-exported as free (0-FLOP) attributes.

These are plain aliases — they perform no FLOP tracking because they
are type objects, not operations.
"""

from __future__ import annotations

import numpy as _np

# Abstract dtype hierarchy
floating = _np.floating
integer = _np.integer
number = _np.number
dtype = _np.dtype

# Missing unsigned integer dtypes (uint8 is already exported from __init__.py)
uint16 = _np.uint16
uint32 = _np.uint32
uint64 = _np.uint64
