"""NumPy runtime configuration and iteration utilities re-exported as free.

These are all pure numpy state management and iteration helpers.
None perform FLOP-costing operations.
"""

from __future__ import annotations

import numpy as _np

# Error state (context manager + getters/setters for numpy warnings)
errstate = _np.errstate
seterr = _np.seterr
geterr = _np.geterr

# Iteration utilities
ndindex = _np.ndindex
ndenumerate = _np.ndenumerate
broadcast = _np.broadcast
nditer = _np.nditer

# Print formatting
set_printoptions = _np.set_printoptions
get_printoptions = _np.get_printoptions
printoptions = _np.printoptions
