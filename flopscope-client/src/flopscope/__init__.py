"""flopscope — transparent proxy to a remote flopscope server.

This client package mirrors the core library's JAX-style public API::

    import flopscope as flops
    import flopscope.numpy as fnp

    with flops.BudgetContext(flop_budget=1_000_000) as ctx:
        a = fnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = fnp.zeros((2, 2))
        c = fnp.add(a, b)

Top-level :mod:`flopscope` exposes flopscope-specific primitives
(budget, symmetry, errors, perm groups).  Numpy-shaped operations live
under :mod:`flopscope.numpy` and dispatch to the remote server over ZMQ.
"""

from __future__ import annotations

import importlib as _importlib

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Primitives (flopscope-specific)
# ---------------------------------------------------------------------------
from flopscope._budget import (  # noqa: F401
    BudgetContext,
    OpRecord,
    budget,
    budget_summary_dict,
)
from flopscope._display import budget_live, budget_summary  # noqa: F401
from flopscope._perm_group import Cycle, Permutation, PermutationGroup  # noqa: F401
from flopscope._remote_array import RemoteArray  # noqa: F401
from flopscope._symmetric_info import SymmetryInfo  # noqa: F401
from flopscope.errors import (  # noqa: F401
    BudgetExhaustedError,
    FlopscopeError,
    FlopscopeServerError,
    FlopscopeWarning,
    NoBudgetContextError,
    SymmetryError,
)

# FlopscopeArray is the counted-array type used by the core. On the client
# side, RemoteArray plays the same role, so expose it under both names.
FlopscopeArray = RemoteArray

# ---------------------------------------------------------------------------
# Registry helpers (useful for introspection)
# ---------------------------------------------------------------------------
from flopscope._registry import (  # noqa: F401
    BLACKLISTED,
    FUNCTION_CATEGORIES,
    get_category,
    is_valid_op,
    iter_proxyable,
)

_LAZY_SUBMODULES = frozenset({"numpy", "accounting", "stats"})


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f"flopscope.{name}")
        globals()[name] = module
        return module
    raise AttributeError(
        f"flopscope does not provide {name!r}. "
        f"Numpy-shaped operations live under 'flopscope.numpy' "
        f"(try `import flopscope.numpy as fnp; fnp.{name}`)."
    )


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)
