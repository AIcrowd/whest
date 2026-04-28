"""Flopscope: FLOP-counting numpy primitives for the Mechanistic Estimation Challenge.

The public API is structured JAX-style:

- ``flopscope`` (this module) exports flopscope-specific primitives: budget
  contexts, configuration, symmetric tensors, permutation groups, errors, and
  the :class:`FlopscopeArray` type.
- ``flopscope.numpy`` exports the full counted numpy-shaped surface
  (``einsum``, ``array``, ``linspace``, ``linalg``, ``fft``, ``random``, ...).
  Attributes not explicitly implemented there fall back to raw ``numpy``.
- ``flopscope.accounting`` exposes analytical cost helpers
  (``einsum_cost``, ``pointwise_cost``, ``reduction_cost``, ...).
- ``flopscope.stats`` hosts statistical-distribution primitives (not a
  numpy submodule; closer in spirit to scipy.stats).

Usage::

    import flopscope as flops
    import flopscope.numpy as fnp

    flops.configure(symmetry_warnings=False)
    with flops.BudgetContext(flop_budget=1_000_000) as budget:
        W = fnp.array(weight_matrix)
        h = fnp.einsum('ij,j->i', W, x)
        h = fnp.maximum(h, 0)
        print(budget.summary())
"""

import importlib as _importlib

import numpy as _np

from flopscope._registry import REGISTRY_META as _REGISTRY_META

__version__ = f"0.2.0+np{_np.__version__}"
__numpy_version__ = _np.__version__
__numpy_pinned__ = _REGISTRY_META["numpy_version"]
__numpy_supported__ = _REGISTRY_META.get("numpy_supported", ">=2.0.0,<2.5.0")

from flopscope._version_check import check_numpy_version as _check_numpy_version

_check_numpy_version(__numpy_supported__)

# --- Budget and diagnostics ---
from flopscope._budget import (  # noqa: F401,E402
    BudgetContext,
    OpRecord,
    budget,
    budget_reset,
    budget_summary_dict,
    namespace,
)
from flopscope._config import configure  # noqa: F401,E402
from flopscope._display import budget_live, budget_summary  # noqa: F401,E402

# --- Array type (flopscope-specific) ---
from flopscope._ndarray import FlopscopeArray  # noqa: F401,E402

# --- Path optimization types ---
from flopscope._opt_einsum import PathInfo, StepInfo  # noqa: F401,E402

# --- Permutation groups ---
from flopscope._perm_group import (  # noqa: F401,E402
    Cycle,
    Permutation,
    PermutationGroup,
)

# --- Symmetric tensor ---
from flopscope._symmetric import (  # noqa: F401,E402
    SymmetricTensor,
    SymmetryInfo,
    as_symmetric,
    is_symmetric,
    symmetrize,
)

# --- Errors ---
from flopscope.errors import (  # noqa: F401,E402
    BudgetExhaustedError,
    FlopscopeError,
    FlopscopeWarning,
    NoBudgetContextError,
    SymmetryError,
    SymmetryLossWarning,
    TimeExhaustedError,
    UnsupportedFunctionError,
)

_LAZY_SUBMODULES = frozenset({"numpy", "accounting", "stats"})

__all__ = [
    "BudgetContext",
    "BudgetExhaustedError",
    "Cycle",
    "FlopscopeArray",
    "FlopscopeError",
    "FlopscopeWarning",
    "NoBudgetContextError",
    "OpRecord",
    "PathInfo",
    "Permutation",
    "PermutationGroup",
    "StepInfo",
    "SymmetricTensor",
    "SymmetryError",
    "SymmetryInfo",
    "SymmetryLossWarning",
    "TimeExhaustedError",
    "UnsupportedFunctionError",
    "__numpy_pinned__",
    "__numpy_supported__",
    "__numpy_version__",
    "__version__",
    "accounting",
    "as_symmetric",
    "budget",
    "budget_live",
    "budget_reset",
    "budget_summary",
    "budget_summary_dict",
    "configure",
    "is_symmetric",
    "namespace",
    "numpy",
    "stats",
    "symmetrize",
]


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
