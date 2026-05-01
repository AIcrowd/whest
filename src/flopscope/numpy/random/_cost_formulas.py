"""Named cost-formula vocabulary for fnp.random method-level entries.

Each formula resolves to a callable ``(args, kwargs, result) -> int`` that
computes the FLOP cost from the call arguments and the numpy result.
The registry's ``cost_formula`` field names which formula a method uses.
"""

from __future__ import annotations

import builtins as _builtins
from collections.abc import Callable
from typing import Any

import numpy as _np

from flopscope._flops import sort_cost as _sort_cost


def _numel_output(args: tuple, kwargs: dict, result: Any) -> int:
    if isinstance(result, _np.ndarray):
        return _builtins.max(int(result.size), 1)
    return 1


def _numel_input(args: tuple, kwargs: dict, result: Any) -> int:
    a = args[0] if args else kwargs.get("x")
    if isinstance(a, _np.ndarray):
        return _builtins.max(int(a.size), 1)
    if hasattr(a, "__len__"):
        return _builtins.max(len(a), 1)
    return 1


def _length(args: tuple, kwargs: dict, result: Any) -> int:
    if args:
        n = int(args[0])
    elif "length" in kwargs:
        n = int(kwargs["length"])
    else:
        n = 1
    return _builtins.max(n, 1)


def _sort_cost_formula(args: tuple, kwargs: dict, result: Any) -> int:
    a = args[0] if args else kwargs.get("a")
    if isinstance(a, (int, _np.integer)):
        n = int(a)
    elif hasattr(a, "shape"):
        n = int(a.shape[0]) if getattr(a, "ndim", 1) > 0 else 1
    elif hasattr(a, "__len__"):
        n = len(a)
    else:
        n = 1
    return _sort_cost(n)


def _choice_cost(args: tuple, kwargs: dict, result: Any) -> int:
    # Generator.choice:    choice(a, size=None, replace=True, p=None, axis=0, shuffle=True)
    # RandomState.choice:  choice(a, size=None, replace=True, p=None)
    # `replace` is the 3rd positional or the `replace` kwarg.
    if len(args) >= 3:
        replace = bool(args[2])
    else:
        replace = bool(kwargs.get("replace", True))
    if replace:
        return _numel_output(args, kwargs, result)
    return _sort_cost_formula(args, kwargs, result)


COST_FORMULAS: dict[str, Callable[[tuple, dict, Any], int]] = {
    "numel(output)": _numel_output,
    "numel(input)": _numel_input,
    "length": _length,
    "sort_cost(n)": _sort_cost_formula,
    "choice_cost": _choice_cost,
}
