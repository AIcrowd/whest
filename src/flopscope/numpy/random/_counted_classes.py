"""Counted subclasses of numpy.random.Generator and numpy.random.RandomState.

The classes use ``__getattribute__`` as a gate that consults registry-derived
``_COUNTED`` and ``_FREE`` sets. Sampler methods are added at module init by
``_build_counted_class``, which iterates the registry and emits wrappers via
``_make_counted_method``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as _np

from flopscope._ndarray import _asflopscope
from flopscope._registry import REGISTRY
from flopscope._validation import require_budget
from flopscope.errors import UnsupportedFunctionError
from flopscope.numpy.random._cost_formulas import COST_FORMULAS


def _make_counted_method(
    op_name: str, formula_name: str, base_cls: type, plain_factory: Callable[..., Any]
) -> Callable[..., Any]:
    """Build a counted-method wrapper for the parent class's named method.

    The wrapper calls the parent method on a *plain* base-class instance that
    shares the same RNG state (so internal sibling-method calls do not go
    through the counted gate and double-count).  After numpy's C path returns,
    cost is computed via the named formula, deducted from the active
    BudgetContext, and the result is wrapped as FlopscopeArray if it is an
    ndarray.
    """
    method_name = op_name.split(".")[-1]
    base_method = getattr(base_cls, method_name)
    formula = COST_FORMULAS[formula_name]

    @functools.wraps(base_method)
    def wrapped(self, *args: Any, **kwargs: Any) -> Any:
        budget = require_budget()
        # Dispatch through a plain base-class instance to avoid the counted
        # __getattribute__ gate intercepting any internal sibling calls
        # (e.g. Generator.choice → Generator.integers).
        plain = plain_factory(self)
        result = base_method(plain, *args, **kwargs)
        cost = formula(args, kwargs, result)
        with budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=()):
            pass  # numpy already executed; deduct is post-hoc for output-dependent cost
        if isinstance(result, _np.ndarray):
            return _asflopscope(result)
        return result

    return wrapped


def _build_counted_class(
    base_cls: type,
    op_prefix: str,
    target_cls: type,
    plain_factory: Callable[..., Any],
) -> None:
    """Populate target_cls._COUNTED, _FREE, and counted method bindings.

    Reads REGISTRY entries whose op_name starts with op_prefix. counted_random_method
    entries become wrapped class methods; free_random_method entries are added to
    _FREE so the __getattribute__ gate lets them through unwrapped.

    plain_factory is called with ``self`` (the counted subclass instance) to
    produce a plain base-class instance sharing the same RNG state, so that
    internal sibling calls inside numpy's C code do not go through the gate.
    """
    counted: set[str] = set()
    free: set[str] = set()
    for op_name, entry in REGISTRY.items():
        if not op_name.startswith(op_prefix):
            continue
        short = op_name[len(op_prefix):]
        category = entry.get("category")
        if category == "counted_random_method":
            counted.add(short)
            setattr(
                target_cls,
                short,
                _make_counted_method(
                    op_name, entry["cost_formula"], base_cls, plain_factory
                ),
            )
        elif category == "free_random_method":
            free.add(short)
    target_cls._COUNTED = frozenset(counted)
    target_cls._FREE = frozenset(free)


class _CountedGenerator(_np.random.Generator):
    """numpy Generator subclass with FLOP-counted sampler methods."""

    _COUNTED: ClassVar[frozenset[str]] = frozenset()
    _FREE: ClassVar[frozenset[str]] = frozenset()

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        cls = type(self)
        if name in cls._FREE or name in cls._COUNTED:
            return super().__getattribute__(name)
        raise UnsupportedFunctionError(
            f"flopscope does not count Generator.{name}. "
            f"This is either a new numpy method or one not yet wrapped. "
            f"See https://github.com/AIcrowd/flopscope/issues/18"
        )


class _CountedRandomState(_np.random.RandomState):
    """numpy RandomState subclass with FLOP-counted sampler methods."""

    _COUNTED: ClassVar[frozenset[str]] = frozenset()
    _FREE: ClassVar[frozenset[str]] = frozenset()

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        cls = type(self)
        if name in cls._FREE or name in cls._COUNTED:
            return super().__getattribute__(name)
        raise UnsupportedFunctionError(
            f"flopscope does not count RandomState.{name}. "
            f"This is either a new numpy method or one not yet wrapped. "
            f"See https://github.com/AIcrowd/flopscope/issues/18"
        )


# Wire counted methods at import time.
# plain_factory creates a plain base-class instance sharing the same RNG state;
# this prevents numpy's internal sibling-method calls from going through the
# counted __getattribute__ gate and double-counting FLOPs.
_build_counted_class(
    _np.random.Generator,
    "random.Generator.",
    _CountedGenerator,
    plain_factory=lambda self: _np.random.Generator(self._bit_generator),
)
_build_counted_class(
    _np.random.RandomState,
    "random.RandomState.",
    _CountedRandomState,
    plain_factory=lambda self: _np.random.RandomState(self._bit_generator),
)
