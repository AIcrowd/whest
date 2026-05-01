"""Counted subclasses of numpy.random.Generator and numpy.random.RandomState.

The classes use ``__getattribute__`` as a gate that consults registry-derived
``_COUNTED`` and ``_FREE`` sets. Any attribute not in those sets raises
``UnsupportedFunctionError``. Sampler methods are added to the class at module
init time by ``_build_counted_class``; see Task 5/7 for the factory.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as _np

from flopscope.errors import UnsupportedFunctionError


class _CountedGenerator(_np.random.Generator):
    """numpy Generator subclass with FLOP-counted sampler methods.

    Methods listed in ``_COUNTED`` are wrapped to deduct FLOPs and return
    ``FlopscopeArray``. Methods in ``_FREE`` pass through. Anything else
    raises ``UnsupportedFunctionError`` — the structural guarantee that
    new numpy methods cannot silently bypass FLOP accounting.
    """

    _COUNTED: ClassVar[frozenset[str]] = frozenset()
    # bit_generator is a structural property accessed by Generator.__repr__
    _FREE: ClassVar[frozenset[str]] = frozenset({"bit_generator"})

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
    """numpy RandomState subclass with FLOP-counted sampler methods.

    Same gate pattern as ``_CountedGenerator``; method set differs (legacy names).
    """

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
