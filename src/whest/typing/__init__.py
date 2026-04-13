"""Type hints re-exported from numpy.typing.

Usage::

    from mechestim.typing import NDArray, ArrayLike

    def predict(x: NDArray) -> NDArray:
        ...

All names are plain aliases to numpy.typing. Because MechestimArray
is a subclass of numpy.ndarray, annotations like `NDArray[float32]`
accept mechestim arrays without any runtime overhead.
"""

from __future__ import annotations

# Re-export everything numpy.typing publicly exposes
from numpy import typing as _np_typing
from numpy.typing import (  # noqa: F401
    ArrayLike,
    DTypeLike,
    NBitBase,
    NDArray,
)

__all__ = [name for name in dir(_np_typing) if not name.startswith("_")]

for _name in __all__:
    if _name not in globals():
        globals()[_name] = getattr(_np_typing, _name)
