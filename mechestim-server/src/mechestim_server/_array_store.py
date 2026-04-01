"""ArrayStore — in-process dict-based mapping from handle IDs to numpy arrays."""

from __future__ import annotations

import numpy as np


class ArrayStore:
    """Simple store that maps string handle IDs to numpy arrays.

    Handle IDs are monotonically increasing strings of the form "a0", "a1",
    "a2", … The counter never resets, so IDs remain unique across put/free
    cycles.
    """

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def put(self, arr: np.ndarray) -> str:
        """Store *arr* and return its handle ID."""
        handle = f"a{self._counter}"
        self._arrays[handle] = arr
        self._counter += 1
        return handle

    def get(self, handle: str) -> np.ndarray:
        """Return the array for *handle*.

        Raises
        ------
        KeyError
            If *handle* is not in the store.
        """
        if handle not in self._arrays:
            raise KeyError(f"Array handle {handle!r} not found in store")
        return self._arrays[handle]

    def metadata(self, handle: str) -> dict:
        """Return metadata dict for *handle*.

        Returns
        -------
        dict
            ``{"id": handle, "shape": list[int], "dtype": str}``

        Raises
        ------
        KeyError
            If *handle* is not in the store.
        """
        arr = self.get(handle)  # propagates KeyError with helpful message
        return {
            "id": handle,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    def free(self, handles: list[str]) -> None:
        """Remove arrays by handle; silently ignore unknown handles."""
        for handle in handles:
            self._arrays.pop(handle, None)

    def clear(self) -> None:
        """Remove all arrays from the store."""
        self._arrays.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of arrays currently in the store."""
        return len(self._arrays)
