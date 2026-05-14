"""FlopscopePathInfo: wraps opt_einsum's PathInfo and adds an accumulation field.

opt_einsum's PathInfo may be a frozen dataclass; we hold a reference to it
and forward field accesses through __getattr__ rather than subclassing,
which is the safest cross-version approach.
"""

from __future__ import annotations

from typing import Any

from ._cost import AccumulationCost


class FlopscopePathInfo:
    """Mutable wrapper around opt_einsum's PathInfo with an accumulation field.

    Field forwarding: any attribute not on the wrapper itself is looked up on
    the wrapped inner PathInfo. The `optimized_cost` property prefers the
    accumulation total when attached.
    """

    __slots__ = ("_inner", "accumulation")

    def __init__(
        self,
        *,
        inner: Any,
        accumulation: AccumulationCost | None,
    ) -> None:
        self._inner = inner
        self.accumulation = accumulation

    @classmethod
    def from_inner(
        cls,
        *,
        inner: Any,
        accumulation: AccumulationCost | None,
    ) -> FlopscopePathInfo:
        return cls(inner=inner, accumulation=accumulation)

    @property
    def optimized_cost(self) -> int:
        """The charged FLOP cost. Returns accumulation.total when attached;
        otherwise falls back to the inner PathInfo's optimized_cost."""
        if self.accumulation is not None:
            return self.accumulation.total
        return getattr(self._inner, "optimized_cost", 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __str__(self) -> str:
        """Return the full formatted table.

        Overrides the inner upstream PathInfo's ``naive_cost`` and ``optimized_cost``
        to match ``self.optimized_cost`` while rendering, so the table's
        "(flopscope)" cost rows are consistent with the wrapper's attribute. The
        α/M cost is path-independent — naive and optimized are the same number,
        speedup is 1.000x by construction. Use ``self.accumulation.describe()``
        for the symmetric-vs-dense breakdown.
        """
        fmt = getattr(self._inner, "format_table", None)
        if fmt is None:
            return self.__repr__()

        flopscope_cost = self.optimized_cost
        original_naive = getattr(self._inner, "naive_cost", None)
        original_opt = getattr(self._inner, "optimized_cost", None)
        try:
            # Best-effort override — if the inner uses __slots__ or is frozen,
            # fall back to returning the un-overridden table.
            try:
                self._inner.naive_cost = flopscope_cost
                self._inner.optimized_cost = flopscope_cost
            except (AttributeError, TypeError):
                return fmt()
            return fmt()
        finally:
            if original_naive is not None:
                try:
                    self._inner.naive_cost = original_naive
                except (AttributeError, TypeError):
                    pass
            if original_opt is not None:
                try:
                    self._inner.optimized_cost = original_opt
                except (AttributeError, TypeError):
                    pass

    def __repr__(self) -> str:
        return (
            f"FlopscopePathInfo(optimized_cost={self.optimized_cost}, "
            f"path={getattr(self._inner, 'path', [])}, "
            f"accumulation={'<attached>' if self.accumulation else None})"
        )
