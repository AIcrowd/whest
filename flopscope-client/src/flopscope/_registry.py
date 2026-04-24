"""Registry helpers for the flopscope client API surface.

Thin wrapper around :mod:`flopscope._registry_data` that provides category
constants and lookup functions.
"""

from __future__ import annotations

from flopscope._registry_data import FUNCTION_CATEGORIES

# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

COUNTED_UNARY = "counted_unary"
COUNTED_BINARY = "counted_binary"
COUNTED_REDUCTION = "counted_reduction"
COUNTED_CUSTOM = "counted_custom"
FREE = "free"
BLACKLISTED = "blacklisted"

_PROXY_CATEGORIES = frozenset(
    {
        COUNTED_UNARY,
        COUNTED_BINARY,
        COUNTED_REDUCTION,
        COUNTED_CUSTOM,
        FREE,
    }
)

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def is_valid_op(name: str) -> bool:
    """Return ``True`` if *name* is a known function in the registry."""
    return name in FUNCTION_CATEGORIES


def get_category(name: str) -> str | None:
    """Return the category string for *name*, or ``None`` if unknown."""
    return FUNCTION_CATEGORIES.get(name)


def is_blacklisted(name: str) -> bool:
    """Return ``True`` if *name* is explicitly blacklisted."""
    return FUNCTION_CATEGORIES.get(name) == BLACKLISTED


def iter_proxyable(*, prefix: str = "") -> list[str]:
    """Return sorted list of op names that should get auto-generated proxies.

    Parameters
    ----------
    prefix:
        If non-empty, only return names starting with this prefix
        (e.g. ``"random."``).  The prefix is **not** stripped from the
        returned names.
    """
    return sorted(
        name
        for name, cat in FUNCTION_CATEGORIES.items()
        if cat in _PROXY_CATEGORIES and (not prefix or name.startswith(prefix))
    )
