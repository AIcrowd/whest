"""Global configuration for whest."""

from __future__ import annotations

_SETTINGS: dict[str, object] = {
    "symmetry_warnings": True,
    "use_inner_symmetry": True,
    "einsum_path_cache_size": 4096,
}


def configure(**kwargs: object) -> None:
    """Update whest global settings.

    Parameters
    ----------
    symmetry_warnings : bool
        If ``False``, suppress :class:`~whest.errors.SymmetryLossWarning`
        warnings.  Default ``True``.
    use_inner_symmetry : bool
        If ``True``, exploit inner (W-side) symmetry to reduce FLOP costs
        when all W-group labels are contracted at the same step.
        Default ``True``.
    einsum_path_cache_size : int
        Maximum number of entries in the einsum path cache.
        Changing this rebuilds the cache (old entries are discarded).
        Default ``4096``.
    """
    for key, value in kwargs.items():
        if key not in _SETTINGS:
            raise ValueError(f"Unknown setting: {key!r}")
        _SETTINGS[key] = value

    if "einsum_path_cache_size" in kwargs:
        from whest._einsum import _rebuild_einsum_cache

        _rebuild_einsum_cache()


def get_setting(key: str) -> object:
    """Return the current value of a global setting."""
    return _SETTINGS[key]
