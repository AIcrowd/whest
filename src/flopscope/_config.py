"""Global configuration for flopscope."""

from __future__ import annotations

_SETTINGS: dict[str, object] = {
    "symmetry_warnings": True,
    "use_inner_symmetry": True,
    "einsum_path_cache_size": 4096,
    "check_nan_inf": False,
}


def configure(**kwargs: object) -> None:
    """Update flopscope global settings.

    Parameters
    ----------
    symmetry_warnings : bool
        If ``False``, suppress :class:`~flopscope.errors.SymmetryLossWarning`
        warnings.  Default ``True``.
    use_inner_symmetry : bool
        If ``True``, exploit inner (W-side) symmetry to reduce FLOP costs
        when all W-group labels are contracted at the same step.
        Default ``True``.
    einsum_path_cache_size : int
        Maximum number of entries in the einsum path cache.
        Changing this rebuilds the cache (old entries are discarded).
        Default ``4096``.
    check_nan_inf : bool
        If ``True``, scan every counted op's output for NaN/Inf values and
        emit a :class:`~flopscope.errors.FlopscopeWarning` if any are found.
        The scan is two full O(n) sweeps over the result and is silently
        attributed to ``untracked_time``, so it is off by default for
        production scoring.  Opt in when debugging an estimator that
        produces NaN/Inf to identify the introducing op.  Default ``False``.

    Returns
    -------
    None
        Updates the in-process global configuration immediately.

    Examples
    --------
    >>> import flopscope as flops
    >>> flops.configure(einsum_path_cache_size=8192)
    >>> flops.configure(symmetry_warnings=False)
    >>> flops.configure(check_nan_inf=True)
    """
    for key, value in kwargs.items():
        if key not in _SETTINGS:
            raise ValueError(f"Unknown setting: {key!r}")
        _SETTINGS[key] = value

    if "einsum_path_cache_size" in kwargs:
        from flopscope._einsum import _rebuild_einsum_cache

        _rebuild_einsum_cache()


def get_setting(key: str) -> object:
    """Return the current value of a global setting."""
    return _SETTINGS[key]
