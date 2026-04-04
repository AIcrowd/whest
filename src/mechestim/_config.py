"""Global configuration for mechestim."""

from __future__ import annotations

_SETTINGS: dict[str, object] = {
    "symmetry_warnings": True,
}


def configure(**kwargs: object) -> None:
    """Update mechestim global settings.

    Parameters
    ----------
    symmetry_warnings : bool
        If ``False``, suppress :class:`~mechestim.errors.SymmetryLossWarning`
        warnings.  Default ``True``.
    """
    for key, value in kwargs.items():
        if key not in _SETTINGS:
            raise ValueError(f"Unknown setting: {key!r}")
        _SETTINGS[key] = value


def get_setting(key: str) -> object:
    """Return the current value of a global setting."""
    return _SETTINGS[key]
