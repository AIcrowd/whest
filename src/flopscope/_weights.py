"""Configurable per-operation FLOP weights for flopscope.

Flopscope supports official packaged weights, custom overrides via
``FLOPSCOPE_WEIGHTS_FILE``, and complete disabling via ``FLOPSCOPE_DISABLE_WEIGHTS=1``.
Packaged official weights are loaded automatically on import unless explicitly
overridden or disabled.

The JSON payload must have a ``"weights"`` key mapping
``op_name -> float multiplier``.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from importlib import resources

_ACTIVE_WEIGHTS: dict[str, float] = {}
_WARNED_MESSAGES: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    _WARNED_MESSAGES.add(message)


def _weights_disabled() -> bool:
    value = os.environ.get("FLOPSCOPE_DISABLE_WEIGHTS")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _extract_weights(data: dict, *, source: str) -> dict[str, float]:
    weights = data.get("weights")
    if not isinstance(weights, dict):
        raise ValueError(f"{source} is missing a valid 'weights' mapping")
    validated: dict[str, float] = {}
    for op_name, value in weights.items():
        if not isinstance(op_name, str):
            raise ValueError(f"{source} has a non-string operation name: {op_name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{source} weight for {op_name!r} must be a non-negative finite number"
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(
                f"{source} weight for {op_name!r} must be a non-negative finite number"
            )
        validated[op_name] = numeric_value
    return validated


def _read_weights_file(path: str, *, source: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _extract_weights(data, source=source)


def _load_packaged_weights() -> dict[str, float] | None:
    try:
        resource = resources.files("flopscope").joinpath("data/default_weights.json")
        with resource.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _extract_weights(data, source="Packaged official weights")
    except Exception as exc:  # pragma: no cover - defensive fallback path
        _warn_once(
            "Flopscope could not load packaged official weights "
            f"({exc}); falling back to unit weights (1.0 for all operations)."
        )
        return None


def get_weight(op_name: str) -> float:
    """Return the FLOP weight multiplier for an operation.

    Parameters
    ----------
    op_name : str
        Operation name as passed to ``BudgetContext.deduct()``,
        e.g. ``"exp"``, ``"linalg.cholesky"``, ``"fft.fft"``.

    Returns
    -------
    float
        Multiplicative weight. Defaults to 1.0 if not configured.
    """
    return _ACTIVE_WEIGHTS.get(op_name, 1.0)


def load_weights(path: str | None = None, *, use_packaged_default: bool = True) -> None:
    """Resolve and load active FLOP weights.

    Parameters
    ----------
    path : str or None
        Explicit path to a weights JSON file. If None, reads from the
        ``FLOPSCOPE_WEIGHTS_FILE`` environment variable when present.
    use_packaged_default : bool, optional
        Whether to fall back to the packaged official weights when no valid
        override is available.

    Notes
    -----
    Resolution order:

    1. If ``FLOPSCOPE_DISABLE_WEIGHTS=1``, all weights are disabled.
    2. If an explicit path or ``FLOPSCOPE_WEIGHTS_FILE`` is provided and valid,
       it is used.
    3. If the override is unusable, a warning is emitted and Flopscope falls back
       to the packaged official weights when enabled, otherwise to unit
       weights.
    4. If packaged weights are enabled but unavailable, a warning is emitted
       and Flopscope falls back to unit weights.
    """
    _ACTIVE_WEIGHTS.clear()

    if _weights_disabled():
        return

    override_path = (
        path if path is not None else os.environ.get("FLOPSCOPE_WEIGHTS_FILE")
    )

    if override_path is not None:
        try:
            _ACTIVE_WEIGHTS.update(
                _read_weights_file(
                    override_path,
                    source=f"Custom weights file '{override_path}'",
                )
            )
            return
        except Exception as exc:
            fallback_target = (
                "packaged official weights" if use_packaged_default else "unit weights"
            )
            _warn_once(
                f"Flopscope could not load custom weights from '{override_path}' "
                f"({exc}); falling back to {fallback_target}."
            )

    if not use_packaged_default:
        return

    packaged_weights = _load_packaged_weights()
    if packaged_weights is not None:
        _ACTIVE_WEIGHTS.update(packaged_weights)


def reset_weights() -> None:
    """Clear all loaded weights. For testing."""
    _ACTIVE_WEIGHTS.clear()
    _WARNED_MESSAGES.clear()


load_weights()
