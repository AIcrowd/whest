"""Configurable per-operation FLOP weights for mechestim.

Weights are multiplicative constants applied to analytical FLOP costs.
Without a weights file, all weights default to 1.0 (backward compatible).

Set MECHESTIM_WEIGHTS_FILE=/path/to/weights.json to load custom weights.
The JSON must have a "weights" key mapping op_name -> float multiplier.
"""

from __future__ import annotations

import json
import os

_ACTIVE_WEIGHTS: dict[str, float] = {}


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


def load_weights(path: str | None = None) -> None:
    """Load FLOP weights from a JSON file.

    Parameters
    ----------
    path : str or None
        Path to weights JSON file. If None, reads from the
        ``MECHESTIM_WEIGHTS_FILE`` environment variable. If neither
        is set, does nothing.

    Raises
    ------
    FileNotFoundError
        If an explicit path is given but does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    """
    if path is None:
        path = os.environ.get("MECHESTIM_WEIGHTS_FILE")
    if path is None:
        return
    with open(path) as f:
        data = json.load(f)
    _ACTIVE_WEIGHTS.update(data.get("weights", {}))


def reset_weights() -> None:
    """Clear all loaded weights. For testing."""
    _ACTIVE_WEIGHTS.clear()


# Auto-load from env var at import time
load_weights()
