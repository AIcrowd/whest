"""Guardrails for keeping public op exports in sync with the registry.

If a new public callable is added to the whest surface and it is meant to
behave like a supported operation, it must be added to ``whest._registry`` so
it shows up in generated docs artifacts such as ``website/public/ops.json``.

Truly non-registry helpers should be explicitly listed in the exemption set
below, with a reason kept in code review rather than silently drifting out of
sync.
"""

from __future__ import annotations

import inspect

import whest as we
from whest._registry import REGISTRY

# These are intentionally public helpers, not registry-backed supported ops.
TOP_LEVEL_NON_REGISTRY_CALLABLES = {
    "as_symmetric",
    "budget",
    "budget_live",
    "budget_reset",
    "budget_summary",
    "budget_summary_dict",
    "clear_einsum_cache",
    "configure",
    "einsum_cache_info",
    "get_printoptions",
    "is_symmetric",
    "printoptions",
    "set_printoptions",
}


def _public_callables(module):
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if inspect.ismodule(obj) or inspect.isclass(obj) or not callable(obj):
            continue
        yield name


def _public_module_defined_callables(module):
    for name, obj in module.__dict__.items():
        if name.startswith("_"):
            continue
        if inspect.ismodule(obj) or inspect.isclass(obj) or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        yield name


def test_top_level_public_callables_are_registry_backed_or_explicitly_exempt():
    missing = sorted(
        name
        for name in _public_callables(we)
        if name not in REGISTRY and name not in TOP_LEVEL_NON_REGISTRY_CALLABLES
    )
    assert not missing, (
        "Top-level public callables missing from whest._registry: "
        f"{missing}. Add them to the registry so docs/api can see them, or "
        "explicitly exempt them here if they are intentionally non-registry "
        "helpers."
    )


def test_random_module_defined_public_callables_are_registry_backed():
    missing = sorted(
        f"random.{name}"
        for name in _public_module_defined_callables(we.random)
        if f"random.{name}" not in REGISTRY
    )
    assert not missing, (
        "whest.random public callables missing from whest._registry: "
        f"{missing}. Add them to the registry so docs/api can see them."
    )


def test_linalg_public_exports_are_registry_backed():
    missing = sorted(
        f"linalg.{name}"
        for name in we.linalg.__all__
        if f"linalg.{name}" not in REGISTRY
    )
    assert not missing, (
        "whest.linalg public exports missing from whest._registry: "
        f"{missing}. Add them to the registry so docs/api can see them."
    )


def test_fft_public_exports_are_registry_backed():
    missing = sorted(
        f"fft.{name}" for name in we.fft.__all__ if f"fft.{name}" not in REGISTRY
    )
    assert not missing, (
        "whest.fft public exports missing from whest._registry: "
        f"{missing}. Add them to the registry so docs/api can see them."
    )
