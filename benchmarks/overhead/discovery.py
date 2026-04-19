"""Discovery and accountability ledger for overhead benchmark operations."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from benchmarks.overhead.specs import seed_cases

DISCOVERY_MODULES = (
    "whest",
    "whest.linalg",
    "whest.fft",
    "whest.random",
    "whest.stats",
)

_EXCLUSION_PATH = Path(__file__).resolve().parents[1] / "overhead_exclusions.json"


def _module_surface(module_name: str) -> str:
    if module_name == "whest":
        return "api"
    return module_name.rsplit(".", 1)[-1]


def _public_callable_names(module: Any) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is None:
        names = [name for name in dir(module) if not name.startswith("_")]
    else:
        names = [name for name in exported if not name.startswith("_")]
    return names


def _iter_public_callables(module_name: str):
    module = importlib.import_module(module_name)
    for name in _public_callable_names(module):
        try:
            value = getattr(module, name)
        except AttributeError:
            continue
        if inspect.isroutine(value):
            yield name, value


def _load_exclusion_policies() -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(_EXCLUSION_PATH.read_text())
    return payload["excluded"], payload["unsupported_for_ratio"]


def _build_inventory() -> list[dict[str, Any]]:
    records_by_id: dict[int, dict[str, Any]] = {}

    for module_name in DISCOVERY_MODULES:
        for name, value in _iter_public_callables(module_name):
            record = records_by_id.setdefault(
                id(value),
                {
                    "module_surface": _module_surface(module_name),
                    "callable": value,
                    "names": set(),
                    "qualified_names": set(),
                    "modules": set(),
                },
            )
            record["names"].add(name)
            record["qualified_names"].add(f"{module_name}.{name}")
            record["modules"].add(module_name)

    inventory: list[dict[str, Any]] = []
    for record in records_by_id.values():
        names = sorted(record["names"])
        qualified_names = sorted(record["qualified_names"])
        inventory.append(
            {
                "op_name": names[0],
                "surface": record["module_surface"],
                "module": qualified_names[0].rsplit(".", 1)[0],
                "qualified_name": qualified_names[0],
                "qualified_names": qualified_names,
                "names": names,
                "aliases": names[1:],
                "modules": sorted(record["modules"]),
            }
        )

    return sorted(inventory, key=lambda entry: (entry["surface"], entry["op_name"]))


def classify_public_operations() -> dict[str, list[dict[str, Any]]]:
    """Return a ledger of benchmarked, excluded, unsupported, and unclassified ops."""

    benchmark_cases = seed_cases()
    benchmarked_names = {case.op_name for case in benchmark_cases}
    excluded_reasons, unsupported_reasons = _load_exclusion_policies()
    inventory = _build_inventory()

    benchmarked = [
        {
            "case_id": case.case_id,
            "op_name": case.op_name,
            "surface": case.surface,
            "family": case.family,
            "dtype": case.dtype,
            "size_name": case.size_name,
            "startup_mode": case.startup_mode,
            "source_file": case.source_file,
        }
        for case in benchmark_cases
    ]

    excluded = [
        {
            "op_name": op_name,
            "surface": "api",
            "reason": reason,
        }
        for op_name, reason in sorted(excluded_reasons.items())
    ]

    unsupported = [
        {
            "op_name": op_name,
            "surface": "api",
            "reason": reason,
        }
        for op_name, reason in sorted(unsupported_reasons.items())
    ]

    discovered_accounted_for = benchmarked_names | set(excluded_reasons) | set(
        unsupported_reasons
    )
    unclassified = [
        entry
        for entry in inventory
        if not set(entry["names"]) & discovered_accounted_for
    ]

    return {
        "benchmarked": benchmarked,
        "excluded": excluded,
        "unsupported": unsupported,
        "unclassified": unclassified,
    }
