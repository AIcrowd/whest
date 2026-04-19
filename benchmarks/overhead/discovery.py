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


def _callable_alias_key(value: Any) -> Any:
    if inspect.ismethod(value):
        return (id(value.__self__), value.__func__)
    return value


def _iter_public_operations(module_name: str):
    module = importlib.import_module(module_name)
    for name in _public_callable_names(module):
        try:
            value = getattr(module, name)
        except AttributeError:
            continue
        if inspect.isroutine(value):
            yield {
                "module": module_name,
                "op_name": name,
                "qualified_name": f"{module_name}.{name}",
                "surface": _module_surface(module_name),
                "value": value,
                "alias_key": _callable_alias_key(value),
            }
            continue
        if module_name == "whest.stats" and all(
            hasattr(value, attr) for attr in ("pdf", "cdf", "ppf")
        ):
            for method_name in ("pdf", "cdf", "ppf"):
                method = getattr(value, method_name)
                if not callable(method):
                    continue
                yield {
                    "module": f"{module_name}.{name}",
                    "op_name": method_name,
                    "qualified_name": f"{module_name}.{name}.{method_name}",
                    "surface": "stats",
                    "value": method,
                    "alias_key": _callable_alias_key(method),
                }


def _load_exclusion_policies() -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(_EXCLUSION_PATH.read_text())
    return payload["excluded"], payload["unsupported_for_ratio"]


def _build_inventory() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for module_name in DISCOVERY_MODULES:
        for entry in _iter_public_operations(module_name):
            records.append(
                {
                    "op_name": entry["op_name"],
                    "surface": entry["surface"],
                    "module": entry["module"],
                    "qualified_name": entry["qualified_name"],
                    "aliases": [],
                    "alias_key": entry["alias_key"],
                }
            )

    aliases_by_key: dict[Any, list[str]] = {}
    for entry in records:
        aliases_by_key.setdefault(entry["alias_key"], []).append(entry["qualified_name"])

    for entry in records:
        entry["aliases"] = [
            qualified_name
            for qualified_name in sorted(aliases_by_key[entry["alias_key"]])
            if qualified_name != entry["qualified_name"]
        ]
        del entry["alias_key"]

    return sorted(records, key=lambda entry: (entry["surface"], entry["qualified_name"]))


def _public_export_name(case: Any) -> str:
    return f"whest.{case.op_name}"


def classify_public_operations() -> dict[str, list[dict[str, Any]]]:
    """Return a ledger of benchmarked, excluded, unsupported, and unclassified ops."""

    benchmark_cases = seed_cases()
    benchmarked_exports = {_public_export_name(case) for case in benchmark_cases}
    excluded_reasons, unsupported_reasons = _load_exclusion_policies()
    inventory = _build_inventory()

    benchmarked = [
        {
            "case_id": case.case_id,
            "op_name": case.op_name,
            "surface": case.surface,
            "qualified_name": _public_export_name(case),
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
            "qualified_name": f"whest.{op_name}",
            "reason": reason,
        }
        for op_name, reason in sorted(excluded_reasons.items())
    ]

    unsupported = [
        {
            "op_name": op_name,
            "surface": "api",
            "qualified_name": f"whest.{op_name}",
            "reason": reason,
        }
        for op_name, reason in sorted(unsupported_reasons.items())
    ]

    excluded_exports = {f"whest.{name}" for name in excluded_reasons}
    unsupported_exports = {f"whest.{name}" for name in unsupported_reasons}
    accounted_for = benchmarked_exports | excluded_exports | unsupported_exports

    inventory_by_status = []
    for entry in inventory:
        qualified_name = entry["qualified_name"]
        if qualified_name in benchmarked_exports:
            status = "benchmarked"
        elif qualified_name in excluded_exports:
            status = "excluded"
        elif qualified_name in unsupported_exports:
            status = "unsupported"
        else:
            status = "unclassified"
        inventory_by_status.append({**entry, "status": status})

    unclassified = [
        entry
        for entry in inventory_by_status
        if entry["qualified_name"] not in accounted_for
    ]

    return {
        "benchmarked": benchmarked,
        "excluded": excluded,
        "unsupported": unsupported,
        "unclassified": unclassified,
        "inventory": inventory_by_status,
    }
