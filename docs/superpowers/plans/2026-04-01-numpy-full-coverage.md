# Exhaustive NumPy Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make mechestim exhaustively cover every public numpy 2.1.x function — each one explicitly categorized, wrapped, tested, and tracked.

**Architecture:** A central registry (`_registry.py`) maps every numpy 2.1.x public callable to a category (counted/free/blacklisted). An audit script introspects numpy at runtime and compares against the registry to ensure zero gaps. Existing factory wrappers are extended to cover new functions. CI enforces completeness.

**Tech Stack:** Python 3.10+, NumPy 2.1.x, rich (dev dependency), pytest

---

### Task 1: Pin NumPy 2.1.x and Update Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml to pin numpy and add rich**

```toml
[project]
name = "mechestim"
version = "0.2.0"
description = "NumPy-compatible math primitives with FLOP counting for the Mechanistic Estimation Challenge"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "numpy>=2.1.0,<2.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "rich>=13.0",
]
docs = [
    "mkdocs-material",
    "mkdocstrings[python]",
]
```

- [ ] **Step 2: Install updated dependencies**

Run: `uv sync --reinstall`
Expected: numpy 2.1.x installed, rich installed

- [ ] **Step 3: Verify numpy version**

Run: `python -c "import numpy; print(numpy.__version__)"`
Expected: `2.1.x` (some 2.1 version)

- [ ] **Step 4: Run existing tests to check for numpy 2.x breakage**

Run: `pytest tests/ -v`
Expected: Some tests may fail due to numpy 2.x API changes. Note failures for later fixing.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: pin numpy 2.1.x, add rich dev dependency"
```

---

### Task 2: Build the NumPy Audit Introspection Engine

**Files:**
- Create: `scripts/numpy_audit.py`
- Test: `tests/test_audit.py`

This task builds only the introspection logic — the part that walks numpy and discovers all public callables. No registry comparison yet.

- [ ] **Step 1: Write the test for numpy introspection**

```python
# tests/test_audit.py
"""Tests for the numpy audit script."""
import subprocess
import sys
import json


def test_audit_introspection_discovers_numpy_functions():
    """The audit introspector should find well-known numpy functions."""
    # Import the introspection function directly
    sys.path.insert(0, "scripts")
    from numpy_audit import introspect_numpy

    discovered = introspect_numpy()

    # Should discover well-known top-level functions
    assert "exp" in discovered
    assert "sin" in discovered
    assert "reshape" in discovered
    assert "zeros" in discovered
    assert "concatenate" in discovered

    # Should discover linalg functions
    assert "linalg.svd" in discovered
    assert "linalg.solve" in discovered
    assert "linalg.norm" in discovered

    # Should discover fft functions
    assert "fft.fft" in discovered
    assert "fft.ifft" in discovered

    # Should discover random functions
    assert "random.rand" in discovered
    assert "random.normal" in discovered

    # Should NOT discover private names
    for name in discovered:
        parts = name.split(".")
        assert not any(p.startswith("_") for p in parts), f"Private name found: {name}"

    # Should NOT discover names from excluded modules
    for name in discovered:
        assert not name.startswith("testing."), f"testing submodule found: {name}"
        assert not name.startswith("lib."), f"lib submodule found: {name}"


def test_audit_introspection_returns_metadata():
    """Each discovered function should have module and callable type info."""
    sys.path.insert(0, "scripts")
    from numpy_audit import introspect_numpy

    discovered = introspect_numpy()

    entry = discovered["exp"]
    assert entry["module"] == "numpy"
    assert entry["kind"] in ("function", "ufunc", "builtin", "class")

    entry_linalg = discovered["linalg.svd"]
    assert entry_linalg["module"] == "numpy.linalg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError` since `scripts/numpy_audit.py` doesn't exist yet.

- [ ] **Step 3: Create the audit script with introspection logic**

```python
#!/usr/bin/env python3
"""NumPy audit tool for mechestim.

Introspects numpy to discover all public callables, compares against the
mechestim registry, and produces a coverage report.

Usage:
    python scripts/numpy_audit.py              # rich terminal report
    python scripts/numpy_audit.py --json       # machine-readable JSON
    python scripts/numpy_audit.py --ci         # exit non-zero if gaps exist
    python scripts/numpy_audit.py --filter unclassified
    python scripts/numpy_audit.py --module numpy.linalg
"""
from __future__ import annotations

import argparse
import inspect
import json as json_module
import sys
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCANNED_MODULES = {
    "numpy": np,
    "numpy.linalg": np.linalg,
    "numpy.fft": np.fft,
    "numpy.random": np.random,
}

EXCLUDED_MODULES = [
    "numpy.testing",
    "numpy.lib",
    "numpy.dtypes",
    "numpy.ma",
    "numpy.polynomial",
    "numpy.strings",
    "numpy.char",
    "numpy.rec",
    "numpy.ctypeslib",
]

# Names that are technically callable but are types/exceptions/classes
# we don't want to wrap as math operations.
SKIP_NAMES = {
    # Exception/warning classes
    "AxisError", "ComplexWarning", "ModuleDeprecationWarning",
    "RankWarning", "TooHardError", "VisibleDeprecationWarning",
    # Utility classes
    "DataSource", "Tester",
    # Deprecated or internal-utility callables
    "add_docstring", "add_newdoc", "add_newdoc_ufunc",
    "lookfor", "who", "source", "info",
    # Type/dtype classes (not math operations)
    "bool_", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "float128",
    "complex64", "complex128", "complex256",
    "str_", "bytes_", "void", "object_",
    "intp", "uintp", "intc", "uintc",
    "short", "ushort", "longlong", "ulonglong",
    "half", "single", "double", "longdouble",
    "csingle", "cdouble", "clongdouble",
    "character", "integer", "signedinteger", "unsignedinteger",
    "inexact", "floating", "complexfloating", "number",
    "flexible", "generic",
    "byte", "ubyte", "longfloat", "clongfloat",
    "singlecomplex", "complex_", "float_", "int_",
    "unicode_", "string_",
    # ndarray itself and core types
    "ndarray", "nditer", "ndenumerate", "ndindex",
    "array", "asarray", "ascontiguousarray", "asfortranarray",
    "chararray", "matrix", "memmap", "recarray", "record",
    "dtype", "finfo", "iinfo", "errstate", "seterr", "geterr",
    "seterrcall", "geterrcall",
    "broadcast", "busdaycalendar", "datetime64", "timedelta64",
    "vectorize", "poly1d", "polynomial",
    # Printing config
    "set_printoptions", "get_printoptions", "printoptions",
    "set_string_function",
    "format_float_positional", "format_float_scientific",
    # Random module classes
    "BitGenerator", "Generator", "MT19937", "PCG64", "PCG64DXSM",
    "Philox", "SFC64", "SeedSequence", "RandomState",
}


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def _classify_callable(obj: Any) -> str:
    """Classify a callable into a kind string."""
    if isinstance(obj, np.ufunc):
        return "ufunc"
    if inspect.isbuiltin(obj):
        return "builtin"
    if inspect.isfunction(obj):
        return "function"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "callable"
    return "unknown"


def introspect_numpy() -> dict[str, dict[str, str]]:
    """Walk numpy and return all public callables.

    Returns a dict mapping qualified names (e.g. "exp", "linalg.svd")
    to metadata: {"module": "numpy", "kind": "ufunc"}.
    """
    discovered: dict[str, dict[str, str]] = {}

    for module_name, module_obj in SCANNED_MODULES.items():
        prefix = "" if module_name == "numpy" else module_name.replace("numpy.", "") + "."

        for name in sorted(dir(module_obj)):
            # Skip private names
            if name.startswith("_"):
                continue

            obj = getattr(module_obj, name, None)
            if obj is None:
                continue

            # Skip non-callables
            if not callable(obj):
                continue

            # Skip sub-modules (they're scanned separately or excluded)
            if inspect.ismodule(obj):
                continue

            qualified_name = f"{prefix}{name}"

            # Skip names in the skip list
            if name in SKIP_NAMES:
                continue

            # For submodules, also skip if the name appears in the top-level
            # (it's the same function, we track it under "numpy" module)
            if prefix and hasattr(np, name) and getattr(np, name) is obj:
                continue

            discovered[qualified_name] = {
                "module": module_name,
                "kind": _classify_callable(obj),
            }

    return discovered


# ---------------------------------------------------------------------------
# Registry comparison
# ---------------------------------------------------------------------------

def load_registry() -> tuple[dict, dict]:
    """Load the mechestim registry. Returns (REGISTRY_META, REGISTRY)."""
    try:
        from mechestim._registry import REGISTRY, REGISTRY_META
        return REGISTRY_META, REGISTRY
    except ImportError:
        return {}, {}


def compare(discovered: dict, registry: dict) -> dict[str, list[dict]]:
    """Compare discovered numpy callables against the registry.

    Returns a dict with keys: covered, registered_not_implemented,
    unclassified, blacklisted, stale.
    """
    # Determine which registered functions are actually implemented
    # by checking if they're importable from mechestim
    import mechestim as me
    implemented_names = set()
    for name in registry:
        parts = name.split(".")
        try:
            obj = me
            for part in parts:
                obj = getattr(obj, part)
            implemented_names.add(name)
        except (AttributeError, TypeError):
            pass

    result: dict[str, list[dict]] = {
        "covered": [],
        "registered_not_implemented": [],
        "unclassified": [],
        "blacklisted": [],
        "stale": [],
    }

    # Check each discovered numpy function against registry
    for name, info in sorted(discovered.items()):
        if name in registry:
            entry = registry[name]
            if entry["category"] == "blacklisted":
                result["blacklisted"].append({"name": name, **info, **entry})
            elif name in implemented_names:
                result["covered"].append({"name": name, **info, **entry})
            else:
                result["registered_not_implemented"].append({"name": name, **info, **entry})
        else:
            result["unclassified"].append({"name": name, **info})

    # Check for stale registry entries (in registry but not in numpy)
    for name, entry in sorted(registry.items()):
        if name not in discovered:
            result["stale"].append({"name": name, **entry})

    return result


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------

def print_rich_report(comparison: dict, registry_meta: dict, filter_category: str | None = None, filter_module: str | None = None) -> None:
    """Print a rich terminal table of the coverage report."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print("ERROR: rich is not installed. Install with: pip install rich", file=sys.stderr)
        print("Falling back to plain text output.", file=sys.stderr)
        print_plain_report(comparison, registry_meta, filter_category, filter_module)
        return

    console = Console()

    CATEGORY_COLORS = {
        "covered": "green",
        "registered_not_implemented": "yellow",
        "unclassified": "red",
        "blacklisted": "dim",
        "stale": "magenta",
    }

    CATEGORY_LABELS = {
        "covered": "Covered",
        "registered_not_implemented": "Registered (not impl)",
        "unclassified": "UNCLASSIFIED",
        "blacklisted": "Blacklisted",
        "stale": "Stale",
    }

    np_version = registry_meta.get("numpy_version", "unknown")
    console.print(f"\n[bold]mechestim NumPy Audit Report[/bold] (numpy {np_version})\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Function", style="cyan", no_wrap=True)
    table.add_column("Module", style="blue")
    table.add_column("Category", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Notes", max_width=50)

    categories_to_show = [filter_category] if filter_category else list(CATEGORY_COLORS.keys())

    # Group by module
    all_entries = []
    for cat in categories_to_show:
        for entry in comparison.get(cat, []):
            module = entry.get("module", "unknown")
            if filter_module and module != filter_module:
                continue
            all_entries.append((module, cat, entry))

    all_entries.sort(key=lambda x: (x[0], x[1], x[2]["name"]))

    current_module = None
    for module, cat, entry in all_entries:
        if module != current_module:
            if current_module is not None:
                table.add_section()
            current_module = module

        color = CATEGORY_COLORS[cat]
        status_text = Text(CATEGORY_LABELS[cat], style=color)
        category_text = Text(entry.get("category", cat), style=color)
        notes = entry.get("notes", "")
        if len(notes) > 50:
            notes = notes[:47] + "..."

        table.add_row(
            entry["name"],
            module,
            category_text,
            status_text,
            notes,
        )

    console.print(table)

    # Summary footer
    total_discovered = sum(len(comparison[k]) for k in ["covered", "registered_not_implemented", "unclassified", "blacklisted"])
    console.print(f"\n[bold]Coverage:[/bold] "
                  f"[green]{len(comparison['covered'])}[/green] implemented | "
                  f"[yellow]{len(comparison['registered_not_implemented'])}[/yellow] registered | "
                  f"[red]{len(comparison['unclassified'])}[/red] unclassified | "
                  f"[dim]{len(comparison['blacklisted'])}[/dim] blacklisted | "
                  f"[magenta]{len(comparison['stale'])}[/magenta] stale | "
                  f"numpy {np_version}\n")


def print_plain_report(comparison: dict, registry_meta: dict, filter_category: str | None = None, filter_module: str | None = None) -> None:
    """Print a plain text report for CI mode."""
    np_version = registry_meta.get("numpy_version", "unknown")
    print(f"mechestim NumPy Audit Report (numpy {np_version})")
    print("=" * 60)

    for cat in ["covered", "registered_not_implemented", "unclassified", "blacklisted", "stale"]:
        if filter_category and cat != filter_category:
            continue
        entries = comparison.get(cat, [])
        if filter_module:
            entries = [e for e in entries if e.get("module") == filter_module]
        if not entries:
            continue
        print(f"\n{cat.upper()} ({len(entries)}):")
        for entry in entries:
            notes = entry.get("notes", "")
            print(f"  {entry['name']:<40} {entry.get('module', ''):<16} {notes[:40]}")

    total_discovered = sum(len(comparison[k]) for k in ["covered", "registered_not_implemented", "unclassified", "blacklisted"])
    print(f"\nCoverage: {len(comparison['covered'])}/{total_discovered} implemented | "
          f"{len(comparison['unclassified'])} unclassified | "
          f"{len(comparison['blacklisted'])} blacklisted | numpy {np_version}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Audit mechestim numpy coverage")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of rich table")
    parser.add_argument("--ci", action="store_true", help="CI mode: plain text, exit non-zero if unclassified entries exist")
    parser.add_argument("--filter", type=str, default=None, help="Filter to a specific category")
    parser.add_argument("--module", type=str, default=None, help="Filter to a specific module (e.g. numpy.linalg)")
    args = parser.parse_args()

    discovered = introspect_numpy()
    registry_meta, registry = load_registry()
    comparison = compare(discovered, registry)

    if args.json:
        # Make it JSON serializable
        print(json_module.dumps(comparison, indent=2, default=str))
        return 0

    if args.ci:
        print_plain_report(comparison, registry_meta, args.filter, args.module)
        n_unclassified = len(comparison["unclassified"])
        if n_unclassified > 0:
            print(f"\nFAIL: {n_unclassified} unclassified numpy functions", file=sys.stderr)
            return 1
        print("\nPASS: All numpy functions are classified", file=sys.stderr)
        return 0

    print_rich_report(comparison, registry_meta, args.filter, args.module)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_audit.py -v`
Expected: PASS

- [ ] **Step 5: Run the audit script standalone to see discovered functions**

Run: `python scripts/numpy_audit.py --json | python -c "import sys,json; d=json.load(sys.stdin); print(f'Total unclassified: {len(d[\"unclassified\"])}')"`
Expected: Shows count of all discovered functions (all unclassified since no registry exists yet)

- [ ] **Step 6: Commit**

```bash
git add scripts/numpy_audit.py tests/test_audit.py
git commit -m "feat: add numpy audit introspection engine"
```

---

### Task 3: Build the Registry with Existing Functions Pre-Classified

**Files:**
- Create: `src/mechestim/_registry.py`
- Test: `tests/test_registry.py`

This task creates the registry file with all currently-wrapped functions correctly categorized, and all remaining numpy functions as `unclassified`. We'll use the audit script to discover what exists and build the initial registry.

- [ ] **Step 1: Write registry tests**

```python
# tests/test_registry.py
"""Tests for the mechestim numpy function registry."""
import numpy as np
from mechestim._registry import REGISTRY, REGISTRY_META


def test_registry_meta_has_numpy_version():
    assert "numpy_version" in REGISTRY_META
    # Should be a valid version string
    parts = REGISTRY_META["numpy_version"].split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts)


def test_registry_meta_has_last_updated():
    assert "last_updated" in REGISTRY_META


def test_all_entries_have_required_fields():
    for name, entry in REGISTRY.items():
        assert "category" in entry, f"{name} missing 'category'"
        assert "module" in entry, f"{name} missing 'module'"
        assert "notes" in entry, f"{name} missing 'notes'"
        assert entry["category"] in (
            "counted_unary", "counted_binary", "counted_reduction",
            "counted_custom", "free", "blacklisted", "unclassified",
        ), f"{name} has invalid category: {entry['category']}"


def test_existing_counted_unary_ops_in_registry():
    """All currently-wrapped unary ops should be registered as counted_unary."""
    expected = ["exp", "log", "log2", "log10", "abs", "negative", "sqrt",
                "square", "sin", "cos", "tanh", "sign", "ceil", "floor"]
    for name in expected:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "counted_unary", f"{name} is {REGISTRY[name]['category']}"


def test_existing_counted_binary_ops_in_registry():
    expected = ["add", "subtract", "multiply", "divide", "maximum", "minimum", "power", "mod"]
    for name in expected:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "counted_binary", f"{name} is {REGISTRY[name]['category']}"


def test_existing_counted_reduction_ops_in_registry():
    expected = ["sum", "max", "min", "prod", "mean", "std", "var",
                "argmax", "argmin", "cumsum", "cumprod"]
    for name in expected:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "counted_reduction", f"{name} is {REGISTRY[name]['category']}"


def test_existing_free_ops_in_registry():
    expected = ["zeros", "ones", "reshape", "transpose", "concatenate",
                "stack", "eye", "diag", "arange", "linspace", "where",
                "sort", "argsort", "unique", "pad", "triu", "tril",
                "allclose", "isnan", "isinf", "isfinite"]
    for name in expected:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "free", f"{name} is {REGISTRY[name]['category']}"


def test_existing_custom_ops_in_registry():
    expected = ["dot", "matmul", "einsum", "clip"]
    for name in expected:
        assert name in REGISTRY, f"{name} not in registry"
        assert REGISTRY[name]["category"] == "counted_custom", f"{name} is {REGISTRY[name]['category']}"


def test_linalg_svd_in_registry():
    assert "linalg.svd" in REGISTRY
    assert REGISTRY["linalg.svd"]["category"] == "counted_custom"
    assert REGISTRY["linalg.svd"]["module"] == "numpy.linalg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — `ImportError` since `_registry.py` doesn't exist

- [ ] **Step 3: Generate the initial registry using the audit script**

Run the audit script to dump all numpy functions as JSON, then use that to build the registry:

```bash
python -c "
import sys
sys.path.insert(0, 'scripts')
from numpy_audit import introspect_numpy
import json
discovered = introspect_numpy()
print(json.dumps(discovered, indent=2))
" > /tmp/numpy_discovered.json
```

Then create `src/mechestim/_registry.py` by hand, populating it with:
- All currently-wrapped functions in their correct categories
- All remaining discovered functions as `unclassified`

The file structure:

```python
"""NumPy function registry for mechestim.

Every public callable in the pinned numpy version has an entry here.
Categories:
  - counted_unary: unary math ops, cost = numel(output)
  - counted_binary: binary math ops, cost = numel(output) after broadcast
  - counted_reduction: reductions, cost = numel(input) * cost_multiplier
  - counted_custom: ops with bespoke cost formulas (einsum, dot, linalg ops)
  - free: zero FLOP cost (reshaping, indexing, type casting, creation)
  - blacklisted: intentionally unsupported
  - unclassified: not yet triaged (temporary — blocked from main by CI)
"""
from __future__ import annotations

import numpy as _np

REGISTRY_META = {
    "numpy_version": _np.__version__,
    "last_updated": "2026-04-01",
}

REGISTRY: dict[str, dict] = {
    # ===================================================================
    # COUNTED UNARY — cost = numel(output) FLOPs
    # ===================================================================
    "exp": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Pointwise exponential. Cost: numel(output) FLOPs.",
    },
    "log": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Natural logarithm. Cost: numel(output) FLOPs.",
    },
    "log2": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Base-2 logarithm. Cost: numel(output) FLOPs.",
    },
    "log10": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Base-10 logarithm. Cost: numel(output) FLOPs.",
    },
    "abs": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Absolute value. Cost: numel(output) FLOPs.",
    },
    "negative": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Numerical negative. Cost: numel(output) FLOPs.",
    },
    "sqrt": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Square root. Cost: numel(output) FLOPs.",
    },
    "square": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Element-wise square. Cost: numel(output) FLOPs.",
    },
    "sin": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Trigonometric sine. Cost: numel(output) FLOPs.",
    },
    "cos": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Trigonometric cosine. Cost: numel(output) FLOPs.",
    },
    "tanh": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Hyperbolic tangent. Cost: numel(output) FLOPs.",
    },
    "sign": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Sign function. Cost: numel(output) FLOPs.",
    },
    "ceil": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Ceiling. Cost: numel(output) FLOPs.",
    },
    "floor": {
        "category": "counted_unary",
        "module": "numpy",
        "notes": "Floor. Cost: numel(output) FLOPs.",
    },

    # ===================================================================
    # COUNTED BINARY — cost = numel(output) FLOPs after broadcast
    # ===================================================================
    "add": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise addition with broadcasting. Cost: numel(output) FLOPs.",
    },
    "subtract": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise subtraction. Cost: numel(output) FLOPs.",
    },
    "multiply": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise multiplication. Cost: numel(output) FLOPs.",
    },
    "divide": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise division. Cost: numel(output) FLOPs.",
    },
    "maximum": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise maximum. Cost: numel(output) FLOPs.",
    },
    "minimum": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise minimum. Cost: numel(output) FLOPs.",
    },
    "power": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise power. Cost: numel(output) FLOPs.",
    },
    "mod": {
        "category": "counted_binary",
        "module": "numpy",
        "notes": "Element-wise modulus. Cost: numel(output) FLOPs.",
    },

    # ===================================================================
    # COUNTED REDUCTION — cost = numel(input) * cost_multiplier
    # ===================================================================
    "sum": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Sum reduction. Cost: numel(input) FLOPs.",
    },
    "max": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Max reduction. Cost: numel(input) FLOPs.",
    },
    "min": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Min reduction. Cost: numel(input) FLOPs.",
    },
    "prod": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Product reduction. Cost: numel(input) FLOPs.",
    },
    "mean": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Mean reduction. Cost: numel(input) + numel(output) FLOPs. Extra output cost for division.",
    },
    "std": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 2,
        "notes": "Standard deviation. Cost: 2*numel(input) + numel(output). Two passes: variance then sqrt.",
    },
    "var": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 2,
        "notes": "Variance. Cost: 2*numel(input) + numel(output). Two passes: mean then squared diff.",
    },
    "argmax": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Index of maximum. Cost: numel(input) FLOPs.",
    },
    "argmin": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Index of minimum. Cost: numel(input) FLOPs.",
    },
    "cumsum": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Cumulative sum. Cost: numel(input) FLOPs.",
    },
    "cumprod": {
        "category": "counted_reduction",
        "module": "numpy",
        "cost_multiplier": 1,
        "notes": "Cumulative product. Cost: numel(input) FLOPs.",
    },

    # ===================================================================
    # COUNTED CUSTOM — bespoke cost formulas
    # ===================================================================
    "dot": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Dot product / matrix multiply. Cost depends on operand dimensions.",
    },
    "matmul": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Matrix multiplication. Cost depends on operand dimensions.",
    },
    "einsum": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Einstein summation. Cost = product of all index dimensions.",
    },
    "clip": {
        "category": "counted_custom",
        "module": "numpy",
        "notes": "Clip to range. Cost: numel(input) FLOPs.",
    },
    "linalg.svd": {
        "category": "counted_custom",
        "module": "numpy.linalg",
        "notes": "Truncated SVD. Cost: m * n * k FLOPs.",
    },

    # ===================================================================
    # FREE — zero FLOP cost
    # ===================================================================
    "array": {
        "category": "free",
        "module": "numpy",
        "notes": "Create array from data. No computation.",
    },
    "zeros": {
        "category": "free",
        "module": "numpy",
        "notes": "Array of zeros. No computation.",
    },
    "ones": {
        "category": "free",
        "module": "numpy",
        "notes": "Array of ones. No computation.",
    },
    "full": {
        "category": "free",
        "module": "numpy",
        "notes": "Array filled with value. No computation.",
    },
    "eye": {
        "category": "free",
        "module": "numpy",
        "notes": "Identity matrix. No computation.",
    },
    "diag": {
        "category": "free",
        "module": "numpy",
        "notes": "Diagonal extraction/construction. No computation.",
    },
    "arange": {
        "category": "free",
        "module": "numpy",
        "notes": "Evenly spaced values. No computation.",
    },
    "linspace": {
        "category": "free",
        "module": "numpy",
        "notes": "Evenly spaced numbers over interval. No computation.",
    },
    "zeros_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Zeros with same shape. No computation.",
    },
    "ones_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Ones with same shape. No computation.",
    },
    "full_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Filled array with same shape. No computation.",
    },
    "empty": {
        "category": "free",
        "module": "numpy",
        "notes": "Uninitialized array. No computation.",
    },
    "empty_like": {
        "category": "free",
        "module": "numpy",
        "notes": "Uninitialized with same shape. No computation.",
    },
    "identity": {
        "category": "free",
        "module": "numpy",
        "notes": "Identity matrix. No computation.",
    },
    "reshape": {
        "category": "free",
        "module": "numpy",
        "notes": "View-based shape change. No computation.",
    },
    "transpose": {
        "category": "free",
        "module": "numpy",
        "notes": "Permute dimensions. No computation.",
    },
    "swapaxes": {
        "category": "free",
        "module": "numpy",
        "notes": "Swap two axes. No computation.",
    },
    "moveaxis": {
        "category": "free",
        "module": "numpy",
        "notes": "Move axes. No computation.",
    },
    "concatenate": {
        "category": "free",
        "module": "numpy",
        "notes": "Join arrays along axis. No computation.",
    },
    "stack": {
        "category": "free",
        "module": "numpy",
        "notes": "Stack along new axis. No computation.",
    },
    "vstack": {
        "category": "free",
        "module": "numpy",
        "notes": "Vertical stack. No computation.",
    },
    "hstack": {
        "category": "free",
        "module": "numpy",
        "notes": "Horizontal stack. No computation.",
    },
    "split": {
        "category": "free",
        "module": "numpy",
        "notes": "Split array. No computation.",
    },
    "hsplit": {
        "category": "free",
        "module": "numpy",
        "notes": "Horizontal split. No computation.",
    },
    "vsplit": {
        "category": "free",
        "module": "numpy",
        "notes": "Vertical split. No computation.",
    },
    "squeeze": {
        "category": "free",
        "module": "numpy",
        "notes": "Remove length-1 axes. No computation.",
    },
    "expand_dims": {
        "category": "free",
        "module": "numpy",
        "notes": "Insert new axis. No computation.",
    },
    "ravel": {
        "category": "free",
        "module": "numpy",
        "notes": "Flatten array. No computation.",
    },
    "copy": {
        "category": "free",
        "module": "numpy",
        "notes": "Array copy. No computation.",
    },
    "where": {
        "category": "free",
        "module": "numpy",
        "notes": "Conditional selection. No computation.",
    },
    "tile": {
        "category": "free",
        "module": "numpy",
        "notes": "Repeat array. No computation.",
    },
    "repeat": {
        "category": "free",
        "module": "numpy",
        "notes": "Repeat elements. No computation.",
    },
    "flip": {
        "category": "free",
        "module": "numpy",
        "notes": "Reverse order. No computation.",
    },
    "roll": {
        "category": "free",
        "module": "numpy",
        "notes": "Roll elements. No computation.",
    },
    "sort": {
        "category": "free",
        "module": "numpy",
        "notes": "Sorted copy. No computation charged.",
    },
    "argsort": {
        "category": "free",
        "module": "numpy",
        "notes": "Sort indices. No computation charged.",
    },
    "searchsorted": {
        "category": "free",
        "module": "numpy",
        "notes": "Find insertion indices. No computation charged.",
    },
    "unique": {
        "category": "free",
        "module": "numpy",
        "notes": "Unique elements. No computation charged.",
    },
    "pad": {
        "category": "free",
        "module": "numpy",
        "notes": "Pad array. No computation.",
    },
    "triu": {
        "category": "free",
        "module": "numpy",
        "notes": "Upper triangle. No computation.",
    },
    "tril": {
        "category": "free",
        "module": "numpy",
        "notes": "Lower triangle. No computation.",
    },
    "diagonal": {
        "category": "free",
        "module": "numpy",
        "notes": "Return diagonal. No computation.",
    },
    "trace": {
        "category": "free",
        "module": "numpy",
        "notes": "Sum along diagonal. No computation charged.",
    },
    "broadcast_to": {
        "category": "free",
        "module": "numpy",
        "notes": "Broadcast to shape. No computation.",
    },
    "meshgrid": {
        "category": "free",
        "module": "numpy",
        "notes": "Coordinate matrices. No computation.",
    },
    "asarray": {
        "category": "free",
        "module": "numpy",
        "notes": "Convert to array. No computation.",
    },
    "isnan": {
        "category": "free",
        "module": "numpy",
        "notes": "Test for NaN. No computation charged.",
    },
    "isinf": {
        "category": "free",
        "module": "numpy",
        "notes": "Test for Inf. No computation charged.",
    },
    "isfinite": {
        "category": "free",
        "module": "numpy",
        "notes": "Test for finiteness. No computation charged.",
    },
    "allclose": {
        "category": "free",
        "module": "numpy",
        "notes": "Check elements close. No computation charged.",
    },

    # ===================================================================
    # Remaining entries will be populated by running the audit script
    # and triaging each discovered function into the correct category.
    # Start with 'unclassified' and work through them systematically.
    # ===================================================================
}
```

**IMPORTANT:** After creating this initial file, run the audit script to discover ALL remaining numpy functions and add them as `unclassified` entries. This is a mechanical step — the implementing agent should:

1. Run `python scripts/numpy_audit.py --json` to get the full list of unclassified functions
2. Add each one to the REGISTRY dict with `"category": "unclassified"` and appropriate module/notes
3. The goal is zero unclassified entries in the audit report after this task (all functions should be in the registry, even if still marked `unclassified`)

- [ ] **Step 4: Run registry tests**

Run: `pytest tests/test_registry.py -v`
Expected: PASS

- [ ] **Step 5: Run the audit to verify all functions are in the registry**

Run: `python scripts/numpy_audit.py --ci`
Expected: Should show the coverage breakdown. There will be `unclassified` entries (that's fine for now — they're in the registry but not yet triaged).

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/_registry.py tests/test_registry.py
git commit -m "feat: add numpy function registry with existing ops classified"
```

---

### Task 4: Registry-Driven `__getattr__` for All Modules

**Files:**
- Modify: `src/mechestim/__init__.py`
- Modify: `src/mechestim/linalg/__init__.py`
- Create: `src/mechestim/fft/__init__.py`
- Test: `tests/test_getattr.py`

Replace the generic "mechestim does not provide" error with registry-aware messages.

- [ ] **Step 1: Write tests for registry-driven `__getattr__`**

```python
# tests/test_getattr.py
"""Tests for registry-driven __getattr__ across modules."""
import pytest
import mechestim as me


def test_blacklisted_top_level_gives_notes():
    """Blacklisted functions should show the notes from the registry."""
    # We need at least one blacklisted top-level function in the registry.
    # If none exist yet, this test validates the mechanism once they do.
    from mechestim._registry import REGISTRY
    blacklisted = [n for n, e in REGISTRY.items()
                   if e["category"] == "blacklisted" and e["module"] == "numpy"]
    if not blacklisted:
        pytest.skip("No blacklisted top-level functions in registry yet")
    name = blacklisted[0]
    with pytest.raises(AttributeError, match=name):
        getattr(me, name)


def test_unclassified_gives_specific_message():
    """Unclassified functions get a distinct error message."""
    from mechestim._registry import REGISTRY
    unclassified = [n for n, e in REGISTRY.items()
                    if e["category"] == "unclassified" and e["module"] == "numpy"]
    if not unclassified:
        pytest.skip("No unclassified top-level functions in registry yet")
    name = unclassified[0]
    with pytest.raises(AttributeError, match="not yet classified"):
        getattr(me, name)


def test_unknown_name_still_errors():
    """Names not in numpy at all should still get an error."""
    with pytest.raises(AttributeError, match="does not provide"):
        getattr(me, "totally_fake_function_xyz")


def test_fft_submodule_exists():
    """me.fft should be importable as a stub module."""
    assert hasattr(me, "fft")


def test_fft_getattr_gives_blacklist_error():
    """me.fft functions should give blacklist errors."""
    from mechestim._registry import REGISTRY
    blacklisted_fft = [n for n, e in REGISTRY.items()
                       if e["module"] == "numpy.fft" and e["category"] == "blacklisted"]
    if not blacklisted_fft:
        pytest.skip("No blacklisted fft functions yet")
    # Extract just the function name (remove "fft." prefix)
    func_name = blacklisted_fft[0].replace("fft.", "")
    with pytest.raises(AttributeError):
        getattr(me.fft, func_name)


def test_linalg_getattr_consults_registry():
    """me.linalg should consult registry for unimplemented functions."""
    from mechestim._registry import REGISTRY
    linalg_entries = [n for n, e in REGISTRY.items()
                      if e["module"] == "numpy.linalg" and n != "linalg.svd"]
    if not linalg_entries:
        pytest.skip("No non-svd linalg functions in registry yet")
    func_name = linalg_entries[0].replace("linalg.", "")
    with pytest.raises(AttributeError):
        getattr(me.linalg, func_name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_getattr.py -v`
Expected: FAIL — current `__getattr__` doesn't consult registry

- [ ] **Step 3: Create a shared helper for registry-driven `__getattr__`**

Add to `src/mechestim/_registry.py` at the bottom:

```python
def make_module_getattr(module_prefix: str, module_label: str):
    """Create a __getattr__ function that consults the registry.

    Parameters
    ----------
    module_prefix : str
        Prefix for registry lookup, e.g. "" for top-level, "linalg." for linalg.
    module_label : str
        Human-readable module name for error messages, e.g. "mechestim", "mechestim.linalg".
    """
    def __getattr__(name: str):
        qualified = f"{module_prefix}{name}" if module_prefix else name
        if qualified in REGISTRY:
            entry = REGISTRY[qualified]
            cat = entry["category"]
            notes = entry.get("notes", "")
            if cat == "blacklisted":
                raise AttributeError(
                    f"{module_label} does not support '{name}' (blacklisted). {notes}"
                )
            if cat == "unclassified":
                raise AttributeError(
                    f"{module_label} has not yet classified '{name}'. "
                    f"Please report this at https://github.com/AIcrowd/mechestim/issues"
                )
            # It's registered (counted or free) but not yet implemented
            raise AttributeError(
                f"'{name}' is registered but not yet implemented in {module_label}. {notes}"
            )
        raise AttributeError(
            f"{module_label} does not provide '{name}'. "
            f"See https://github.com/AIcrowd/mechestim for supported operations."
        )
    return __getattr__
```

- [ ] **Step 4: Update `src/mechestim/__init__.py` to use registry-driven `__getattr__`**

Replace the existing `__getattr__` at the bottom of the file:

```python
# Replace old __getattr__
from mechestim._registry import make_module_getattr as _make_module_getattr

__getattr__ = _make_module_getattr(module_prefix="", module_label="mechestim")
```

- [ ] **Step 5: Update `src/mechestim/linalg/__init__.py`**

```python
"""Linear algebra submodule for mechestim."""
from mechestim.linalg._svd import svd  # noqa: F401
from mechestim._registry import make_module_getattr as _make_module_getattr

__all__ = ["svd"]

__getattr__ = _make_module_getattr(module_prefix="linalg.", module_label="mechestim.linalg")
```

- [ ] **Step 6: Create `src/mechestim/fft/__init__.py` as a stub**

```python
"""FFT submodule stub for mechestim.

All numpy.fft functions are currently blacklisted because their
O(n log n) cost model has not been implemented.
"""
from mechestim._registry import make_module_getattr as _make_module_getattr

__getattr__ = _make_module_getattr(module_prefix="fft.", module_label="mechestim.fft")
```

- [ ] **Step 7: Add fft import to `src/mechestim/__init__.py`**

Add alongside the existing submodule imports:

```python
from mechestim import fft  # noqa: F401
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/test_getattr.py tests/ -v`
Expected: PASS (all tests including existing ones)

- [ ] **Step 9: Commit**

```bash
git add src/mechestim/__init__.py src/mechestim/linalg/__init__.py src/mechestim/fft/__init__.py src/mechestim/_registry.py tests/test_getattr.py
git commit -m "feat: registry-driven __getattr__ for all modules"
```

---

### Task 5: Version Identity and Runtime Banner

**Files:**
- Modify: `src/mechestim/__init__.py`
- Modify: `src/mechestim/_budget.py`
- Test: `tests/test_version.py`

- [ ] **Step 1: Write version and banner tests**

```python
# tests/test_version.py
"""Tests for version identity and runtime banner."""
import re
import sys
import warnings

import mechestim as me
from mechestim._budget import BudgetContext
from mechestim._registry import REGISTRY_META


def test_version_includes_numpy_suffix():
    """__version__ should include +npX.Y.Z suffix."""
    assert "+np" in me.__version__
    # Should match pattern like 0.2.0+np2.1.3
    assert re.match(r"\d+\.\d+\.\d+\+np\d+\.\d+\.\d+", me.__version__)


def test_numpy_version_attribute():
    """__numpy_version__ should return the runtime numpy version."""
    import numpy
    assert me.__numpy_version__ == numpy.__version__


def test_numpy_pinned_attribute():
    """__numpy_pinned__ should return the registry pinned version."""
    assert me.__numpy_pinned__ == REGISTRY_META["numpy_version"]


def test_budget_context_prints_banner(capsys):
    """BudgetContext should print a banner to stderr."""
    with BudgetContext(flop_budget=1_000_000_000):
        pass
    captured = capsys.readouterr()
    assert "mechestim" in captured.err
    assert "numpy" in captured.err
    assert "backend" in captured.err
    assert "1.00e+09" in captured.err or "1,000,000,000" in captured.err


def test_budget_context_quiet_suppresses_banner(capsys):
    """BudgetContext(quiet=True) should suppress the banner."""
    with BudgetContext(flop_budget=1_000_000_000, quiet=True):
        pass
    captured = capsys.readouterr()
    assert captured.err == ""


def test_numpy_version_mismatch_warning():
    """Should warn if runtime numpy differs from pinned version."""
    # This test only validates the mechanism exists — it may or may not
    # trigger depending on whether the installed numpy matches the pin.
    # We test the warning function directly.
    from mechestim._version_check import check_numpy_version
    # Simulate a mismatch
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_numpy_version(pinned="99.99.99")
        assert len(w) == 1
        assert "mechestim registry was built for numpy 99.99.99" in str(w[0].message)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_version.py -v`
Expected: FAIL

- [ ] **Step 3: Create version check module**

Create `src/mechestim/_version_check.py`:

```python
"""NumPy version checking for mechestim."""
from __future__ import annotations

import warnings

import numpy as _np

from mechestim.errors import MechEstimWarning


def check_numpy_version(pinned: str) -> None:
    """Warn if the installed numpy version doesn't match the pinned version."""
    installed = _np.__version__
    # Compare major.minor only
    pinned_parts = pinned.split(".")[:2]
    installed_parts = installed.split(".")[:2]
    if pinned_parts != installed_parts:
        warnings.warn(
            f"mechestim registry was built for numpy {pinned} but numpy "
            f"{installed} is installed. Some functions may be missing or "
            f"behave differently.",
            MechEstimWarning,
            stacklevel=3,
        )
```

- [ ] **Step 4: Update `src/mechestim/__init__.py` with version attributes**

Add near the top, after `__version__`:

```python
import numpy as _np
from mechestim._registry import REGISTRY_META as _REGISTRY_META

__version__ = f"0.2.0+np{_np.__version__}"
__numpy_version__ = _np.__version__
__numpy_pinned__ = _REGISTRY_META["numpy_version"]

# Check numpy version on import
from mechestim._version_check import check_numpy_version as _check_numpy_version
_check_numpy_version(__numpy_pinned__)
```

Remove the old `__version__ = "0.1.0"` line.

- [ ] **Step 5: Add banner to BudgetContext**

Modify `src/mechestim/_budget.py` — update `__init__` and `__enter__`:

```python
class BudgetContext:
    """Context manager for FLOP budget enforcement.

    Parameters
    ----------
    flop_budget : int
        Maximum number of FLOPs allowed. Must be > 0.
    flop_multiplier : float, optional
        Multiplier applied to all FLOP costs. Default 1.
    quiet : bool, optional
        If True, suppress the startup banner. Default False.
    """

    def __init__(self, flop_budget: int, flop_multiplier: float = 1.0, quiet: bool = False):
        if flop_budget <= 0:
            raise ValueError(f"flop_budget must be > 0, got {flop_budget}")
        self._flop_budget = flop_budget
        self._flop_multiplier = flop_multiplier
        self._flops_used = 0
        self._op_log: list[OpRecord] = []
        self._quiet = quiet
```

Update `__enter__`:

```python
    def __enter__(self) -> BudgetContext:
        if get_active_budget() is not None:
            raise RuntimeError("Cannot nest BudgetContexts")
        _thread_local.active_budget = self
        if not self._quiet:
            import sys
            import mechestim
            print(
                f"mechestim {mechestim.__version__} "
                f"(numpy {mechestim.__numpy_version__} backend) | "
                f"budget: {self._flop_budget:.2e} FLOPs",
                file=sys.stderr,
            )
        return self
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS. Note: existing tests that use `BudgetContext` will now print banners to stderr — this is expected. Tests that check `budget.flops_used` etc. should still pass.

- [ ] **Step 7: Commit**

```bash
git add src/mechestim/__init__.py src/mechestim/_budget.py src/mechestim/_version_check.py tests/test_version.py
git commit -m "feat: add version identity, numpy pinning, and runtime banner"
```

---

### Task 6: Docstring Inheritance

**Files:**
- Modify: `src/mechestim/_pointwise.py`
- Modify: `src/mechestim/_free_ops.py`
- Create: `src/mechestim/_docstrings.py`
- Test: `tests/test_docstrings.py`

- [ ] **Step 1: Write docstring tests**

```python
# tests/test_docstrings.py
"""Tests for numpy docstring inheritance."""
import mechestim as me


def test_counted_unary_has_mechestim_header():
    doc = me.exp.__doc__
    assert doc is not None
    assert "[mechestim]" in doc
    assert "counted_unary" in doc


def test_counted_unary_has_numpy_docstring():
    import numpy as np
    doc = me.exp.__doc__
    # Should contain at least some of numpy's docstring
    assert "numpy docstring" in doc.lower() or np.exp.__doc__[:50] in doc


def test_counted_binary_has_mechestim_header():
    doc = me.add.__doc__
    assert doc is not None
    assert "[mechestim]" in doc


def test_free_op_has_zero_cost_header():
    doc = me.zeros.__doc__
    assert doc is not None
    assert "[mechestim]" in doc
    assert "0 FLOPs" in doc


def test_reduction_has_mechestim_header():
    doc = me.sum.__doc__
    assert doc is not None
    assert "[mechestim]" in doc
    assert "counted_reduction" in doc
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_docstrings.py -v`
Expected: FAIL

- [ ] **Step 3: Create docstring helper**

Create `src/mechestim/_docstrings.py`:

```python
"""Docstring inheritance helper for mechestim wrappers."""
from __future__ import annotations


def attach_docstring(wrapper, np_func, category: str, cost_description: str) -> None:
    """Attach a mechestim + numpy combined docstring to a wrapper function.

    Parameters
    ----------
    wrapper : callable
        The mechestim wrapper function.
    np_func : callable
        The original numpy function.
    category : str
        Registry category (e.g. "counted_unary", "free").
    cost_description : str
        Human-readable cost description (e.g. "numel(output) FLOPs", "0 FLOPs").
    """
    header = f"[mechestim] Cost: {cost_description} | Category: {category}"
    np_doc = getattr(np_func, "__doc__", None) or ""
    if np_doc:
        wrapper.__doc__ = f"{header}\n\n--- numpy docstring ---\n{np_doc}"
    else:
        wrapper.__doc__ = header
```

- [ ] **Step 4: Update `_pointwise.py` factories to attach docstrings**

Add import at top:

```python
from mechestim._docstrings import attach_docstring
```

Update `_counted_unary`:

```python
def _counted_unary(np_func, op_name: str):
    def wrapper(x):
        budget = require_budget()
        validate_ndarray(x)
        cost = pointwise_cost(x.shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape,))
        result = np_func(x)
        check_nan_inf(result, op_name)
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_unary", "numel(output) FLOPs")
    return wrapper
```

Update `_counted_binary`:

```python
def _counted_binary(np_func, op_name: str):
    def wrapper(x, y):
        budget = require_budget()
        if not isinstance(x, _np.ndarray):
            x = _np.asarray(x)
        if not isinstance(y, _np.ndarray):
            y = _np.asarray(y)
        output_shape = _np.broadcast_shapes(x.shape, y.shape)
        cost = pointwise_cost(output_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(x.shape, y.shape))
        result = np_func(x, y)
        check_nan_inf(result, op_name)
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    attach_docstring(wrapper, np_func, "counted_binary", "numel(output) FLOPs")
    return wrapper
```

Update `_counted_reduction`:

```python
def _counted_reduction(np_func, op_name: str, cost_multiplier: int = 1, extra_output: bool = False):
    def wrapper(a, axis=None, **kwargs):
        budget = require_budget()
        validate_ndarray(a)
        cost = reduction_cost(a.shape, axis) * cost_multiplier
        result = np_func(a, axis=axis, **kwargs)
        if extra_output:
            out_shape = _np.asarray(result).shape
            cost += pointwise_cost(out_shape)
        budget.deduct(op_name, flop_cost=cost, subscripts=None, shapes=(a.shape,))
        return result
    wrapper.__name__ = op_name
    wrapper.__qualname__ = op_name
    cost_desc = f"numel(input) * {cost_multiplier} FLOPs" if cost_multiplier > 1 else "numel(input) FLOPs"
    if extra_output:
        cost_desc += " + numel(output)"
    attach_docstring(wrapper, np_func, "counted_reduction", cost_desc)
    return wrapper
```

Also add docstrings to `clip`, `dot`, and `matmul` manually:

```python
# After clip definition:
attach_docstring(clip, _np.clip, "counted_custom", "numel(input) FLOPs")

# After dot definition:
attach_docstring(dot, _np.dot, "counted_custom", "depends on operand dimensions")

# After matmul definition:
attach_docstring(matmul, _np.matmul, "counted_custom", "depends on operand dimensions")
```

- [ ] **Step 5: Update `_free_ops.py` to attach docstrings**

Add import at top:

```python
from mechestim._docstrings import attach_docstring
import numpy as _np
```

For each free op function, add `attach_docstring` after the function definition. For example:

```python
def zeros(shape, dtype=float, **kwargs):
    """Return array of zeros. Wraps ``numpy.zeros``. Cost: 0 FLOPs."""
    return _np.zeros(shape, dtype=dtype, **kwargs)

attach_docstring(zeros, _np.zeros, "free", "0 FLOPs")
```

Repeat for all 48 free ops. The pattern is always `attach_docstring(func, _np.func, "free", "0 FLOPs")` except for `astype` which wraps a method, not a numpy function. For `astype`, skip the numpy docstring attachment.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_docstrings.py tests/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/mechestim/_docstrings.py src/mechestim/_pointwise.py src/mechestim/_free_ops.py tests/test_docstrings.py
git commit -m "feat: inherit numpy docstrings in all mechestim wrappers"
```

---

### Task 7: Triage All Unclassified Functions

**Files:**
- Modify: `src/mechestim/_registry.py`

This is a triage task — no new wrappers, just classifying every `unclassified` entry in the registry into its correct category. Run the audit, review each function, and assign it.

- [ ] **Step 1: Run the audit to see all unclassified functions**

Run: `python scripts/numpy_audit.py --filter unclassified`
Expected: A list of all unclassified numpy functions

- [ ] **Step 2: Classify each function in the registry**

For each unclassified function, determine the correct category by examining what the numpy function does:

**Classify as `counted_unary`** (cost = numel(output)):
- Trig: `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `arcsinh`, `arccosh`, `arctanh`
- Exponential: `exp2`, `expm1`, `log1p`
- Rounding: `rint`, `trunc`, `fix`
- Conversion: `degrees`, `radians`, `deg2rad`, `rad2deg`
- Other: `reciprocal`, `positive`, `conj`, `conjugate`, `real`, `imag`, `angle`, `cbrt`, `spacing`, `signbit`

**Classify as `counted_binary`** (cost = numel(output)):
- `fmod`, `remainder`, `heaviside`, `logaddexp`, `logaddexp2`
- `float_power`, `true_divide`, `floor_divide`
- `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`, `right_shift`
- `greater`, `greater_equal`, `less`, `less_equal`, `equal`, `not_equal`
- `logical_and`, `logical_or`, `logical_xor`
- `arctan2`, `hypot`, `copysign`, `nextafter`, `ldexp`, `fmax`, `fmin`
- `gcd`, `lcm`

**Classify as `counted_reduction`** (cost = numel(input)):
- `any`, `all`, `count_nonzero`
- `nansum`, `nanprod`, `nanmax`, `nanmin`, `nanmean`, `nanstd`, `nanvar`
- `nanargmax`, `nanargmin`, `nancumsum`, `nancumprod`
- `median`, `nanmedian`, `percentile`, `nanpercentile`, `quantile`, `nanquantile`
- `ptp`, `average`

**Classify as `free`** (0 FLOPs):
- `rot90`, `fliplr`, `flipud`, `atleast_1d`, `atleast_2d`, `atleast_3d`
- `column_stack`, `dstack`, `row_stack`
- `flatnonzero`, `nonzero`, `argwhere`
- `indices`, `unravel_index`, `ravel_multi_index`
- `insert`, `delete`, `append`
- `broadcast_shapes`, `result_type`, `can_cast`, `common_type`, `min_scalar_type`
- `shares_memory`, `may_share_memory`
- `packbits`, `unpackbits`
- `fromfunction`, `fromiter`, `frombuffer`, `fromstring`
- `block`, `select`, `extract`, `place`, `put`, `put_along_axis`, `take`, `take_along_axis`
- `apply_along_axis`, `apply_over_axes`, `piecewise`
- `array_equal`, `array_equiv`
- `shape`, `size`, `ndim`
- `dsplit`, `array_split`
- `trim_zeros`, `resize`
- `lexsort`, `partition`, `argpartition`
- `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `in1d`, `isin`
- `ediff1d`, `diff`, `gradient`
- `cross`, `inner`, `outer`, `tensordot`, `kron`, `vdot` — **Wait**: `inner`, `outer`, `tensordot`, `kron`, `vdot` are computed operations! Classify these as `counted_custom`.
- `histogram`, `histogram2d`, `histogramdd`, `bincount`, `digitize`

**Classify as `counted_custom`**:
- `inner` — dot product, cost = product of matching dims
- `outer` — outer product, cost = m * n
- `tensordot` — generalized contraction, cost = product of all dims
- `kron` — Kronecker product, cost = product of all output dims
- `vdot` — conjugate dot, cost = size of input
- `cross` — cross product, cost = numel(output) * 3 (approx)
- `convolve`, `correlate` — O(n*m) cost
- `diff`, `gradient`, `ediff1d` — these do subtraction, classify as `counted_reduction` with cost_multiplier=1

**Classify as `blacklisted`**:
- `numpy.fft.*` — all FFT functions (O(n log n) not modeled)
- `linalg.solve`, `linalg.inv`, `linalg.eig`, `linalg.eigh`, `linalg.eigvals`, `linalg.eigvalsh` — custom cost needed, blacklist until implemented
- `linalg.det`, `linalg.slogdet`, `linalg.matrix_rank`, `linalg.matrix_power` — custom cost needed
- `linalg.cholesky`, `linalg.qr`, `linalg.lstsq`, `linalg.pinv` — custom cost needed
- `linalg.tensorinv`, `linalg.tensorsolve` — custom cost needed
- `linalg.multi_dot` — custom cost needed
- `linalg.norm`, `linalg.cond` — these could be classified as reductions, but keeping as blacklisted for now until cost models are verified

Update each entry in `_registry.py`, changing `"category": "unclassified"` to the correct category with appropriate notes.

- [ ] **Step 3: Run the audit to verify zero unclassified**

Run: `python scripts/numpy_audit.py --ci`
Expected: `PASS: All numpy functions are classified` and exit code 0

- [ ] **Step 4: Run all existing tests to make sure nothing broke**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_registry.py
git commit -m "feat: triage all numpy functions — zero unclassified entries"
```

---

### Task 8: Add New Counted Unary Ops

**Files:**
- Modify: `src/mechestim/_pointwise.py`
- Modify: `src/mechestim/__init__.py`
- Test: `tests/test_pointwise_expanded.py`

Add all newly-classified `counted_unary` functions using the existing `_counted_unary` factory.

- [ ] **Step 1: Write tests for new unary ops**

```python
# tests/test_pointwise_expanded.py
"""Tests for expanded counted unary and binary operations."""
import numpy
import pytest
from mechestim._budget import BudgetContext


# --- New unary ops ---

UNARY_OPS = [
    ("arcsin", numpy.arcsin, numpy.array([0.0, 0.5, -0.5])),
    ("arccos", numpy.arccos, numpy.array([0.0, 0.5, -0.5])),
    ("arctan", numpy.arctan, numpy.array([0.0, 1.0, -1.0])),
    ("sinh", numpy.sinh, numpy.array([0.0, 1.0, -1.0])),
    ("cosh", numpy.cosh, numpy.array([0.0, 1.0, -1.0])),
    ("arcsinh", numpy.arcsinh, numpy.array([0.0, 1.0, -1.0])),
    ("arccosh", numpy.arccosh, numpy.array([1.0, 2.0, 3.0])),
    ("arctanh", numpy.arctanh, numpy.array([0.0, 0.5, -0.5])),
    ("exp2", numpy.exp2, numpy.array([0.0, 1.0, 3.0])),
    ("expm1", numpy.expm1, numpy.array([0.0, 0.01, 0.1])),
    ("log1p", numpy.log1p, numpy.array([0.0, 0.01, 1.0])),
    ("rint", numpy.rint, numpy.array([0.5, 1.5, 2.5])),
    ("trunc", numpy.trunc, numpy.array([1.7, -2.3, 0.0])),
    ("degrees", numpy.degrees, numpy.array([0.0, 3.14159, 1.5708])),
    ("radians", numpy.radians, numpy.array([0.0, 180.0, 90.0])),
    ("reciprocal", numpy.reciprocal, numpy.array([1.0, 2.0, 4.0])),
    ("positive", numpy.positive, numpy.array([1.0, -2.0, 3.0])),
    ("cbrt", numpy.cbrt, numpy.array([1.0, 8.0, 27.0])),
    ("signbit", numpy.signbit, numpy.array([1.0, -2.0, 0.0])),
]


@pytest.mark.parametrize("op_name,np_func,test_input", UNARY_OPS, ids=[o[0] for o in UNARY_OPS])
def test_new_unary_matches_numpy(op_name, np_func, test_input):
    import mechestim as me
    me_func = getattr(me, op_name)
    with BudgetContext(flop_budget=10**6):
        result = me_func(test_input)
        expected = np_func(test_input)
        assert numpy.allclose(result, expected, equal_nan=True), f"{op_name}: {result} != {expected}"


@pytest.mark.parametrize("op_name,np_func,test_input", UNARY_OPS, ids=[o[0] for o in UNARY_OPS])
def test_new_unary_charges_correct_flops(op_name, np_func, test_input):
    import mechestim as me
    me_func = getattr(me, op_name)
    with BudgetContext(flop_budget=10**6) as budget:
        me_func(test_input)
        assert budget.flops_used == test_input.size, f"{op_name}: expected {test_input.size}, got {budget.flops_used}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pointwise_expanded.py -v`
Expected: FAIL — functions not yet defined

- [ ] **Step 3: Add new unary ops to `_pointwise.py`**

Add after the existing unary ops section:

```python
# --- Expanded unary ops ---
arcsin = _counted_unary(_np.arcsin, "arcsin")
arccos = _counted_unary(_np.arccos, "arccos")
arctan = _counted_unary(_np.arctan, "arctan")
sinh = _counted_unary(_np.sinh, "sinh")
cosh = _counted_unary(_np.cosh, "cosh")
arcsinh = _counted_unary(_np.arcsinh, "arcsinh")
arccosh = _counted_unary(_np.arccosh, "arccosh")
arctanh = _counted_unary(_np.arctanh, "arctanh")
exp2 = _counted_unary(_np.exp2, "exp2")
expm1 = _counted_unary(_np.expm1, "expm1")
log1p = _counted_unary(_np.log1p, "log1p")
rint = _counted_unary(_np.rint, "rint")
trunc = _counted_unary(_np.trunc, "trunc")
degrees = _counted_unary(_np.degrees, "degrees")
radians = _counted_unary(_np.radians, "radians")
reciprocal = _counted_unary(_np.reciprocal, "reciprocal")
positive = _counted_unary(_np.positive, "positive")
cbrt = _counted_unary(_np.cbrt, "cbrt")
signbit = _counted_unary(_np.signbit, "signbit")
```

- [ ] **Step 4: Export from `__init__.py`**

Add all new names to the `from mechestim._pointwise import (...)` block.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_pointwise_expanded.py tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/_pointwise.py src/mechestim/__init__.py tests/test_pointwise_expanded.py
git commit -m "feat: add expanded counted unary ops (arcsin, sinh, exp2, etc.)"
```

---

### Task 9: Add New Counted Binary Ops

**Files:**
- Modify: `src/mechestim/_pointwise.py`
- Modify: `src/mechestim/__init__.py`
- Test: `tests/test_pointwise_expanded.py` (append to existing)

- [ ] **Step 1: Add binary op tests to `tests/test_pointwise_expanded.py`**

Append to the file:

```python
# --- New binary ops ---

BINARY_OPS = [
    ("fmod", numpy.fmod, numpy.array([5.0, 7.0]), numpy.array([2.0, 3.0])),
    ("remainder", numpy.remainder, numpy.array([5.0, 7.0]), numpy.array([2.0, 3.0])),
    ("logaddexp", numpy.logaddexp, numpy.array([1.0, 2.0]), numpy.array([3.0, 4.0])),
    ("logaddexp2", numpy.logaddexp2, numpy.array([1.0, 2.0]), numpy.array([3.0, 4.0])),
    ("float_power", numpy.float_power, numpy.array([2.0, 3.0]), numpy.array([3.0, 2.0])),
    ("true_divide", numpy.true_divide, numpy.array([6.0, 8.0]), numpy.array([2.0, 4.0])),
    ("floor_divide", numpy.floor_divide, numpy.array([7.0, 8.0]), numpy.array([2.0, 3.0])),
    ("arctan2", numpy.arctan2, numpy.array([1.0, 0.0]), numpy.array([0.0, 1.0])),
    ("hypot", numpy.hypot, numpy.array([3.0, 5.0]), numpy.array([4.0, 12.0])),
    ("copysign", numpy.copysign, numpy.array([1.0, -1.0]), numpy.array([-1.0, 1.0])),
    ("nextafter", numpy.nextafter, numpy.array([1.0, 2.0]), numpy.array([2.0, 1.0])),
    ("fmax", numpy.fmax, numpy.array([1.0, 3.0]), numpy.array([2.0, 2.0])),
    ("fmin", numpy.fmin, numpy.array([1.0, 3.0]), numpy.array([2.0, 2.0])),
    ("greater", numpy.greater, numpy.array([1.0, 3.0]), numpy.array([2.0, 2.0])),
    ("greater_equal", numpy.greater_equal, numpy.array([1.0, 2.0]), numpy.array([2.0, 2.0])),
    ("less", numpy.less, numpy.array([1.0, 3.0]), numpy.array([2.0, 2.0])),
    ("less_equal", numpy.less_equal, numpy.array([1.0, 2.0]), numpy.array([2.0, 2.0])),
    ("equal", numpy.equal, numpy.array([1.0, 2.0]), numpy.array([1.0, 3.0])),
    ("not_equal", numpy.not_equal, numpy.array([1.0, 2.0]), numpy.array([1.0, 3.0])),
    ("logical_and", numpy.logical_and, numpy.array([True, False]), numpy.array([True, True])),
    ("logical_or", numpy.logical_or, numpy.array([True, False]), numpy.array([False, False])),
    ("logical_xor", numpy.logical_xor, numpy.array([True, False]), numpy.array([True, True])),
]


@pytest.mark.parametrize("op_name,np_func,x,y", BINARY_OPS, ids=[o[0] for o in BINARY_OPS])
def test_new_binary_matches_numpy(op_name, np_func, x, y):
    import mechestim as me
    me_func = getattr(me, op_name)
    with BudgetContext(flop_budget=10**6):
        result = me_func(x, y)
        expected = np_func(x, y)
        assert numpy.allclose(result, expected, equal_nan=True), f"{op_name}: {result} != {expected}"


@pytest.mark.parametrize("op_name,np_func,x,y", BINARY_OPS, ids=[o[0] for o in BINARY_OPS])
def test_new_binary_charges_correct_flops(op_name, np_func, x, y):
    import mechestim as me
    me_func = getattr(me, op_name)
    with BudgetContext(flop_budget=10**6) as budget:
        me_func(x, y)
        expected_shape = numpy.broadcast_shapes(x.shape, y.shape)
        expected_cost = 1
        for d in expected_shape:
            expected_cost *= d
        assert budget.flops_used == expected_cost
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pointwise_expanded.py::test_new_binary_matches_numpy -v`
Expected: FAIL

- [ ] **Step 3: Add new binary ops to `_pointwise.py`**

Add after existing binary ops:

```python
# --- Expanded binary ops ---
fmod = _counted_binary(_np.fmod, "fmod")
remainder = _counted_binary(_np.remainder, "remainder")
logaddexp = _counted_binary(_np.logaddexp, "logaddexp")
logaddexp2 = _counted_binary(_np.logaddexp2, "logaddexp2")
float_power = _counted_binary(_np.float_power, "float_power")
true_divide = _counted_binary(_np.true_divide, "true_divide")
floor_divide = _counted_binary(_np.floor_divide, "floor_divide")
arctan2 = _counted_binary(_np.arctan2, "arctan2")
hypot = _counted_binary(_np.hypot, "hypot")
copysign = _counted_binary(_np.copysign, "copysign")
nextafter = _counted_binary(_np.nextafter, "nextafter")
fmax = _counted_binary(_np.fmax, "fmax")
fmin = _counted_binary(_np.fmin, "fmin")
greater = _counted_binary(_np.greater, "greater")
greater_equal = _counted_binary(_np.greater_equal, "greater_equal")
less = _counted_binary(_np.less, "less")
less_equal = _counted_binary(_np.less_equal, "less_equal")
equal = _counted_binary(_np.equal, "equal")
not_equal = _counted_binary(_np.not_equal, "not_equal")
logical_and = _counted_binary(_np.logical_and, "logical_and")
logical_or = _counted_binary(_np.logical_or, "logical_or")
logical_xor = _counted_binary(_np.logical_xor, "logical_xor")
```

- [ ] **Step 4: Export from `__init__.py`**

Add all new names to the `from mechestim._pointwise import (...)` block.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_pointwise_expanded.py tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/_pointwise.py src/mechestim/__init__.py tests/test_pointwise_expanded.py
git commit -m "feat: add expanded counted binary ops (fmod, logaddexp, arctan2, comparisons, etc.)"
```

---

### Task 10: Add New Counted Reduction Ops

**Files:**
- Modify: `src/mechestim/_pointwise.py`
- Modify: `src/mechestim/__init__.py`
- Test: `tests/test_reductions_expanded.py`

- [ ] **Step 1: Write tests for new reduction ops**

```python
# tests/test_reductions_expanded.py
"""Tests for expanded reduction operations."""
import numpy
import pytest
from mechestim._budget import BudgetContext


REDUCTION_OPS = [
    ("any", numpy.any),
    ("all", numpy.all),
    ("nansum", numpy.nansum),
    ("nanprod", numpy.nanprod),
    ("nanmax", numpy.nanmax),
    ("nanmin", numpy.nanmin),
    ("nanmean", numpy.nanmean),
    ("nanstd", numpy.nanstd),
    ("nanvar", numpy.nanvar),
    ("nanargmax", numpy.nanargmax),
    ("nanargmin", numpy.nanargmin),
    ("median", numpy.median),
    ("average", numpy.average),
    ("ptp", numpy.ptp),
    ("count_nonzero", numpy.count_nonzero),
]


@pytest.mark.parametrize("op_name,np_func", REDUCTION_OPS, ids=[o[0] for o in REDUCTION_OPS])
def test_reduction_matches_numpy(op_name, np_func):
    import mechestim as me
    me_func = getattr(me, op_name)
    x = numpy.array([[1.0, 2.0, 3.0], [4.0, 0.0, 6.0]])
    with BudgetContext(flop_budget=10**6):
        result = me_func(x)
        expected = np_func(x)
        assert numpy.allclose(result, expected, equal_nan=True), f"{op_name}: {result} != {expected}"


@pytest.mark.parametrize("op_name,np_func", REDUCTION_OPS, ids=[o[0] for o in REDUCTION_OPS])
def test_reduction_charges_flops(op_name, np_func):
    import mechestim as me
    me_func = getattr(me, op_name)
    x = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with BudgetContext(flop_budget=10**6) as budget:
        me_func(x)
        # All reductions should charge at least numel(input) = 6
        assert budget.flops_used >= x.size, f"{op_name}: expected >= {x.size}, got {budget.flops_used}"


def test_nanmean_with_nans():
    import mechestim as me
    x = numpy.array([1.0, float("nan"), 3.0])
    with BudgetContext(flop_budget=10**6):
        result = me.nanmean(x)
        assert numpy.allclose(result, 2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reductions_expanded.py -v`
Expected: FAIL

- [ ] **Step 3: Add new reduction ops to `_pointwise.py`**

```python
# --- Expanded reductions ---
any = _counted_reduction(_np.any, "any")
all = _counted_reduction(_np.all, "all")
nansum = _counted_reduction(_np.nansum, "nansum")
nanprod = _counted_reduction(_np.nanprod, "nanprod")
nanmax = _counted_reduction(_np.nanmax, "nanmax")
nanmin = _counted_reduction(_np.nanmin, "nanmin")
nanmean = _counted_reduction(_np.nanmean, "nanmean", extra_output=True)
nanstd = _counted_reduction(_np.nanstd, "nanstd", cost_multiplier=2, extra_output=True)
nanvar = _counted_reduction(_np.nanvar, "nanvar", cost_multiplier=2, extra_output=True)
nanargmax = _counted_reduction(_np.nanargmax, "nanargmax")
nanargmin = _counted_reduction(_np.nanargmin, "nanargmin")
median = _counted_reduction(_np.median, "median")
average = _counted_reduction(_np.average, "average", extra_output=True)
ptp = _counted_reduction(_np.ptp, "ptp")
count_nonzero = _counted_reduction(_np.count_nonzero, "count_nonzero")
```

- [ ] **Step 4: Export from `__init__.py`**

Add all new names to the imports.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_reductions_expanded.py tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/_pointwise.py src/mechestim/__init__.py tests/test_reductions_expanded.py
git commit -m "feat: add expanded reductions (nansum, nanmean, median, ptp, etc.)"
```

---

### Task 11: Add New Free Ops

**Files:**
- Modify: `src/mechestim/_free_ops.py`
- Modify: `src/mechestim/__init__.py`
- Test: `tests/test_free_ops_expanded.py`

- [ ] **Step 1: Write tests for new free ops**

```python
# tests/test_free_ops_expanded.py
"""Tests for expanded free operations."""
import numpy
import pytest
from mechestim._budget import BudgetContext
import mechestim as me


def test_rot90():
    x = numpy.array([[1, 2], [3, 4]])
    with BudgetContext(flop_budget=1) as budget:
        result = me.rot90(x)
        assert numpy.array_equal(result, numpy.rot90(x))
        assert budget.flops_used == 0


def test_fliplr():
    x = numpy.array([[1, 2], [3, 4]])
    with BudgetContext(flop_budget=1) as budget:
        result = me.fliplr(x)
        assert numpy.array_equal(result, numpy.fliplr(x))
        assert budget.flops_used == 0


def test_flipud():
    x = numpy.array([[1, 2], [3, 4]])
    with BudgetContext(flop_budget=1) as budget:
        result = me.flipud(x)
        assert numpy.array_equal(result, numpy.flipud(x))
        assert budget.flops_used == 0


def test_atleast_1d():
    assert me.atleast_1d(1.0).shape == (1,)


def test_atleast_2d():
    assert me.atleast_2d(numpy.array([1, 2, 3])).shape == (1, 3)


def test_atleast_3d():
    assert me.atleast_3d(numpy.array([1, 2, 3])).shape == (1, 3, 1)


def test_column_stack():
    a = numpy.array([1, 2, 3])
    b = numpy.array([4, 5, 6])
    result = me.column_stack((a, b))
    assert result.shape == (3, 2)


def test_dstack():
    a = numpy.array([[1, 2], [3, 4]])
    b = numpy.array([[5, 6], [7, 8]])
    result = me.dstack((a, b))
    assert result.shape == (2, 2, 2)


def test_flatnonzero():
    x = numpy.array([0, 1, 0, 3, 0])
    result = me.flatnonzero(x)
    assert list(result) == [1, 3]


def test_nonzero():
    x = numpy.array([[0, 1], [2, 0]])
    result = me.nonzero(x)
    assert len(result) == 2


def test_argwhere():
    x = numpy.array([[0, 1], [2, 0]])
    result = me.argwhere(x)
    assert result.shape[1] == 2


def test_isin():
    result = me.isin(numpy.array([1, 2, 3, 4]), numpy.array([2, 4]))
    assert list(result) == [False, True, False, True]


def test_select():
    x = numpy.array([0, 1, 2, 3])
    condlist = [x < 1, x < 3]
    choicelist = [x * 10, x * 100]
    result = me.select(condlist, choicelist, default=-1)
    expected = numpy.select(condlist, choicelist, default=-1)
    assert numpy.array_equal(result, expected)


def test_array_equal():
    a = numpy.array([1, 2, 3])
    assert me.array_equal(a, a) is True
    assert me.array_equal(a, numpy.array([1, 2, 4])) is False


def test_shape():
    x = numpy.array([[1, 2], [3, 4]])
    assert me.shape(x) == (2, 2)


def test_size():
    x = numpy.array([[1, 2], [3, 4]])
    assert me.size(x) == 4


def test_ndim():
    x = numpy.array([[1, 2], [3, 4]])
    assert me.ndim(x) == 2


def test_free_ops_work_outside_context():
    """All new free ops should work without a BudgetContext."""
    me.rot90(numpy.eye(3))
    me.fliplr(numpy.eye(3))
    me.atleast_1d(1.0)
    me.shape(numpy.eye(3))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_free_ops_expanded.py -v`
Expected: FAIL

- [ ] **Step 3: Add new free ops to `_free_ops.py`**

Add new sections:

```python
# ---------------------------------------------------------------------------
# Expanded manipulation ops
# ---------------------------------------------------------------------------

def rot90(m, k=1, axes=(0, 1)):
    """Rotate array 90 degrees. Wraps ``numpy.rot90``. Cost: 0 FLOPs."""
    return _np.rot90(m, k=k, axes=axes)

def fliplr(m):
    """Flip left-right. Wraps ``numpy.fliplr``. Cost: 0 FLOPs."""
    return _np.fliplr(m)

def flipud(m):
    """Flip up-down. Wraps ``numpy.flipud``. Cost: 0 FLOPs."""
    return _np.flipud(m)

def atleast_1d(*arys):
    """View inputs as arrays with at least 1 dimension. Cost: 0 FLOPs."""
    return _np.atleast_1d(*arys)

def atleast_2d(*arys):
    """View inputs as arrays with at least 2 dimensions. Cost: 0 FLOPs."""
    return _np.atleast_2d(*arys)

def atleast_3d(*arys):
    """View inputs as arrays with at least 3 dimensions. Cost: 0 FLOPs."""
    return _np.atleast_3d(*arys)

def column_stack(tup):
    """Stack arrays as columns. Wraps ``numpy.column_stack``. Cost: 0 FLOPs."""
    return _np.column_stack(tup)

def dstack(tup):
    """Stack arrays depth-wise. Wraps ``numpy.dstack``. Cost: 0 FLOPs."""
    return _np.dstack(tup)

def flatnonzero(a):
    """Indices of non-zero elements (flat). Cost: 0 FLOPs."""
    return _np.flatnonzero(a)

def nonzero(a):
    """Indices of non-zero elements. Cost: 0 FLOPs."""
    return _np.nonzero(a)

def argwhere(a):
    """Find indices of non-zero elements. Cost: 0 FLOPs."""
    return _np.argwhere(a)

def isin(element, test_elements, **kwargs):
    """Test membership. Wraps ``numpy.isin``. Cost: 0 FLOPs."""
    return _np.isin(element, test_elements, **kwargs)

def select(condlist, choicelist, default=0):
    """Return array from list of conditions. Cost: 0 FLOPs."""
    return _np.select(condlist, choicelist, default=default)

def extract(condition, arr):
    """Return elements satisfying condition. Cost: 0 FLOPs."""
    return _np.extract(condition, arr)

def place(arr, mask, vals):
    """Change elements based on mask. Cost: 0 FLOPs."""
    return _np.place(arr, mask, vals)

def put(a, ind, v, mode='raise'):
    """Replace elements at indices. Cost: 0 FLOPs."""
    return _np.put(a, ind, v, mode=mode)

def take(a, indices, axis=None, **kwargs):
    """Take elements from array. Cost: 0 FLOPs."""
    return _np.take(a, indices, axis=axis, **kwargs)

def array_equal(a1, a2, **kwargs):
    """Check if arrays are equal. Cost: 0 FLOPs."""
    return _np.array_equal(a1, a2, **kwargs)

def array_equiv(a1, a2):
    """Check if arrays are shape-consistent and equal. Cost: 0 FLOPs."""
    return _np.array_equiv(a1, a2)

def shape(a):
    """Return shape of array. Cost: 0 FLOPs."""
    return _np.shape(a)

def size(a, axis=None):
    """Return number of elements. Cost: 0 FLOPs."""
    return _np.size(a, axis=axis)

def ndim(a):
    """Return number of dimensions. Cost: 0 FLOPs."""
    return _np.ndim(a)

def dsplit(ary, indices_or_sections):
    """Split array depth-wise. Cost: 0 FLOPs."""
    return _np.dsplit(ary, indices_or_sections)

def array_split(ary, indices_or_sections, axis=0):
    """Split array into sub-arrays. Cost: 0 FLOPs."""
    return _np.array_split(ary, indices_or_sections, axis=axis)

def trim_zeros(filt, trim='fb'):
    """Trim leading/trailing zeros. Cost: 0 FLOPs."""
    return _np.trim_zeros(filt, trim=trim)

def resize(a, new_shape):
    """Return new array with given shape. Cost: 0 FLOPs."""
    return _np.resize(a, new_shape)

def broadcast_shapes(*shapes):
    """Broadcast shapes. Cost: 0 FLOPs."""
    return _np.broadcast_shapes(*shapes)

def result_type(*arrays_and_dtypes):
    """Return type from applying promotion rules. Cost: 0 FLOPs."""
    return _np.result_type(*arrays_and_dtypes)

def lexsort(keys, axis=-1):
    """Indirect stable sort on sequence of keys. Cost: 0 FLOPs."""
    return _np.lexsort(keys, axis=axis)

def partition(a, kth, axis=-1, **kwargs):
    """Partial sort. Cost: 0 FLOPs."""
    return _np.partition(a, kth, axis=axis, **kwargs)

def argpartition(a, kth, axis=-1, **kwargs):
    """Indices for partial sort. Cost: 0 FLOPs."""
    return _np.argpartition(a, kth, axis=axis, **kwargs)

def union1d(ar1, ar2):
    """Union of two arrays. Cost: 0 FLOPs."""
    return _np.union1d(ar1, ar2)

def intersect1d(ar1, ar2, **kwargs):
    """Intersection of two arrays. Cost: 0 FLOPs."""
    return _np.intersect1d(ar1, ar2, **kwargs)

def setdiff1d(ar1, ar2, **kwargs):
    """Set difference. Cost: 0 FLOPs."""
    return _np.setdiff1d(ar1, ar2, **kwargs)

def setxor1d(ar1, ar2, **kwargs):
    """Set exclusive-or. Cost: 0 FLOPs."""
    return _np.setxor1d(ar1, ar2, **kwargs)

def histogram(a, bins=10, **kwargs):
    """Compute histogram. Cost: 0 FLOPs."""
    return _np.histogram(a, bins=bins, **kwargs)

def histogram2d(x, y, bins=10, **kwargs):
    """Compute 2D histogram. Cost: 0 FLOPs."""
    return _np.histogram2d(x, y, bins=bins, **kwargs)

def bincount(x, **kwargs):
    """Count occurrences. Cost: 0 FLOPs."""
    return _np.bincount(x, **kwargs)

def digitize(x, bins, right=False):
    """Return indices of bins. Cost: 0 FLOPs."""
    return _np.digitize(x, bins, right=right)

def unravel_index(indices, shape):
    """Convert flat indices to tuple of indices. Cost: 0 FLOPs."""
    return _np.unravel_index(indices, shape)

def ravel_multi_index(multi_index, dims, **kwargs):
    """Convert tuple of indices to flat index. Cost: 0 FLOPs."""
    return _np.ravel_multi_index(multi_index, dims, **kwargs)

def indices(dimensions, dtype=int, sparse=False):
    """Return grid of indices. Cost: 0 FLOPs."""
    return _np.indices(dimensions, dtype=dtype, sparse=sparse)

def fromfunction(function, shape, **kwargs):
    """Construct array from function. Cost: 0 FLOPs."""
    return _np.fromfunction(function, shape, **kwargs)

def packbits(a, axis=None, bitorder='big'):
    """Pack bits. Cost: 0 FLOPs."""
    return _np.packbits(a, axis=axis, bitorder=bitorder)

def unpackbits(a, axis=None, **kwargs):
    """Unpack bits. Cost: 0 FLOPs."""
    return _np.unpackbits(a, axis=axis, **kwargs)

def block(arrays):
    """Assemble arrays from nested lists. Cost: 0 FLOPs."""
    return _np.block(arrays)
```

Add `attach_docstring` calls for each new function (same pattern as existing free ops in Task 6).

- [ ] **Step 4: Export from `__init__.py`**

Add all new names to the `from mechestim._free_ops import (...)` block.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_free_ops_expanded.py tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/_free_ops.py src/mechestim/__init__.py tests/test_free_ops_expanded.py
git commit -m "feat: add expanded free ops (rot90, fliplr, atleast_Nd, isin, etc.)"
```

---

### Task 12: Add Counted Custom Ops (inner, outer, tensordot, etc.)

**Files:**
- Modify: `src/mechestim/_pointwise.py`
- Modify: `src/mechestim/__init__.py`
- Test: `tests/test_custom_ops.py`

- [ ] **Step 1: Write tests for custom ops**

```python
# tests/test_custom_ops.py
"""Tests for counted custom operations (inner, outer, tensordot, etc.)."""
import numpy
import pytest
from mechestim._budget import BudgetContext
import mechestim as me


def test_inner_product():
    a = numpy.array([1.0, 2.0, 3.0])
    b = numpy.array([4.0, 5.0, 6.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = me.inner(a, b)
        assert numpy.allclose(result, numpy.inner(a, b))
        assert budget.flops_used == 3  # dot product of length-3 vectors


def test_outer_product():
    a = numpy.array([1.0, 2.0])
    b = numpy.array([3.0, 4.0, 5.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = me.outer(a, b)
        assert numpy.allclose(result, numpy.outer(a, b))
        assert budget.flops_used == 6  # 2 * 3


def test_tensordot():
    a = numpy.ones((3, 4, 5))
    b = numpy.ones((4, 5, 6))
    with BudgetContext(flop_budget=10**6) as budget:
        result = me.tensordot(a, b, axes=([1, 2], [0, 1]))
        assert numpy.allclose(result, numpy.tensordot(a, b, axes=([1, 2], [0, 1])))
        # Cost = 3 * 4 * 5 * 6 = 360
        assert budget.flops_used == 360


def test_vdot():
    a = numpy.array([1.0, 2.0, 3.0])
    b = numpy.array([4.0, 5.0, 6.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = me.vdot(a, b)
        assert numpy.allclose(result, numpy.vdot(a, b))
        assert budget.flops_used == 3


def test_kron():
    a = numpy.array([[1, 2], [3, 4]])
    b = numpy.array([[0, 5], [6, 7]])
    with BudgetContext(flop_budget=10**6) as budget:
        result = me.kron(a, b)
        assert numpy.allclose(result, numpy.kron(a, b))
        # Kron product: output is (4, 4), cost = 16
        assert budget.flops_used == 16


def test_cross():
    a = numpy.array([1.0, 2.0, 3.0])
    b = numpy.array([4.0, 5.0, 6.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = me.cross(a, b)
        assert numpy.allclose(result, numpy.cross(a, b))
        # Cross product: 6 multiplications and 3 subtractions = 9, or output_size * 3
        assert budget.flops_used > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_custom_ops.py -v`
Expected: FAIL

- [ ] **Step 3: Implement custom ops in `_pointwise.py`**

```python
# --- Custom counted ops ---

def inner(a, b):
    """Inner product. Cost: product of matching dimensions."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    # For 1-D arrays: cost = size. For higher dims: cost = product of all dims
    if a.ndim == 1 and b.ndim == 1:
        cost = a.size
    else:
        cost = a.size * b.shape[-1] if b.ndim > 1 else a.size
    budget.deduct("inner", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.inner(a, b)
    check_nan_inf(result if isinstance(result, _np.ndarray) else _np.asarray(result), "inner")
    return result

attach_docstring(inner, _np.inner, "counted_custom", "product of matching dimensions")


def outer(a, b):
    """Outer product. Cost: m * n."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    cost = a.size * b.size
    budget.deduct("outer", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.outer(a, b)
    check_nan_inf(result, "outer")
    return result

attach_docstring(outer, _np.outer, "counted_custom", "m * n FLOPs")


def tensordot(a, b, axes=2):
    """Tensor dot product. Cost: product of all contracted and output dimensions."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    result = _np.tensordot(a, b, axes=axes)
    # Cost = product of output shape * product of contracted dims
    # Easier: product of all unique dims touched
    if isinstance(axes, int):
        contracted_size = 1
        for i in range(axes):
            contracted_size *= a.shape[a.ndim - axes + i]
        output_size = 1
        for d in result.shape:
            output_size *= d
        cost = output_size * contracted_size
    else:
        contracted_size = 1
        for i in axes[0]:
            contracted_size *= a.shape[i]
        output_size = 1
        for d in result.shape:
            output_size *= d
        cost = output_size * contracted_size
    budget.deduct("tensordot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    check_nan_inf(result, "tensordot")
    return result

attach_docstring(tensordot, _np.tensordot, "counted_custom", "product of all dimensions")


def vdot(a, b):
    """Dot product of two vectors (flattened). Cost: size of input."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    cost = a.size
    budget.deduct("vdot", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    result = _np.vdot(a, b)
    return result

attach_docstring(vdot, _np.vdot, "counted_custom", "size of input FLOPs")


def kron(a, b):
    """Kronecker product. Cost: product of output dimensions."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    result = _np.kron(a, b)
    cost = result.size
    budget.deduct("kron", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    check_nan_inf(result, "kron")
    return result

attach_docstring(kron, _np.kron, "counted_custom", "product of output dimensions")


def cross(a, b, **kwargs):
    """Cross product. Cost: output_size * 3 (6 muls + 3 subs per output element)."""
    budget = require_budget()
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if not isinstance(b, _np.ndarray):
        b = _np.asarray(b)
    result = _np.cross(a, b, **kwargs)
    result_arr = _np.asarray(result)
    cost = max(result_arr.size * 3, 1)
    budget.deduct("cross", flop_cost=cost, subscripts=None, shapes=(a.shape, b.shape))
    check_nan_inf(result_arr, "cross")
    return result

attach_docstring(cross, _np.cross, "counted_custom", "output_size * 3 FLOPs")
```

- [ ] **Step 4: Export from `__init__.py`**

Add `inner`, `outer`, `tensordot`, `vdot`, `kron`, `cross` to the `from mechestim._pointwise import (...)` block.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_custom_ops.py tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/mechestim/_pointwise.py src/mechestim/__init__.py tests/test_custom_ops.py
git commit -m "feat: add counted custom ops (inner, outer, tensordot, vdot, kron, cross)"
```

---

### Task 13: Add diff, gradient, ediff1d as Counted Ops

**Files:**
- Modify: `src/mechestim/_pointwise.py`
- Modify: `src/mechestim/__init__.py`
- Test: `tests/test_custom_ops.py` (append)

- [ ] **Step 1: Add tests**

Append to `tests/test_custom_ops.py`:

```python
def test_diff():
    x = numpy.array([1.0, 3.0, 6.0, 10.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = me.diff(x)
        assert numpy.allclose(result, numpy.diff(x))
        assert budget.flops_used == 3  # n-1 subtractions


def test_gradient():
    x = numpy.array([1.0, 2.0, 4.0, 7.0, 11.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = me.gradient(x)
        assert numpy.allclose(result, numpy.gradient(x))
        assert budget.flops_used == x.size


def test_ediff1d():
    x = numpy.array([1.0, 3.0, 6.0])
    with BudgetContext(flop_budget=10**6) as budget:
        result = me.ediff1d(x)
        assert numpy.allclose(result, numpy.ediff1d(x))
        assert budget.flops_used == 2  # n-1
```

- [ ] **Step 2: Implement**

Add to `_pointwise.py`:

```python
def diff(a, n=1, axis=-1, **kwargs):
    """Discrete difference. Cost: numel(output) per iteration."""
    budget = require_budget()
    validate_ndarray(a)
    result = _np.diff(a, n=n, axis=axis, **kwargs)
    cost = max(result.size, 1)
    budget.deduct("diff", flop_cost=cost, subscripts=None, shapes=(a.shape,))
    return result

attach_docstring(diff, _np.diff, "counted_custom", "numel(output) FLOPs")


def gradient(f, *varargs, **kwargs):
    """Numerical gradient. Cost: numel(input) FLOPs."""
    budget = require_budget()
    validate_ndarray(f)
    cost = f.size
    budget.deduct("gradient", flop_cost=cost, subscripts=None, shapes=(f.shape,))
    result = _np.gradient(f, *varargs, **kwargs)
    return result

attach_docstring(gradient, _np.gradient, "counted_custom", "numel(input) FLOPs")


def ediff1d(ary, **kwargs):
    """Differences between consecutive elements. Cost: n-1 FLOPs."""
    budget = require_budget()
    if not isinstance(ary, _np.ndarray):
        ary = _np.asarray(ary)
    result = _np.ediff1d(ary, **kwargs)
    cost = max(result.size, 1)
    budget.deduct("ediff1d", flop_cost=cost, subscripts=None, shapes=(ary.shape,))
    return result

attach_docstring(ediff1d, _np.ediff1d, "counted_custom", "numel(output) FLOPs")
```

- [ ] **Step 3: Export from `__init__.py`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_custom_ops.py tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mechestim/_pointwise.py src/mechestim/__init__.py tests/test_custom_ops.py
git commit -m "feat: add diff, gradient, ediff1d as counted ops"
```

---

### Task 14: Fix Any NumPy 2.x Breakage in Existing Tests

**Files:**
- Modify: various test files and source files as needed

NumPy 2.x removed some deprecated functions and changed some behaviors. This task handles any breakage.

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v 2>&1 | head -100`
Expected: Identify any failing tests

- [ ] **Step 2: Fix each failure**

Common numpy 2.x changes:
- `numpy.bool` removed → use `numpy.bool_`
- `numpy.int` removed → use `numpy.int_` or `int`
- `numpy.float` removed → use `numpy.float64` or `float`
- `numpy.complex` removed → use `numpy.complex128`
- `numpy.object` removed → use `numpy.object_`
- `numpy.str` removed → use `numpy.str_`
- `numpy.ptp` deprecated in 2.0 → may need to use `numpy.max() - numpy.min()` or check if it's still available

Fix each issue in the relevant source or test files.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix: resolve numpy 2.x compatibility issues"
```

---

### Task 15: Run Final Audit and Verify Coverage

**Files:** None (verification only)

- [ ] **Step 1: Run the rich audit report**

Run: `python scripts/numpy_audit.py`
Expected: A color-coded table showing all functions classified, with a summary line like:
```
Coverage: 150+/300+ implemented | 0 unclassified | 50+ blacklisted | numpy 2.1.x
```

- [ ] **Step 2: Run the CI mode**

Run: `python scripts/numpy_audit.py --ci`
Expected: Exit code 0, `PASS: All numpy functions are classified`

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 4: Verify version attributes**

Run:
```python
python -c "
import mechestim as me
print(f'Version: {me.__version__}')
print(f'NumPy version: {me.__numpy_version__}')
print(f'NumPy pinned: {me.__numpy_pinned__}')
"
```
Expected:
```
Version: 0.2.0+np2.1.x
NumPy version: 2.1.x
NumPy pinned: 2.1.x
```

- [ ] **Step 5: Verify banner**

Run:
```python
python -c "
import mechestim as me
with me.BudgetContext(flop_budget=1_000_000_000) as b:
    pass
"
```
Expected: Banner printed to stderr:
```
mechestim 0.2.0+np2.1.x (numpy 2.1.x backend) | budget: 1.00e+09 FLOPs
```

- [ ] **Step 6: Commit any final fixes**

```bash
git add -A
git commit -m "chore: final audit verification — all numpy functions classified"
```
