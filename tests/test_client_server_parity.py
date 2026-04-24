"""API surface parity tests: core library vs. flopscope-client.

Loads modules from specific file paths using importlib so that both
``src/flopscope`` (core) and ``flopscope-client/src/flopscope`` (client)
can be examined in the same process without namespace conflicts.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_TEST_DIR = Path(__file__).resolve().parent
ROOT = _TEST_DIR.parent
CORE_SRC = ROOT / "src"
CLIENT_SRC = ROOT / "flopscope-client" / "src"


# ---------------------------------------------------------------------------
# Module-loading helpers
# ---------------------------------------------------------------------------


def _load_module(file_path: Path, module_name: str) -> types.ModuleType:
    """Load a Python source file as a module with an isolated name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so intra-package imports resolve correctly
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_core(relative: str, alias: str) -> types.ModuleType:
    """Load a core module file relative to CORE_SRC."""
    return _load_module(CORE_SRC / "flopscope" / relative, alias)


def _load_client(relative: str, alias: str) -> types.ModuleType:
    """Load a client module file relative to CLIENT_SRC."""
    return _load_module(CLIENT_SRC / "flopscope" / relative, alias)


# ---------------------------------------------------------------------------
# TestRegistryParity
# ---------------------------------------------------------------------------


class TestRegistryParity:
    """core REGISTRY entries match client FUNCTION_CATEGORIES (exact set, exact categories)."""

    @pytest.fixture(scope="class")
    def core_registry(self):
        mod = _load_core("_registry.py", "_parity_core_registry")
        return mod.REGISTRY  # dict[str, dict]

    @pytest.fixture(scope="class")
    def client_categories(self):
        mod = _load_client("_registry_data.py", "_parity_client_registry_data")
        return mod.FUNCTION_CATEGORIES  # dict[str, str]

    def test_same_function_names(self, core_registry, client_categories):
        """Every function name in core REGISTRY must appear in client FUNCTION_CATEGORIES."""
        core_names = set(core_registry.keys())
        client_names = set(client_categories.keys())
        missing_in_client = core_names - client_names
        extra_in_client = client_names - core_names
        assert not missing_in_client, (
            f"Functions in core but missing from client: {sorted(missing_in_client)}"
        )
        assert not extra_in_client, (
            f"Functions in client but not in core: {sorted(extra_in_client)}"
        )

    def test_same_categories(self, core_registry, client_categories):
        """For every shared function name the category must match exactly."""
        mismatches: list[str] = []
        for name in core_registry:
            if name not in client_categories:
                continue  # already caught by test_same_function_names
            core_cat = core_registry[name]["category"]
            client_cat = client_categories[name]
            if core_cat != client_cat:
                mismatches.append(
                    f"{name!r}: core={core_cat!r} vs client={client_cat!r}"
                )
        assert not mismatches, (
            "Category mismatches between core and client:\n" + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# TestErrorParity
# ---------------------------------------------------------------------------


class TestErrorParity:
    """All error/warning classes from core errors.py exist in client errors.py."""

    # Names that are public exception/warning classes in the core module
    _EXPECTED_CORE_CLASSES = [
        "FlopscopeError",
        "BudgetExhaustedError",
        "NoBudgetContextError",
        "SymmetryError",
        "UnsupportedFunctionError",
        "FlopscopeWarning",
        "SymmetryLossWarning",
    ]

    @pytest.fixture(scope="class")
    def core_errors(self):
        # errors.py has no local imports that would break isolation
        return _load_core("errors.py", "_parity_core_errors")

    @pytest.fixture(scope="class")
    def client_errors(self):
        return _load_client("errors.py", "_parity_client_errors")

    def _public_exception_names(self, mod: types.ModuleType) -> set[str]:
        """Return names of public exception/warning classes defined in *mod*."""
        result: set[str] = set()
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, (Exception, Warning))
                and obj.__module__ == mod.__name__
            ):
                result.add(name)
        return result

    def test_core_has_expected_classes(self, core_errors):
        for cls_name in self._EXPECTED_CORE_CLASSES:
            assert hasattr(core_errors, cls_name), (
                f"Core errors.py missing expected class: {cls_name}"
            )

    def test_client_has_all_core_error_classes(self, core_errors, client_errors):
        """Every public exception/warning class in core must exist in client."""
        core_names = self._public_exception_names(core_errors)
        client_names = set(dir(client_errors))
        missing = core_names - client_names
        assert not missing, (
            f"Error classes in core but absent from client: {sorted(missing)}"
        )

    def test_client_error_classes_are_exceptions(self, core_errors, client_errors):
        """Client counterparts must actually be exception/warning subclasses."""
        core_names = self._public_exception_names(core_errors)
        for name in core_names:
            client_obj = getattr(client_errors, name, None)
            assert client_obj is not None, f"Client missing {name}"
            assert isinstance(client_obj, type), f"Client {name} is not a type"
            assert issubclass(client_obj, (Exception, Warning)), (
                f"Client {name} is not a subclass of Exception or Warning"
            )


# ---------------------------------------------------------------------------
# TestSubmoduleParity
# ---------------------------------------------------------------------------


class TestSubmoduleParity:
    """Client has matching numpy submodules (linalg/fft/random) plus top-level stats."""

    _NUMPY_SUBMODULES = ["linalg", "fft", "random"]
    _TOP_LEVEL_SUBMODULES = ["stats"]

    def _submodule_paths(self, package_src):
        for submod in self._NUMPY_SUBMODULES:
            yield package_src / "flopscope" / "numpy" / submod / "__init__.py"
        for submod in self._TOP_LEVEL_SUBMODULES:
            yield package_src / "flopscope" / submod / "__init__.py"

    def test_client_submodule_init_files_exist(self):
        for init_path in self._submodule_paths(CLIENT_SRC):
            assert init_path.exists(), (
                f"Client submodule init not found at {init_path}"
            )

    def test_core_submodule_init_files_exist(self):
        for init_path in self._submodule_paths(CORE_SRC):
            assert init_path.exists(), (
                f"Core submodule init not found at {init_path}"
            )

    def test_client_submodule_init_files_are_non_empty(self):
        for init_path in self._submodule_paths(CLIENT_SRC):
            if init_path.exists():
                assert init_path.stat().st_size > 0, (
                    f"Client {init_path} exists but is empty"
                )


# ---------------------------------------------------------------------------
# TestPermGroupParity
# ---------------------------------------------------------------------------


class TestPermGroupParity:
    """Client _perm_group.py exposes Permutation, PermutationGroup, Cycle."""

    _EXPECTED_NAMES = ["Permutation", "PermutationGroup", "Cycle"]

    @pytest.fixture(scope="class")
    def client_perm_group(self):
        return _load_client("_perm_group.py", "_parity_client_perm_group")

    @pytest.fixture(scope="class")
    def core_perm_group(self):
        return _load_core("_perm_group.py", "_parity_core_perm_group")

    def test_client_has_required_classes(self, client_perm_group):
        for name in self._EXPECTED_NAMES:
            assert hasattr(client_perm_group, name), (
                f"Client _perm_group.py missing: {name}"
            )

    def test_client_classes_are_types(self, client_perm_group):
        for name in self._EXPECTED_NAMES:
            obj = getattr(client_perm_group, name, None)
            assert obj is not None and isinstance(obj, type), (
                f"Client {name} is not a class"
            )

    def test_core_has_required_classes(self, core_perm_group):
        for name in self._EXPECTED_NAMES:
            assert hasattr(core_perm_group, name), (
                f"Core _perm_group.py missing: {name}"
            )

    def test_client_and_core_class_names_match(
        self, core_perm_group, client_perm_group
    ):
        """The set of public class names in client _perm_group must include all from core."""

        def _public_classes(mod: types.ModuleType) -> set[str]:
            return {
                name
                for name in dir(mod)
                if not name.startswith("_") and isinstance(getattr(mod, name), type)
            }

        core_classes = _public_classes(core_perm_group)
        client_classes = _public_classes(client_perm_group)
        missing = core_classes - client_classes
        assert not missing, (
            f"Classes in core _perm_group.py but absent from client: {sorted(missing)}"
        )
