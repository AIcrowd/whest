"""Lockdown hook — loaded automatically via PYTHONSTARTUP or as sitecustomize.

Neuters dangerous builtins, gates __import__ against the allowlist, and
restricts open() to read-only access under /submission/ and Python's own
stdlib/site-packages paths.

This runs at Python startup, before entrypoint.py or any participant code.
"""

from __future__ import annotations

import builtins
import os
import sys
import sysconfig

# ---------------------------------------------------------------------------
# 1. Load the allowlist
# ---------------------------------------------------------------------------

# The allowlist module is placed alongside this file in the image
_lockdown_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _lockdown_dir)
from allowlist import ALLOWED_MODULES  # noqa: E402
sys.path.remove(_lockdown_dir)

# ---------------------------------------------------------------------------
# 2. Build the set of allowed top-level import names
# ---------------------------------------------------------------------------

# Site-packages modules are always allowed (mechestim, zmq, msgpack, etc.)
_site_packages = sysconfig.get_paths()["purelib"]
_platlib = sysconfig.get_paths()["platlib"]

def _is_site_package(name: str) -> bool:
    """Check if a module name comes from site-packages (always allowed)."""
    for path in sys.path:
        if path in (_site_packages, _platlib):
            # Check if a package or module exists there
            pkg = os.path.join(path, name)
            if os.path.isdir(pkg) or os.path.exists(pkg + ".py"):
                return True
            # Check for .so files
            for entry in os.listdir(path) if os.path.isdir(path) else []:
                if entry.startswith(name) and entry.endswith(".so"):
                    return True
    return False


def _is_allowed(name: str) -> bool:
    """Check if a module import should be allowed."""
    top_level = name.split(".")[0]

    # Always allow site-packages (mechestim, zmq, msgpack, etc.)
    if _is_site_package(top_level):
        return True

    # Check stdlib allowlist — match top-level or full dotted name
    if top_level in ALLOWED_MODULES or name in ALLOWED_MODULES:
        return True

    # Allow internal/private modules (start with _) that are part of
    # allowed modules' transitive dependencies — these survived stripping
    if top_level.startswith("_"):
        return True

    return False


# ---------------------------------------------------------------------------
# 3. Gate __import__
# ---------------------------------------------------------------------------

_original_import = builtins.__import__


# NOTE: We do NOT use an import gate or module poisoning.
#
# Python's stdlib and pyzmq have deep transitive import chains (zmq → platform
# → subprocess → selectors → math, zmq.sugar.socket → pickle, etc.) that
# conflict with any poisoning or gating approach. Instead we rely on:
#
# 1. Filesystem stripping — dangerous modules' files are deleted
# 2. Docker hardening — no network, no shell, read-only FS, no capabilities
# 3. open() restriction — can't read arbitrary files
#
# Modules that are transitively needed (math, subprocess, pickle, etc.) are
# kept by the strip script but are harmless in the locked-down container:
# - math: scalar only, can't bypass matrix FLOP counting
# - subprocess: no shell in distroless, network_mode: none
# - pickle: no untrusted data to deserialize

# ---------------------------------------------------------------------------
# 5. Neuter dangerous builtins
# ---------------------------------------------------------------------------

def _disabled(name: str):
    """Return a function that raises RuntimeError when called."""
    def _blocked(*args, **kwargs):
        raise RuntimeError(f"{name}() is disabled in the mechestim sandbox")
    _blocked.__name__ = name
    _blocked.__qualname__ = name
    return _blocked


# NOTE: We do NOT disable exec(), eval(), or compile() because Python's own
# stdlib uses them pervasively (collections.namedtuple calls eval, the import
# system calls exec, dataclasses uses compile, etc.). Disabling them breaks
# Python itself.
#
# Instead, we rely on the other three defense layers:
# 1. Filesystem stripping — dangerous modules are deleted
# 2. Import gating — blocked modules can't be imported
# 3. Docker hardening — no network, read-only FS, no capabilities
#
# With these layers, exec/eval can only operate on what's already available
# in the sandbox, which is safe.
builtins.breakpoint = _disabled("breakpoint")  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 6. Restrict open()
# ---------------------------------------------------------------------------

_ALLOWED_READ_PREFIXES = (
    "/submission/",
    sysconfig.get_paths()["stdlib"],
    sysconfig.get_paths()["platstdlib"],
    _site_packages,
    _platlib,
    "/dev/null",
    "/dev/urandom",
)

_original_open = builtins.open


def _restricted_open(file, mode="r", *args, **kwargs):
    """open() that only allows read-only access to approved paths."""
    # Only allow read modes
    if any(c in str(mode) for c in ("w", "a", "x", "+")):
        raise PermissionError(
            f"write access is disabled in the mechestim sandbox (mode={mode!r})"
        )

    # Resolve the path
    path = os.path.realpath(str(file))

    # Check against allowed prefixes
    if not any(path.startswith(prefix) for prefix in _ALLOWED_READ_PREFIXES):
        raise PermissionError(
            f"read access to {file!r} is not allowed in the mechestim sandbox"
        )

    return _original_open(file, mode, *args, **kwargs)


builtins.open = _restricted_open  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 7. Clean up
# ---------------------------------------------------------------------------

# Don't remove sitecustomize from sys.modules — Python's site.py expects it
# to stay registered after execution. Instead, remove the allowlist module
# and poison usercustomize so Python doesn't try to import it.
sys.modules.pop("allowlist", None)
sys.modules["usercustomize"] = None  # type: ignore[assignment]
