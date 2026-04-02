#!/usr/bin/env python3
"""Strip the Python stdlib to only modules required by the allowlist.

Run in the Docker builder stage AFTER installing mechestim-client, pyzmq,
and msgpack.  Discovers transitive dependencies by importing everything
on the allowlist and inspecting sys.modules, then deletes all other
stdlib files.

Usage:
    python strip_stdlib.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
import sysconfig


def _stdlib_dir() -> str:
    """Return the path to the stdlib directory."""
    return sysconfig.get_paths()["stdlib"]


def _platstdlib_dir() -> str:
    """Return the path to the platform-specific stdlib directory."""
    return sysconfig.get_paths()["platstdlib"]


def _discover_kept_files() -> set[str]:
    """Import all allowlisted + site-packages modules, return set of kept file paths."""
    # Import allowlist
    sys.path.insert(0, os.path.dirname(__file__))
    from allowlist import ALLOWED_MODULES

    # Record modules already loaded (Python startup)
    baseline = set(sys.modules.keys())

    # Import each allowlisted module to trigger transitive loads
    for mod_name in sorted(ALLOWED_MODULES):
        try:
            __import__(mod_name)
        except ImportError:
            print(f"  WARN: could not import allowlisted module {mod_name!r}", file=sys.stderr)

    # Also import the packages we need in site-packages
    for pkg in ("zmq", "msgpack", "mechestim"):
        try:
            __import__(pkg)
        except ImportError:
            print(f"  WARN: could not import package {pkg!r}", file=sys.stderr)

    # Collect all file paths from loaded modules
    kept = set()
    for name, mod in sys.modules.items():
        f = getattr(mod, "__file__", None)
        if f:
            kept.add(os.path.realpath(f))
            # Also keep .pyc and __pycache__ for this file
            pyc_dir = os.path.join(os.path.dirname(f), "__pycache__")
            base = os.path.splitext(os.path.basename(f))[0]
            if os.path.isdir(pyc_dir):
                for pyc in os.listdir(pyc_dir):
                    if pyc.startswith(base):
                        kept.add(os.path.realpath(os.path.join(pyc_dir, pyc)))

        # Also keep the package directory's __init__.py for packages
        pkg_path = getattr(mod, "__path__", None)
        if pkg_path:
            for p in pkg_path:
                init = os.path.join(p, "__init__.py")
                if os.path.exists(init):
                    kept.add(os.path.realpath(init))

    # Keep all .so files from loaded C extensions
    for name, mod in sys.modules.items():
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".so"):
            kept.add(os.path.realpath(f))

    # Keep encodings directory entirely (Python needs various codecs)
    stdlib = _stdlib_dir()
    enc_dir = os.path.join(stdlib, "encodings")
    if os.path.isdir(enc_dir):
        for root, dirs, files in os.walk(enc_dir):
            for fname in files:
                kept.add(os.path.realpath(os.path.join(root, fname)))

    return kept


def _collect_stdlib_files() -> list[str]:
    """Return all files under stdlib and platstdlib directories.

    Skips site-packages and dist-packages directories — those contain
    installed packages (pyzmq, msgpack, mechestim) that must not be touched.
    """
    dirs = {_stdlib_dir(), _platstdlib_dir()}
    skip_dirs = {"site-packages", "dist-packages"}
    all_files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, dirnames, filenames in os.walk(d):
            # Don't descend into site-packages/dist-packages
            dirnames[:] = [dn for dn in dirnames if dn not in skip_dirs]
            for fname in filenames:
                all_files.append(os.path.realpath(os.path.join(root, fname)))
    return all_files


def strip(dry_run: bool = False) -> None:
    """Strip stdlib to allowlist-only."""
    print("=== Stdlib Stripping ===")
    print(f"stdlib: {_stdlib_dir()}")
    print(f"platstdlib: {_platstdlib_dir()}")

    kept = _discover_kept_files()
    print(f"Keeping {len(kept)} files (allowlisted + transitive deps)")

    all_files = _collect_stdlib_files()
    print(f"Total stdlib files: {len(all_files)}")

    removed = 0
    for f in all_files:
        if f not in kept:
            if dry_run:
                print(f"  would remove: {f}")
            else:
                os.remove(f)
            removed += 1

    # Clean up empty directories
    if not dry_run:
        for d in [_stdlib_dir(), _platstdlib_dir()]:
            if not os.path.isdir(d):
                continue
            for root, dirnames, filenames in os.walk(d, topdown=False):
                for dirname in dirnames:
                    dirpath = os.path.join(root, dirname)
                    try:
                        os.rmdir(dirpath)  # only removes empty dirs
                    except OSError:
                        pass

    print(f"Removed {removed} files")

    # Also remove pip, setuptools, ensurepip, distutils, lib2to3
    site_packages = sysconfig.get_paths()["purelib"]
    for pkg_name in ("pip", "setuptools", "ensurepip", "distutils", "lib2to3",
                     "pkg_resources", "_distutils_hack"):
        pkg_dir = os.path.join(site_packages, pkg_name)
        if os.path.isdir(pkg_dir):
            _rmtree(pkg_dir, dry_run)
        # Also check stdlib
        stdlib_pkg = os.path.join(_stdlib_dir(), pkg_name)
        if os.path.isdir(stdlib_pkg):
            _rmtree(stdlib_pkg, dry_run)

    # Remove .dist-info for pip and setuptools
    if os.path.isdir(site_packages):
        for entry in os.listdir(site_packages):
            if entry.startswith(("pip-", "setuptools-")) and entry.endswith(".dist-info"):
                _rmtree(os.path.join(site_packages, entry), dry_run)

    # Remove stale .pth files that reference removed packages
    if os.path.isdir(site_packages):
        for entry in os.listdir(site_packages):
            if entry.endswith(".pth"):
                pth_path = os.path.join(site_packages, entry)
                if dry_run:
                    print(f"  would remove .pth: {pth_path}")
                else:
                    os.remove(pth_path)
                    print(f"  removed stale .pth: {entry}")

    print("=== Stripping complete ===")


def _rmtree(path: str, dry_run: bool) -> None:
    """Remove directory tree."""
    if dry_run:
        print(f"  would remove dir: {path}")
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    os.rmdir(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip stdlib to allowlist")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be removed")
    args = parser.parse_args()
    strip(dry_run=args.dry_run)
