#!/usr/bin/env python3
"""Collect only the shared libraries needed by Python + mechestim + pyzmq.

Imports all packages, finds loaded .so files, uses ldd to trace system
library dependencies, then copies them into an output directory preserving
the directory structure.

Usage:
    python collect_libs.py <output_dir>
"""

import os
import shutil
import subprocess
import sys


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    # Import everything to load all .so files
    import mechestim  # noqa: F401
    import zmq  # noqa: F401
    import msgpack  # noqa: F401

    # Collect all loaded .so paths
    so_files = set()
    for mod in sys.modules.values():
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".so"):
            so_files.add(os.path.realpath(f))

    # Also get pyzmq.libs bundled shared libs
    libs_dir = os.path.realpath(
        os.path.join(os.path.dirname(zmq.__file__), "..", "pyzmq.libs")
    )
    if os.path.isdir(libs_dir):
        for fname in os.listdir(libs_dir):
            fpath = os.path.join(libs_dir, fname)
            if os.path.isfile(fpath):
                so_files.add(fpath)

    # Use ldd to find all system lib dependencies
    needed = set()
    for so in sorted(so_files):
        try:
            out = subprocess.check_output(
                ["ldd", so], stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                parts = line.strip().split()
                if "=>" in parts:
                    idx = parts.index("=>")
                    if idx + 1 < len(parts) and parts[idx + 1].startswith("/"):
                        needed.add(os.path.realpath(parts[idx + 1]))
        except subprocess.CalledProcessError:
            pass

    # Filter to only system libraries (not site-packages — those come with python)
    system_libs = {
        p for p in needed
        if p.startswith(("/lib", "/usr/lib"))
        and "site-packages" not in p
        and os.path.exists(p)
    }

    # Copy each needed lib preserving directory structure
    for lib in sorted(system_libs):
        dest = os.path.join(output_dir, lib.lstrip("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(lib, dest)

    print(f"Collected {len(system_libs)} system shared libraries into {output_dir}")
    for lib in sorted(system_libs):
        size_kb = os.path.getsize(lib) // 1024
        print(f"  {lib} ({size_kb} KB)")


if __name__ == "__main__":
    main()
