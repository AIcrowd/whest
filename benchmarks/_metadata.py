"""Hardware & software metadata collection for benchmark reproducibility."""

from __future__ import annotations

import datetime
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _read_cpu_model() -> str:
    """Read CPU model from /proc/cpuinfo (Linux) or fall back to platform."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _read_cache_info() -> dict[str, int | None]:
    """Read CPU cache sizes from sysfs (Linux only)."""
    cache_dir = Path("/sys/devices/system/cpu/cpu0/cache")
    info: dict[str, int | None] = {"L1d": None, "L1i": None, "L2": None, "L3": None}
    if not cache_dir.exists():
        return info
    for idx in cache_dir.iterdir():
        if not idx.is_dir():
            continue
        try:
            level = (idx / "level").read_text().strip()
            ctype = (idx / "type").read_text().strip()
            size_str = (idx / "size").read_text().strip()
            # size_str is like "32K", "256K", "8192K"
            size_kb = int(size_str.rstrip("K"))
            if level == "1" and ctype == "Data":
                info["L1d"] = size_kb
            elif level == "1" and ctype == "Instruction":
                info["L1i"] = size_kb
            elif level == "2":
                info["L2"] = size_kb
            elif level == "3":
                info["L3"] = size_kb
        except (OSError, ValueError):
            continue
    return info


def _cpu_cores_threads() -> tuple[int, int]:
    """Return (physical cores, logical threads)."""
    try:
        import psutil

        return psutil.cpu_count(logical=False) or 1, psutil.cpu_count(logical=True) or 1
    except ImportError:
        count = os.cpu_count() or 1
        return count, count


def _ram_gb() -> float:
    """Return total RAM in GB."""
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        return 0.0


def _blas_info() -> str:
    """Extract BLAS library info from NumPy build config."""
    try:
        info = np.show_config(mode="dicts")
        # NumPy >= 2.0 returns a dict with "Build Dependencies" key
        if isinstance(info, dict):
            blas = info.get("Build Dependencies", {}).get("blas", {})
            if isinstance(blas, dict):
                name = blas.get("name", "unknown")
                version = blas.get("version", "unknown")
                return f"{name} {version}"
        return "unknown"
    except Exception:
        return "unknown"


PERF_EVENTS = [
    "fp_arith_inst_retired.scalar_single",
    "fp_arith_inst_retired.scalar_double",
    "fp_arith_inst_retired.128b_packed_single",
    "fp_arith_inst_retired.128b_packed_double",
]


def collect_metadata(dtype: str, repeats: int, distributions: int) -> dict[str, Any]:
    """Collect hardware/software metadata for a benchmark run.

    Parameters
    ----------
    dtype : str
        NumPy dtype string (e.g. "float64").
    repeats : int
        Number of timing repetitions per operation.
    distributions : int
        Number of input distributions to sample.

    Returns
    -------
    dict
        Metadata dictionary with timestamp, hardware, software, and
        benchmark_config sections.
    """
    cores, threads = _cpu_cores_threads()
    cache = _read_cache_info()

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": {
            "cpu_model": _read_cpu_model(),
            "cpu_cores": cores,
            "cpu_threads": threads,
            "ram_gb": _ram_gb(),
            "arch": platform.machine(),
            "cache": cache,
        },
        "software": {
            "os": f"{platform.system()} {platform.release()}",
            "python": sys.version,
            "numpy": np.__version__,
            "blas": _blas_info(),
        },
        "benchmark_config": {
            "dtype": dtype,
            "repeats": repeats,
            "distributions": distributions,
            "perf_events": PERF_EVENTS,
        },
    }
