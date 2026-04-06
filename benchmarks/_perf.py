"""Wrapper around Linux ``perf stat`` for counting hardware FP operations."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PERF_EVENTS = [
    "fp_arith_inst_retired.scalar_double",
    "fp_arith_inst_retired.128b_packed_double",
    "fp_arith_inst_retired.256b_packed_double",
    "fp_arith_inst_retired.512b_packed_double",
]

# Mapping from event name suffix to SIMD width multiplier.
_WIDTH = {
    "scalar_double": 1,
    "128b_packed_double": 2,
    "256b_packed_double": 4,
    "512b_packed_double": 8,
}


@dataclass(frozen=True)
class PerfResult:
    """Counts of retired floating-point arithmetic instructions."""

    scalar_double: int
    packed_128_double: int
    packed_256_double: int
    packed_512_double: int

    @property
    def total_flops(self) -> int:
        """Total double-precision FLOPs, weighted by SIMD width."""
        return (
            self.scalar_double * 1
            + self.packed_128_double * 2
            + self.packed_256_double * 4
            + self.packed_512_double * 8
        )


def has_perf() -> bool:
    """Return True if the ``perf`` binary is on PATH."""
    return shutil.which("perf") is not None


def _parse_perf_csv(output: str) -> PerfResult:
    """Parse the CSV output produced by ``perf stat -x ,``."""
    counts: dict[str, int] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        raw_count = parts[0].strip()
        event_name = parts[2].strip()
        # Match against known events.
        for evt in PERF_EVENTS:
            if evt == event_name:
                try:
                    counts[evt] = int(raw_count)
                except (ValueError, TypeError):
                    # <not supported> or similar
                    counts[evt] = 0
                break

    return PerfResult(
        scalar_double=counts.get(PERF_EVENTS[0], 0),
        packed_128_double=counts.get(PERF_EVENTS[1], 0),
        packed_256_double=counts.get(PERF_EVENTS[2], 0),
        packed_512_double=counts.get(PERF_EVENTS[3], 0),
    )


def measure_flops(
    setup_code: str,
    bench_code: str,
    repeats: int = 10,
) -> PerfResult:
    """Run *bench_code* under ``perf stat`` and return FP-op counts.

    Parameters
    ----------
    setup_code:
        Python code executed once before the hot loop.
    bench_code:
        Python code executed *repeats* times inside the hot loop.
    repeats:
        Number of iterations of the hot loop.

    Raises
    ------
    RuntimeError
        If ``perf`` is not available on the system.
    """
    if not has_perf():
        raise RuntimeError(
            "perf is not available on this system. "
            "Install linux-tools-common or equivalent."
        )

    script = (
        f"{setup_code}\n"
        f"for _i in range({repeats}):\n"
        f"    {bench_code}\n"
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    )
    try:
        tmp.write(script)
        tmp.close()

        events_arg = ",".join(PERF_EVENTS)
        proc = subprocess.run(
            [
                "perf",
                "stat",
                "-e",
                events_arg,
                "-x",
                ",",
                sys.executable,
                tmp.name,
            ],
            capture_output=True,
            text=True,
        )
        # perf writes CSV to stderr.
        return _parse_perf_csv(proc.stderr)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
