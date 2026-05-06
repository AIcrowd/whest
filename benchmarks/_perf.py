"""Measure floating-point work for benchmark operations.

Primary method: Linux ``perf stat`` hardware counters (exact FP op counts).
Fallback: wall-clock time measurement (relative proxy, works everywhere).
"""

from __future__ import annotations

import os
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


@dataclass(frozen=True)
class TimingResult:
    """Wall-clock timing result used as fallback when perf is unavailable.

    Stores elapsed nanoseconds. Consumers use ``total_flops`` which returns
    the raw nanosecond value — the normalization step (op_time / add_time)
    in the runner cancels units, producing valid relative weights.
    """

    elapsed_ns: int

    @property
    def total_flops(self) -> int:
        """Return elapsed nanoseconds as a proxy for FP work.

        This is intentionally named ``total_flops`` so that all benchmark
        modules can use the same interface regardless of measurement mode.
        The values are only meaningful as ratios (normalized against the
        baseline ``np.add`` measurement).
        """
        return self.elapsed_ns


@dataclass(frozen=True)
class InstructionsResult:
    """Total retired instructions measured via ``perf stat -e instructions``.

    Used as a hardware-counter fallback for integer/bitwise operations
    where ``fp_arith_inst_retired`` reads 0. More stable than wall-clock
    timing because it is deterministic and independent of system load.

    Like ``TimingResult``, the ``total_flops`` property returns the raw
    instruction count — normalization against the baseline (``np.add``)
    in the runner cancels units, producing valid relative weights.
    """

    instructions: int

    @property
    def total_flops(self) -> int:
        """Return total retired instructions as a proxy for work."""
        return self.instructions


# Union type for all measurement modes
MeasureResult = PerfResult | TimingResult | InstructionsResult


def has_perf() -> bool:
    """Return True if the ``perf`` binary is on PATH.

    The check can be overridden by setting the environment variable
    ``FLOPSCOPE_FORCE_TIMING=1``, which forces timing mode regardless
    of whether ``perf`` is available.  This is used by the dual-mode
    validation loop to re-run benchmarks in timing mode.
    """
    if os.environ.get("FLOPSCOPE_FORCE_TIMING") == "1":
        return False
    return shutil.which("perf") is not None


def measurement_mode() -> str:
    """Return the active measurement mode: ``'perf'`` or ``'timing'``."""
    return "perf" if has_perf() else "timing"


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


def _build_script(setup_code: str, bench_code: str, repeats: int) -> str:
    """Build the benchmark Python script content."""
    return (
        "import numpy as np\n"
        f"{setup_code}\n"
        f"for _i in range({repeats}):\n"
        f"    {bench_code}\n"
    )


def _measure_perf(setup_code: str, bench_code: str, repeats: int) -> PerfResult:
    """Measure using Linux perf stat hardware counters."""
    script = _build_script(setup_code, bench_code, repeats)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
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
        return _parse_perf_csv(proc.stderr)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _measure_timing(setup_code: str, bench_code: str, repeats: int) -> TimingResult:
    """Measure using wall-clock time in a subprocess."""
    script = (
        "import time\n"
        "import numpy as np\n"
        f"{setup_code}\n"
        "# Warmup\n"
        f"for _i in range(2):\n"
        f"    {bench_code}\n"
        "# Timed run\n"
        "_t0 = time.perf_counter_ns()\n"
        f"for _i in range({repeats}):\n"
        f"    {bench_code}\n"
        "_t1 = time.perf_counter_ns()\n"
        "print(_t1 - _t0)\n"
    )

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        tmp.write(script)
        tmp.close()

        proc = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Benchmark subprocess failed (exit {proc.returncode}):\n"
                f"stderr: {proc.stderr}"
            )
        elapsed_ns = int(proc.stdout.strip())
        return TimingResult(elapsed_ns=elapsed_ns)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _measure_instructions(
    setup_code: str, bench_code: str, repeats: int
) -> InstructionsResult:
    """Measure total retired instructions via ``perf stat -e instructions``.

    This is a hardware counter that counts all retired instructions (integer,
    FP, branch, load/store). It is deterministic and independent of system
    load, making it a better fallback than wall-clock timing for integer
    and bitwise operations where ``fp_arith_inst_retired`` reads 0.
    """
    script = _build_script(setup_code, bench_code, repeats)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        tmp.write(script)
        tmp.close()

        proc = subprocess.run(
            [
                "perf",
                "stat",
                "-e",
                "instructions",
                "-x",
                ",",
                sys.executable,
                tmp.name,
            ],
            capture_output=True,
            text=True,
        )
        # Parse CSV: first field is count, third is event name
        instructions = 0
        for line in proc.stderr.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3 and "instructions" in parts[2]:
                try:
                    instructions = int(parts[0].strip())
                except (ValueError, TypeError):
                    pass
                break
        return InstructionsResult(instructions=instructions)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def measure_flops(
    setup_code: str,
    bench_code: str,
    repeats: int = 10,
) -> MeasureResult:
    """Measure FP work for a benchmark operation.

    Uses ``perf stat`` hardware counters when available (Linux). Falls back
    to wall-clock time measurement on other platforms. Both return an object
    with a ``total_flops`` property — for perf mode this is actual FP ops,
    for timing mode it is elapsed nanoseconds (valid as a relative proxy
    when normalized against the baseline).

    Parameters
    ----------
    setup_code:
        Python code executed once before the hot loop (numpy is
        already imported as ``np``).
    bench_code:
        Python code executed *repeats* times inside the hot loop.
    repeats:
        Number of iterations of the hot loop.
    """
    if has_perf():
        return _measure_perf(setup_code, bench_code, repeats)
    return _measure_timing(setup_code, bench_code, repeats)


def measure_instructions(
    setup_code: str,
    bench_code: str,
    repeats: int = 10,
) -> InstructionsResult:
    """Measure total retired instructions for a benchmark operation.

    Uses ``perf stat -e instructions`` hardware counter. This is the
    preferred fallback for integer/bitwise operations where
    ``fp_arith_inst_retired`` reads 0. Requires ``perf`` to be available.

    Falls back to timing if ``perf`` is not available.
    """
    if shutil.which("perf") is not None:
        return _measure_instructions(setup_code, bench_code, repeats)
    # If perf isn't available at all, fall back to timing
    timing = _measure_timing(setup_code, bench_code, repeats)
    return InstructionsResult(instructions=timing.elapsed_ns)
