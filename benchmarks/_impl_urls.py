"""Map operation names to GitHub source code URLs.

For each operation in whest, finds where the FLOP cost is charged in the
source tree and constructs a GitHub permalink.

Strategy (tried in order for each op):
1. Literal ``budget.deduct("op_name", ...)`` call — works for hand-written ops.
2. Factory registration like ``sin = _counted_unary(_np.sin, "sin")`` — works
   for pointwise ops created via factory helpers.
3. Falls back to file-level URL if only the file is identified.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/AIcrowd/whest/blob/main"
SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "whest"


def _grep(pattern: str, path: str | None = None) -> list[tuple[str, int, str]]:
    """Run grep -rn and return list of (abs_path, line_no, line_text)."""
    target = path or str(SRC_ROOT)
    try:
        result = subprocess.run(
            ["grep", "-rn", "-E", pattern, target],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    hits: list[tuple[str, int, str]] = []
    if result.returncode == 0 and result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            parts = line.split(":", 2)
            if len(parts) >= 3:
                try:
                    hits.append((parts[0], int(parts[1]), parts[2]))
                except ValueError:
                    pass
    return hits


def _to_rel_path(abs_path: str) -> str:
    """Convert absolute path to repo-relative path."""
    repo_root = SRC_ROOT.parent.parent  # up from src/whest
    try:
        return str(Path(abs_path).relative_to(repo_root))
    except ValueError:
        return str(abs_path)


def _find_deduct_line(op_name: str) -> tuple[str, int | None]:
    """Find where the FLOP cost for *op_name* is charged in the source tree.

    Returns ``(repo_relative_path, line_number)`` or ``("", None)``.
    """
    escaped = re.escape(op_name)

    # --- Strategy 1: direct deduct("op_name" or deduct('op_name' --------
    for quote in ('"', "'"):
        hits = _grep(f"deduct\\({quote}{escaped}{quote}")
        if hits:
            abs_path, line_no, _ = hits[0]
            return (_to_rel_path(abs_path), line_no)

    # --- Strategy 2: factory registration like  sin = _counted_unary(..., "sin")
    # Also handles _counted_binary, _counted_sampler, etc.
    for quote in ('"', "'"):
        # Match patterns like:
        #   sin = _counted_unary(_np.sin, "sin")
        #   random.rand = _counted_dims_sampler(_np.random.rand, "random.rand")
        pattern = f"_counted_\\w+\\([^)]*,\\s*{quote}{escaped}{quote}"
        hits = _grep(pattern)
        if hits:
            abs_path, line_no, _ = hits[0]
            return (_to_rel_path(abs_path), line_no)

    # --- Strategy 3: for reduction ops created via loops or dicts --------
    # Search for the string literal "op_name" near a deduct or factory call
    for quote in ('"', "'"):
        pattern = f"{quote}{escaped}{quote}"
        hits = _grep(pattern)
        # Filter to source files (not test, not __pycache__, not data/)
        src_hits = [
            h
            for h in hits
            if "/src/whest/" in h[0]
            and "__pycache__" not in h[0]
            and "/data/" not in h[0]
            and "_registry.py" not in h[0]
            and "_docstrings.py" not in h[0]
        ]
        if src_hits:
            abs_path, line_no, _ = src_hits[0]
            return (_to_rel_path(abs_path), line_no)

    return ("", None)


def map_op_to_url(op_name: str) -> str:
    """Return GitHub URL for the runtime cost implementation of an operation.

    Returns an empty string if the source cannot be located.
    """
    rel_path, line_no = _find_deduct_line(op_name)
    if not rel_path:
        return ""
    url = f"{REPO_URL}/{rel_path}"
    if line_no is not None:
        url += f"#L{line_no}"
    return url


def build_url_map(op_names: list[str] | None = None) -> dict[str, str]:
    """Build URL map for all given operation names.

    Parameters
    ----------
    op_names : list of str, optional
        If *None*, reads all operation names from ``weights.json``.

    Returns
    -------
    dict mapping operation name to GitHub URL (empty string if not found).
    """
    if op_names is None:
        import json

        weights_path = SRC_ROOT / "data" / "weights.json"
        with open(weights_path) as f:
            data = json.load(f)
        op_names = list(data["weights"].keys())

    return {op: map_op_to_url(op) for op in op_names}
