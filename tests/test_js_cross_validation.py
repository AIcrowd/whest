"""Cross-validate JS symmetry explorer against Python implementation.

Runs the Node.js test script that checks all preset examples through
the JS algorithm engine and compares detected groups with expected
Python results. Requires Node.js installed.
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

import pytest

EXPLORER_DIR = Path(__file__).parent.parent / "docs" / "visualization" / "symmetry-explorer"
TEST_SCRIPT = EXPLORER_DIR / "test-cross-validate.mjs"


@pytest.mark.skipif(
    not shutil.which("node"),
    reason="Node.js not installed",
)
@pytest.mark.skipif(
    not TEST_SCRIPT.exists(),
    reason="JS cross-validation script not found",
)
def test_js_matches_python():
    """All JS preset group detections must match Python expected values."""
    result = subprocess.run(
        ["node", str(TEST_SCRIPT), "--verbose"],
        cwd=str(EXPLORER_DIR),
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Print output for debugging
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, (
        f"JS cross-validation failed:\n{result.stdout}\n{result.stderr}"
    )
