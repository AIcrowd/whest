#!/usr/bin/env python3
"""Generate the slim runtime weights artifact from the rich weights source."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RICH_WEIGHTS = REPO_ROOT / "src" / "whest" / "data" / "weights.json"
DEFAULT_WEIGHTS = REPO_ROOT / "src" / "whest" / "data" / "default_weights.json"


def main() -> None:
    with RICH_WEIGHTS.open(encoding="utf-8") as f:
        rich = json.load(f)

    slim = {"weights": rich["weights"]}

    with DEFAULT_WEIGHTS.open("w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {DEFAULT_WEIGHTS}")


if __name__ == "__main__":
    main()
