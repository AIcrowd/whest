#!/usr/bin/env python3
"""One-time migration script: MkDocs markdown → Docusaurus format.

Copies docs/ → website/docs/, converting admonition syntax and adding
Docusaurus frontmatter + _category_.json files based on mkdocs.yml nav order.

Usage:
    python scripts/migrate_to_docusaurus.py              # run migration
    python scripts/migrate_to_docusaurus.py --dry-run    # preview only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DOCS = REPO_ROOT / "docs"
DST_DOCS = REPO_ROOT / "website" / "docs"

SKIP_DIRS = {"visualization", "javascripts", "stylesheets", "assets"}
SKIP_FILES = {"index.md"}

# Navigation ordering derived from mkdocs.yml.
# Maps directory name → (position, sidebar label, [ordered page stems]).
NAV_ORDER: dict[str, tuple[int, str, list[str]]] = {
    "getting-started": (
        1,
        "Getting Started",
        [
            "installation",
            "first-budget",
        ],
    ),
    "how-to": (
        2,
        "How-To Guides",
        [
            "migrate-from-numpy",
            "use-einsum",
            "exploit-symmetry",
            "use-linalg",
            "use-fft",
            "plan-your-budget",
            "calibrate-weights",
            "debug-budget-overruns",
        ],
    ),
    "concepts": (
        3,
        "Concepts",
        [
            "flop-counting-model",
            "operation-categories",
            "numpy-compatibility-testing",
        ],
    ),
    "architecture": (
        4,
        "Architecture",
        [
            "client-server",
            "docker",
        ],
    ),
    "explanation": (
        5,
        "Explanation",
        [
            "subgraph-symmetry",
            "symmetry-explorer",
        ],
    ),
    "development": (
        6,
        "Development",
        [
            "contributing",
        ],
    ),
    "api": (
        7,
        "API Reference",
        [
            "counted-ops",
            "free-ops",
            "symmetric",
            "linalg",
            "fft",
            "random",
            "stats",
            "polynomial",
            "window",
            "budget",
            "flops",
            "opt-einsum",
            "errors",
        ],
    ),
    "reference": (
        8,
        "Reference",
        [
            "for-agents",
            "operation-audit",
            "empirical-weights",
            "cheat-sheet",
        ],
    ),
    "troubleshooting": (
        9,
        "Troubleshooting",
        [
            "common-errors",
        ],
    ),
}

# Sidebar labels for individual pages (from mkdocs.yml nav entries).
PAGE_LABELS: dict[str, str] = {
    "getting-started/installation": "Installation",
    "getting-started/first-budget": "Your First Budget",
    "how-to/migrate-from-numpy": "Migrate from NumPy",
    "how-to/use-einsum": "Use Einsum",
    "how-to/exploit-symmetry": "Exploit Symmetry",
    "how-to/use-linalg": "Use Linear Algebra",
    "how-to/use-fft": "Use FFT",
    "how-to/plan-your-budget": "Plan Your Budget",
    "how-to/calibrate-weights": "Calibrate Weights",
    "how-to/debug-budget-overruns": "Debug Budget Overruns",
    "concepts/flop-counting-model": "FLOP Counting Model",
    "concepts/operation-categories": "Operation Categories",
    "concepts/numpy-compatibility-testing": "NumPy Compatibility Testing",
    "architecture/client-server": "Client-Server Model",
    "architecture/docker": "Running with Docker",
    "explanation/subgraph-symmetry": "Subgraph Symmetry Detection",
    "explanation/symmetry-explorer": "Symmetry Explorer",
    "development/contributing": "Contributor Guide",
    "api/counted-ops": "Counted Operations",
    "api/free-ops": "Free Operations",
    "api/symmetric": "Symmetric Tensors",
    "api/linalg": "Linear Algebra",
    "api/fft": "FFT",
    "api/random": "Random",
    "api/stats": "Statistical Distributions",
    "api/polynomial": "Polynomial",
    "api/window": "Window Functions",
    "api/budget": "Budget",
    "api/flops": "FLOP Cost Query",
    "api/opt-einsum": "Path Optimizer",
    "api/errors": "Errors",
    "reference/for-agents": "For AI Agents",
    "reference/operation-audit": "Operation Audit",
    "reference/empirical-weights": "FLOP Weight Calibration Results",
    "reference/cheat-sheet": "FLOP Cost Cheat Sheet",
    "troubleshooting/common-errors": "Common Errors",
}

# ---------------------------------------------------------------------------
# Admonition conversion
# ---------------------------------------------------------------------------

_ADMONITION_RE = re.compile(
    r'^(!{3})\s+(\w+)\s+"([^"]+)"\s*\n',
    re.MULTILINE,
)


def convert_admonitions(text: str) -> str:
    """Convert MkDocs admonition blocks to Docusaurus syntax.

    MkDocs:
        !!! type "title"
            indented content
            more content

    Docusaurus:
        :::type[title]
        content
        more content
        :::
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r'^!!!\s+(\w+)\s+"([^"]+)"\s*$', lines[i])
        if m:
            admon_type = m.group(1)
            title = m.group(2)
            result.append(f":::{admon_type}[{title}]")
            i += 1
            # Collect indented body lines (4-space indent)
            while i < len(lines):
                if lines[i] == "":
                    # Blank line inside admonition — peek ahead to see if
                    # the next non-blank line is still indented.
                    j = i + 1
                    while j < len(lines) and lines[j] == "":
                        j += 1
                    if j < len(lines) and lines[j].startswith("    "):
                        result.append("")
                        i += 1
                        continue
                    else:
                        break
                elif lines[i].startswith("    "):
                    result.append(lines[i][4:])  # un-indent
                    i += 1
                else:
                    break
            result.append(":::")
        else:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Frontmatter generation
# ---------------------------------------------------------------------------


def make_frontmatter(rel_path: Path) -> str:
    """Generate YAML frontmatter for a doc page."""
    parts = rel_path.parts  # e.g. ("how-to", "use-einsum.md")
    if len(parts) < 2:
        # Top-level file (e.g. changelog.md) — no nav ordering
        return ""

    dir_name = parts[0]
    stem = rel_path.stem  # filename without .md

    if dir_name not in NAV_ORDER:
        return ""

    _pos, _label, page_order = NAV_ORDER[dir_name]
    page_key = f"{dir_name}/{stem}"

    try:
        sidebar_position = page_order.index(stem) + 1
    except ValueError:
        sidebar_position = 99

    sidebar_label = PAGE_LABELS.get(page_key, stem.replace("-", " ").title())

    lines = [
        "---",
        f"sidebar_position: {sidebar_position}",
        f"sidebar_label: {sidebar_label}",
        "---",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category JSON generation
# ---------------------------------------------------------------------------


def make_category_json(dir_name: str) -> dict | None:
    """Create _category_.json content for a directory."""
    if dir_name not in NAV_ORDER:
        return None
    pos, label, _ = NAV_ORDER[dir_name]
    return {
        "label": label,
        "position": pos,
    }


# ---------------------------------------------------------------------------
# Main migration logic
# ---------------------------------------------------------------------------


def gather_files() -> list[Path]:
    """Collect all .md files to migrate, respecting skip rules."""
    files: list[Path] = []
    for md_file in sorted(SRC_DOCS.rglob("*.md")):
        rel = md_file.relative_to(SRC_DOCS)

        # Skip excluded files
        if rel.name in SKIP_FILES:
            continue

        # Skip excluded directories
        if any(part in SKIP_DIRS for part in rel.parts):
            continue

        files.append(rel)
    return files


def migrate(*, dry_run: bool = False) -> None:
    files = gather_files()

    # Track which directories need _category_.json
    dirs_seen: set[str] = set()

    print(f"{'[DRY RUN] ' if dry_run else ''}Migrating {len(files)} files\n")

    for rel in files:
        src = SRC_DOCS / rel
        dst = DST_DOCS / rel

        content = src.read_text(encoding="utf-8")

        # Convert admonitions
        content = convert_admonitions(content)

        # Add frontmatter
        frontmatter = make_frontmatter(rel)
        if frontmatter:
            content = frontmatter + content

        print(f"  {rel}")

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")

        # Track directory for _category_.json
        if len(rel.parts) >= 2:
            dirs_seen.add(rel.parts[0])

    # Generate _category_.json for each directory
    print()
    for dir_name in sorted(dirs_seen):
        cat = make_category_json(dir_name)
        if cat is None:
            continue
        cat_path = DST_DOCS / dir_name / "_category_.json"
        print(
            f"  {dir_name}/_category_.json  (label={cat['label']!r}, position={cat['position']})"
        )
        if not dry_run:
            cat_path.parent.mkdir(parents=True, exist_ok=True)
            cat_path.write_text(json.dumps(cat, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate MkDocs docs to Docusaurus format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without writing files",
    )
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
