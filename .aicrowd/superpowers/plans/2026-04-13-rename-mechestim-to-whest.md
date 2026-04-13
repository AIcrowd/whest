# Rename mechestim → whest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the `mechestim` package to `whest` across the entire codebase — directories, imports, config, docs, and branding.

**Architecture:** Five layered commits on a single branch: (1) directory renames via `git mv`, (2) package config & build system, (3) Python source code, (4) tests/examples/benchmarks, (5) docs/README/branding. Post-merge manual steps for GitHub repo rename, local directory rename, and Claude Code session migration.

**Tech Stack:** Python, hatchling (build), uv (package manager), mkdocs (docs), pytest (tests), GitHub Actions (CI)

**Spec:** `.aicrowd/superpowers/specs/2026-04-13-rename-mechestim-to-whest-design.md`

---

## Task 1: Create Branch

**Files:** None (git operation only)

- [ ] **Step 1: Create and switch to the rename branch**

```bash
git checkout -b rename-mechestim-to-whest
```

- [ ] **Step 2: Verify clean state**

```bash
git status
```

Expected: `nothing to commit, working tree clean`

---

## Task 2: Directory Renames (Commit 1)

**Files:** Directory-level `git mv` operations only. No file content changes.

- [ ] **Step 1: Rename main package directory**

```bash
git mv src/mechestim src/whest
```

- [ ] **Step 2: Rename client package directory**

```bash
git mv mechestim-client/src/mechestim mechestim-client/src/whest
git mv mechestim-client whest-client
```

- [ ] **Step 3: Rename server package directory**

```bash
git mv mechestim-server/src/mechestim_server mechestim-server/src/whest_server
git mv mechestim-server whest-server
```

- [ ] **Step 4: Verify renames**

```bash
ls src/whest/__init__.py
ls whest-client/src/whest/__init__.py
ls whest-server/src/whest_server/__init__.py
```

Expected: All three files exist.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: rename mechestim directories to whest"
```

---

## Task 3: Package Config & Build System (Commit 2)

**Files:**
- Modify: `pyproject.toml`
- Modify: `whest-client/pyproject.toml`
- Modify: `whest-server/pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `Makefile`
- Modify: `mkdocs.yml` (build-related refs only — branding in Task 6)

### Step-by-step

- [ ] **Step 1: Update main `pyproject.toml`**

Apply these replacements in `pyproject.toml`:
- `name = "mechestim"` → `name = "whest"`
- `packages = ["src/mechestim", "benchmarks"]` → `packages = ["src/whest", "benchmarks"]`
- `source = ["mechestim"]` → `source = ["whest"]`
- `--cov=mechestim` → `--cov=whest`
- `"src/mechestim/` → `"src/whest/` (in all ruff config path entries)
- `"mechestim-client/**"` → `"whest-client/**"` (ruff per-file-ignores)
- `"mechestim-server/**"` → `"whest-server/**"` (ruff per-file-ignores)

- [ ] **Step 2: Update `whest-client/pyproject.toml`**

Apply these replacements:
- `name = "mechestim-client"` → `name = "whest-client"`
- `description` text: `mechestim` → `whest`
- `packages = ["src/mechestim"]` → `packages = ["src/whest"]`

- [ ] **Step 3: Update `whest-server/pyproject.toml`**

Apply these replacements:
- `name = "mechestim-server"` → `name = "whest-server"`
- `description` text: `mechestim` → `whest`
- `"mechestim>=0.2.0"` → `"whest>=0.2.0"`
- `packages = ["src/mechestim_server"]` → `packages = ["src/whest_server"]`
- `mechestim-server = "mechestim_server.__main__:main"` → `whest-server = "whest_server.__main__:main"`

- [ ] **Step 4: Update `.github/workflows/ci.yml`**

Replace all occurrences of `mechestim` with `whest`:
- `--cov=mechestim` → `--cov=whest`
- Any other references (scan the full file)

- [ ] **Step 5: Update `Makefile`**

Replace:
- `--cov=mechestim` → `--cov=whest`
- Any other `mechestim` references

- [ ] **Step 6: Regenerate lockfile**

```bash
uv lock
```

Expected: Completes without error. The lockfile will reflect the new package name.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: update package config and build system for whest"
```

---

## Task 4: Python Source Code (Commit 3)

**Files:** All `.py` files under `src/whest/`, `whest-client/src/whest/`, `whest-server/src/whest_server/`

This is a global find-and-replace within source directories. Order matters — do specific patterns first to avoid partial matches.

- [ ] **Step 1: Replace class and error names (most specific first)**

In all `.py` files under `src/whest/`, `whest-client/src/whest/`, `whest-server/src/whest_server/`:

```
MechestimArray → WhestArray
MechEstimError → WhestError
MechEstim → Whest  (catch any remaining CamelCase variants, but verify no false positives)
```

Use case-sensitive replacement. Verify with:

```bash
grep -r "MechestimArray\|MechEstimError\|MechEstim" src/whest/ whest-client/src/whest/ whest-server/src/whest_server/
```

Expected: No matches remaining.

- [ ] **Step 2: Replace import statements**

In all `.py` files under `src/whest/`, `whest-client/src/whest/`, `whest-server/src/whest_server/`:

```
from mechestim. → from whest.
from mechestim import → from whest import
import mechestim → import whest
```

Also replace the module name in server code:
```
mechestim_server → whest_server
```

Verify:

```bash
grep -r "mechestim" src/whest/ whest-client/src/whest/ whest-server/src/whest_server/
```

Expected: No matches remaining (except possibly in changelog-style comments, which should also be updated).

- [ ] **Step 3: Replace string references**

Search for any remaining `mechestim` in source files — these will be in:
- Module docstrings (`"""mechestim ...`)
- Logger names (`getLogger("mechestim")`)
- Error messages and f-strings
- Version format strings
- `__name__` comparisons

Replace all with `whest`. Verify:

```bash
grep -ri "mechestim" src/whest/ whest-client/src/whest/ whest-server/src/whest_server/
```

Expected: Zero matches.

- [ ] **Step 4: Replace environment variable references**

In all source files:
```
MECHESTIM_SERVER_URL → WHEST_SERVER_URL
```

Verify:

```bash
grep -r "MECHESTIM" src/whest/ whest-client/src/whest/ whest-server/src/whest_server/
```

Expected: Zero matches.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: update all Python source imports and references to whest"
```

---

## Task 5: Tests, Examples & Benchmarks (Commit 4)

**Files:**
- All `.py` files under `tests/`
- All `.py` files under `examples/`
- All `.py` files under `benchmarks/`

### Tests

- [ ] **Step 1: Replace imports in test files**

In all `.py` files under `tests/`:

```
MechestimArray → WhestArray
MechEstimError → WhestError
from mechestim. → from whest.
from mechestim import → from whest import
import mechestim → import whest
```

Verify:

```bash
grep -r "mechestim\|MechEstim\|MechestimArray" tests/
```

Expected: Zero matches.

- [ ] **Step 2: Replace string assertions in tests**

Search for string literals containing "mechestim" in test files — these may appear in:
- `assert "mechestim" in str(error)`
- `assert repr(obj).startswith("mechestim")`
- Expected output comparisons

Replace `mechestim` → `whest` in these strings. Verify:

```bash
grep -ri "mechestim" tests/
```

Expected: Zero matches.

### Examples

- [ ] **Step 3: Replace imports in example files**

In all `.py` files under `examples/`:

```
import whest as we → import whest as we
```

Then replace the module alias. This requires care — only replace `me.` when it's the module alias, not a substring of another word.

Strategy: In each example file, `me` is only used as the module alias. Replace using word-boundary-aware pattern:

```bash
# For each file in examples/:
# 1. Replace the import line
# 2. Replace standalone `me.` at word boundaries
sed -i '' 's/\bme\./we./g' examples/*.py
```

Alternatively, do this with a Python script or manual editing per file, verifying each one.

Verify:

```bash
grep -r "mechestim\|import.*as me" examples/
grep -rn '\bme\.' examples/
```

Expected: No `mechestim` references. No `me.` references (only `we.`).

- [ ] **Step 4: Replace any remaining mechestim strings in examples**

Check for prose/comments mentioning mechestim:

```bash
grep -ri "mechestim" examples/
```

Replace any remaining occurrences with `whest`.

### Benchmarks

- [ ] **Step 5: Replace imports in benchmark files**

In all `.py` files under `benchmarks/`:

```
from mechestim → from whest
import mechestim → import whest
```

Also check for `mechestim` in string references (benchmark names, labels, etc.).

Verify:

```bash
grep -ri "mechestim" benchmarks/
```

Expected: Zero matches.

- [ ] **Step 6: Run tests**

```bash
uv run pytest -x -q
```

Expected: All tests pass. If failures occur, fix import issues or string mismatches before proceeding.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: update tests, examples, and benchmarks for whest"
```

---

## Task 6: Documentation, README & Branding (Commit 5)

**Files:**
- `README.md`
- `mkdocs.yml`
- All `.md` files under `docs/`
- `CHANGELOG.md`
- Docker files: `docker/Dockerfile.*`, `docker-compose.yml`, `docker/lockdown/*.py`, `docker/entrypoint.py`
- Scripts: `scripts/*.py`
- Docs visualization: `docs/visualization/symmetry-explorer/src/*.jsx` and similar

### README

- [ ] **Step 1: Update README.md**

Replace all occurrences:
- `mechestim` → `whest` (package name, text references)
- `github.com/aicrowd/mechestim` → `github.com/aicrowd/whest` (badge URLs, links)
- `import whest as we` → `import whest as we` (code examples)
- `me.` → `we.` in code examples (same word-boundary care as examples)
- `aicrowd.github.io/whest` → `aicrowd.github.io/whest` (docs URL)
- "Mechanical Estimation" → appropriate new branding where it appears as a title/heading

Verify:

```bash
grep -i "mechestim" README.md
```

Expected: Zero matches (except possibly in a "formerly known as" note if desired — user said clean break, so zero).

### mkdocs.yml

- [ ] **Step 2: Update mkdocs.yml**

Replace all occurrences:
- `site_name: mechestim` → `site_name: whest`
- `mechestim` in `site_url`, `repo_url`, `repo_name`
- `mechestim` in nav descriptions and section text
- Any path references to `mechestim`

Verify:

```bash
grep -i "mechestim" mkdocs.yml
```

Expected: Zero matches.

### Docs

- [ ] **Step 3: Update all docs markdown files**

Global find-and-replace across all `.md` files in `docs/`:

```
mechestim → whest
import whest as we → import whest as we
me. → we.  (in code blocks only, with word-boundary care)
MechestimArray → WhestArray
MechEstimError → WhestError
github.com/aicrowd/mechestim → github.com/aicrowd/whest
aicrowd.github.io/whest → aicrowd.github.io/whest
```

Verify:

```bash
grep -ri "mechestim" docs/*.md docs/**/*.md
```

Expected: Zero matches.

- [ ] **Step 4: Update docs visualization files**

Check and update JS/JSX files in `docs/visualization/`:

```bash
grep -ri "mechestim" docs/visualization/
```

Replace any occurrences with `whest`.

- [ ] **Step 5: Update docs/ops.json if needed**

```bash
grep -i "mechestim" docs/ops.json
```

Replace any occurrences.

### CHANGELOG

- [ ] **Step 6: Update CHANGELOG.md**

Add a new entry at the top:

```markdown
## [Unreleased]

### Changed
- Renamed package from `mechestim` to `whest` to reflect the new challenge name
  "ARC Whitebox Estimation Challenge". The import convention changes from
  `import whest as we` to `import whest as we`.
```

Leave historical entries as-is — they describe what happened at that time.

### Docker

- [ ] **Step 7: Update Docker files**

In `docker/Dockerfile.participant`, `docker/Dockerfile.server`, `docker/Dockerfile.participant-hardened`, `docker-compose.yml`:

```
mechestim-client → whest-client
mechestim-server → whest-server
mechestim_server → whest_server
import mechestim → import whest
mechestim.__version__ → whest.__version__
MECHESTIM_SERVER_URL → WHEST_SERVER_URL
mechestim.sock → whest.sock
mechestim-sock → whest-sock
/app/mechestim-lib/ → /app/whest-lib/
/build/mechestim-client/ → /build/whest-client/
/app/mechestim-client/ → /app/whest-client/
/app/mechestim-server/ → /app/whest-server/
```

In `docker/lockdown/*.py` and `docker/entrypoint.py`:

```
mechestim → whest
MECHESTIM_SERVER_URL → WHEST_SERVER_URL
```

Verify:

```bash
grep -ri "mechestim" docker/ docker-compose.yml
```

Expected: Zero matches.

### Scripts

- [ ] **Step 8: Update scripts**

In all `.py` files under `scripts/`:

```
from mechestim → from whest
import mechestim → import whest
mechestim → whest (in string literals, paths, comments)
```

Verify:

```bash
grep -ri "mechestim" scripts/
```

Expected: Zero matches.

### Final Verification

- [ ] **Step 9: Sweep the entire repo for any remaining mechestim references**

```bash
grep -ri "mechestim" --include="*.py" --include="*.md" --include="*.yml" --include="*.yaml" --include="*.toml" --include="*.cfg" --include="*.json" --include="*.jsx" --include="*.js" --include="*.txt" --include="*.lock" .
```

Ignore `uv.lock` (will be regenerated) and `.git/` directory. Fix any remaining occurrences.

- [ ] **Step 10: Regenerate lockfile and build docs**

```bash
uv lock
uv sync
uv run mkdocs build
```

Expected: All three succeed without errors.

- [ ] **Step 11: Run full test suite**

```bash
uv run pytest -x -q
```

Expected: All tests pass.

- [ ] **Step 12: Smoke test import**

```bash
uv run python -c "import whest as we; print(we.__version__)"
```

Expected: Prints version string without error.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "refactor: update docs, README, Docker, scripts, and branding for whest"
```

---

## Task 7: Post-Merge Operations (Manual)

These steps are performed after the PR is merged to `main`. They are **not automated** — the user performs them.

- [ ] **Step 1: Rename GitHub repo**

Go to https://github.com/aicrowd/mechestim → Settings → General → Repository name → change to `whest`.

GitHub auto-redirects the old URL.

- [ ] **Step 2: Rename local directory**

```bash
mv ~/work/AIcrowd/challenges/alignment-research-center/mechestim \
   ~/work/AIcrowd/challenges/alignment-research-center/whest
```

- [ ] **Step 3: Update git remote URL**

```bash
cd ~/work/AIcrowd/challenges/alignment-research-center/whest
git remote set-url origin git@github.com:aicrowd/whest.git
```

- [ ] **Step 4: Migrate Claude Code session history**

```bash
cp -r ~/.claude/projects/-Users-mohanty-work-AIcrowd-challenges-alignment-research-center-mechestim \
      ~/.claude/projects/-Users-mohanty-work-AIcrowd-challenges-alignment-research-center-whest
```

- [ ] **Step 5: Update Claude Code memory files**

Review and update any memory files that reference `mechestim` paths or package names:

```bash
grep -ri "mechestim" ~/.claude/projects/-Users-mohanty-work-AIcrowd-challenges-alignment-research-center-whest/memory/
```

- [ ] **Step 6: Verify everything works from the new location**

```bash
cd ~/work/AIcrowd/challenges/alignment-research-center/whest
uv sync
uv run pytest -x -q
uv run python -c "import whest as we; print(we.__version__)"
```

- [ ] **Step 7: Deploy docs**

Trigger a docs deploy (via CI or manually) to update GitHub Pages at the new URL.
