# Rename mechestim to whest

## Context

The library was developed for the "ARC Whitebox Estimation Challenge" on AIcrowd,
previously called the "Mechanical Estimation Challenge" (hence `mechestim`).
With the name change, the library becomes `whest`, imported as `import whest as we`.

**GitHub repo:** `aicrowd/mechestim` → `aicrowd/whest`
**No backward compatibility** — clean break, no shim package.

## Naming Scheme

| Current | New |
|---|---|
| `mechestim` (package) | `whest` |
| `import whest as we` | `import whest as we` |
| `mechestim-client` | `whest-client` |
| `mechestim-server` | `whest-server` |
| `mechestim_server` (module) | `whest_server` |
| `MechestimArray` | `WhestArray` |
| `MechEstimError` | `WhestError` |
| mkdocs site name | `whest` |

Non-branded names (`SymmetryError`, `BudgetExhaustedError`, etc.) stay as-is.

## Approach: Layered Commits

One branch, one PR, with structured commits for each layer.
Intermediate commits may break tests — that's expected since directory renames
happen before import updates. The branch is merged as a whole.

## Commit 1: Directory & File Renames

Pure `git mv` operations, no content changes.

| Current Path | New Path |
|---|---|
| `src/mechestim/` | `src/whest/` |
| `mechestim-client/` | `whest-client/` |
| `mechestim-client/src/mechestim/` | `whest-client/src/whest/` |
| `mechestim-server/` | `whest-server/` |
| `mechestim-server/src/mechestim_server/` | `whest-server/src/whest_server/` |

Files inside these directories keep their names.

## Commit 2: Package Config & Build System

**Main `pyproject.toml`:**
- `name = "mechestim"` → `name = "whest"`
- `packages = ["src/mechestim", "benchmarks"]` → `packages = ["src/whest", "benchmarks"]`
- `source = ["mechestim"]` → `source = ["whest"]`
- `--cov=mechestim` → `--cov=whest`

**`whest-client/pyproject.toml`:**
- `name = "mechestim-client"` → `name = "whest-client"`
- `packages = ["src/mechestim"]` → `packages = ["src/whest"]`

**`whest-server/pyproject.toml`:**
- `name = "mechestim-server"` → `name = "whest-server"`
- `"mechestim>=0.2.0"` dependency → `"whest>=0.2.0"`

**CI (`.github/workflows/ci.yml`):**
- `--cov=mechestim` → `--cov=whest`

**Other config:** `Makefile`, `mkdocs.yml` build refs, Docker configs.

**`uv.lock`:** regenerated via `uv lock` (not hand-edited).

## Commit 3: Python Source Code

Global find-and-replace within `src/whest/`, `whest-client/src/whest/`, `whest-server/src/whest_server/`:

**Imports:**
- `from mechestim.` → `from whest.`
- `from mechestim import` → `from whest import`
- `import mechestim` → `import whest`

**Class & error renames:**
- `MechestimArray` → `WhestArray`
- `MechEstimError` → `WhestError`

**String references:**
- `__name__` checks, logger names, error messages, docstrings mentioning "mechestim"
- Version strings referencing the package name

Vendored `_opt_einsum` subpackage: verify, likely no mechestim references.

## Commit 4: Tests, Examples & Benchmarks

**Tests (94 files in `tests/`):**
- All `from mechestim` / `import mechestim` → `from whest` / `import whest`
- Class name references: `MechestimArray` → `WhestArray`, `MechEstimError` → `WhestError`
- String assertions referencing "mechestim" (error messages, repr output)

**Examples (11 files in `examples/`):**
- `import whest as we` → `import whest as we`
- All `me.` alias references → `we.` (careful: only replace the module alias, not the substring "me." in words like "name.foo" — use word-boundary-aware replacement)

**Benchmarks (`benchmarks/`):**
- Same import pattern updates as tests/examples.

## Commit 5: Documentation, README & Branding

**`README.md`:**
- Package name, install commands (`pip install whest`)
- Badge URLs → `github.com/aicrowd/whest`
- Code examples: `import whest as we`, `we.` prefix
- Branding: "Mechanical Estimation" → "ARC Whitebox Estimation" / "whest" as appropriate

**`mkdocs.yml`:**
- `site_name: whest`
- `repo_url`, `repo_name` → `aicrowd/whest`
- Nav entries and descriptions

**`docs/` (~20+ markdown files):**
- Code examples, import statements, install instructions
- Prose references to "mechestim" → "whest"
- API reference paths if they encode the package name

**`CHANGELOG.md`:**
- Historical entries: leave as-is
- Add new entry at top documenting the rename

## Post-Merge Operations (Manual)

These are performed after the PR is merged, not as commits.

### GitHub Repo Rename
Rename `aicrowd/mechestim` → `aicrowd/whest` via GitHub Settings > General.
GitHub auto-redirects the old URL.

### Local Directory Rename
```bash
mv ~/work/AIcrowd/challenges/alignment-research-center/mechestim \
   ~/work/AIcrowd/challenges/alignment-research-center/whest
```

### Claude Code Session Migration
```bash
cp -r ~/.claude/projects/-Users-mohanty-work-AIcrowd-challenges-alignment-research-center-mechestim \
      ~/.claude/projects/-Users-mohanty-work-AIcrowd-challenges-alignment-research-center-whest
```
Preserves all session history and memory files at the new path.

### Verification Checklist
- [ ] `uv sync` succeeds
- [ ] `uv run pytest` passes
- [ ] `uv run mkdocs build` succeeds
- [ ] `import whest as we` works in a Python REPL
- [ ] Examples run cleanly
- [ ] GitHub Pages docs deploy correctly
