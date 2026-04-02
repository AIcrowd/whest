# CI/CD and GitHub Pages Docs Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GitHub Actions CI/CD pipeline that lints with ruff, tests across Python 3.10–3.12, and auto-deploys docs to GitHub Pages on every push to main.

**Architecture:** Single `.github/workflows/ci.yml` with three parallel jobs (`lint`, `test`, `docs`). Ruff config lives in `pyproject.toml`. The `docs` job regenerates API reference pages from the operation registry before deploying via `mkdocs gh-deploy`.

**Tech Stack:** GitHub Actions, ruff, pytest, UV, mkdocs-material, mkdocstrings

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `.github/workflows/ci.yml` | CI/CD workflow — lint, test, docs deploy |
| Modify | `pyproject.toml` (lines 30–31, append after `[tool.pytest.ini_options]`) | Add `[tool.ruff]` configuration |
| Modify | `README.md` (lines 6–9, 17) | Add CI/docs badges, update docs link |

---

### Task 1: Add ruff configuration to pyproject.toml

**Files:**
- Modify: `pyproject.toml:30-31` (append after existing `[tool.pytest.ini_options]` block)

- [ ] **Step 1: Add ruff config**

Append the following after the `[tool.pytest.ini_options]` section at the end of `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
```

- [ ] **Step 2: Verify ruff runs cleanly**

Run:
```bash
uvx ruff check . --config pyproject.toml
uvx ruff format --check . --config pyproject.toml
```

Expected: Either clean output (exit 0) or a list of violations. If there are violations, we need to fix them before the CI would pass. Fix any issues or add targeted `per-file-ignores` if the violations are in generated/vendored code.

- [ ] **Step 3: Fix any ruff violations**

If Step 2 produced violations, fix them. Common fixes:
- Import sorting (`I` rules): `uvx ruff check --fix .` auto-fixes these
- Unused imports (`F401`): remove them
- If a few files have many cosmetic issues that aren't worth fixing right now, add a targeted ignore in pyproject.toml:
```toml
[tool.ruff.lint.per-file-ignores]
"scripts/*" = ["E501"]
```

- [ ] **Step 4: Verify ruff passes clean**

Run:
```bash
uvx ruff check .
uvx ruff format --check .
```

Expected: Exit 0, no output.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git add -u  # any files fixed by ruff
git commit -m "chore: add ruff configuration and fix lint violations"
```

---

### Task 2: Create GitHub Actions CI/CD workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Write the workflow file**

Create `.github/workflows/ci.yml` with this content:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install ruff
        run: pip install ruff

      - name: Ruff lint check
        run: ruff check .

      - name: Ruff format check
        run: ruff format --check .

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run tests
        run: uv run pytest --cov=mechestim

  docs:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Generate API docs
        run: |
          uv run python scripts/generate_api_docs.py
          uv run python scripts/generate_api_docs.py --verify

      - name: Deploy docs
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          uv run mkdocs gh-deploy --force
```

- [ ] **Step 3: Validate YAML syntax**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

Expected: No output (valid YAML). If `yaml` is not available, use:
```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

If neither works (pyyaml not installed), just visually confirm the indentation is correct — every `uses:`, `run:`, `name:` under `steps:` is indented with 8 spaces, and matrix entries are properly quoted strings.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for lint, test, and docs deploy"
```

---

### Task 3: Update README with badges and docs link

**Files:**
- Modify: `README.md:6-9` (badge section)
- Modify: `README.md:17` (docs link)

- [ ] **Step 1: Add CI and docs badges**

In `README.md`, replace the existing badge block (lines 6–9):

```markdown
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen.svg)]()
```

with:

```markdown
[![CI](https://github.com/AIcrowd/mechestim/actions/workflows/ci.yml/badge.svg)](https://github.com/AIcrowd/mechestim/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://aicrowd.github.io/mechestim/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
```

This removes the static "140 passing" badge (CI will be the live source of truth) and adds the CI + docs badges.

- [ ] **Step 2: Update the docs link**

In `README.md`, replace line 17:

```markdown
**[📚 Full Documentation](docs/index.md)**
```

with:

```markdown
**[📚 Full Documentation](https://aicrowd.github.io/mechestim/)**
```

- [ ] **Step 3: Verify README renders correctly**

Visually inspect the changes look right:
```bash
head -20 README.md
```

Expected: The badge URLs and docs link point to GitHub Actions and GitHub Pages respectively.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add CI/docs badges and link to GitHub Pages"
```

---

### Task 4: Verify everything works locally

- [ ] **Step 1: Run the full lint check**

```bash
uvx ruff check .
uvx ruff format --check .
```

Expected: Exit 0, clean.

- [ ] **Step 2: Run the full test suite**

```bash
uv run pytest --cov=mechestim
```

Expected: All tests pass.

- [ ] **Step 3: Run the docs generation pipeline**

```bash
uv run python scripts/generate_api_docs.py
uv run python scripts/generate_api_docs.py --verify
uv run mkdocs build --strict
```

Expected: All commands succeed. `mkdocs build --strict` catches any broken links or missing references.

- [ ] **Step 4: Commit any remaining fixes**

If any step above required fixes, commit them:
```bash
git add -u
git commit -m "fix: address lint/build issues found during local verification"
```

---

## Post-Merge Manual Step

After this branch is merged to `main` and the workflow runs for the first time, a repo admin needs to:

1. Go to **Settings → Pages** in the GitHub repo
2. Set Source to **"Deploy from a branch"**
3. Select branch **`gh-pages`** and root `/`
4. Save

The `gh-pages` branch is created automatically by the first `mkdocs gh-deploy` run.
