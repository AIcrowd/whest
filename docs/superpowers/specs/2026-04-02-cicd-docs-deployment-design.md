# CI/CD and GitHub Pages Documentation Deployment

**Date:** 2026-04-02
**Status:** Approved

## Overview

Add CI/CD via GitHub Actions and deploy documentation to GitHub Pages. Single workflow file with three parallel jobs: lint, test, and docs deployment.

## Design

### 1. GitHub Actions Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main`
- Pull requests targeting `main`

**Jobs (run in parallel):**

#### Job: `lint`
- Runner: `ubuntu-latest`, Python 3.12
- Install ruff via pip
- Run `ruff check .` (linting)
- Run `ruff format --check .` (formatting verification)

#### Job: `test`
- Runner: `ubuntu-latest`
- Matrix: Python 3.10, 3.11, 3.12
- Install dependencies via `uv sync --all-extras`
- Run `uv run pytest --cov=mechestim`

#### Job: `docs`
- Condition: only on push to `main` (not PRs)
- Runner: `ubuntu-latest`, Python 3.12
- Install all dependencies via `uv sync --all-extras` (needs both `dev` and `docs` extras since `generate_api_docs.py` imports `mechestim._registry`)
- Run `uv run python scripts/generate_api_docs.py` to regenerate API reference pages from the registry
- Run `uv run python scripts/generate_api_docs.py --verify` to ensure full coverage
- Deploy via `uv run mkdocs gh-deploy --force`
- Requires `contents: write` permission for pushing to `gh-pages` branch

**Note:** `mkdocs.yml` already has `exclude_docs: superpowers/` so internal specs and plans won't appear in the deployed site.

### 2. Ruff Configuration (in `pyproject.toml`)

Add `[tool.ruff]` section:
- `line-length = 88`
- `target-version = "py310"`
- `select = ["E", "F", "W", "I"]` (pyflakes, pycodestyle errors/warnings, isort)
- No aggressive rules to avoid large-scale reformatting of existing code

### 3. README Updates

- Replace the static `[Full Documentation](docs/index.md)` link with the live GitHub Pages URL: `https://aicrowd.github.io/mechestim/`
- Add CI status badge: `![CI](https://github.com/AIcrowd/mechestim/actions/workflows/ci.yml/badge.svg)`
- Add docs badge linking to the GitHub Pages site

### 4. GitHub Pages Setup Requirement

The repository needs GitHub Pages enabled in Settings:
- Source: "Deploy from a branch"
- Branch: `gh-pages` (created automatically by `mkdocs gh-deploy`)

This is a one-time manual step by a repo admin.

## Files Changed

1. **New:** `.github/workflows/ci.yml` — CI/CD workflow
2. **Edit:** `pyproject.toml` — add `[tool.ruff]` configuration
3. **Edit:** `README.md` — add CI/docs badges, update docs link to GitHub Pages URL

## Out of Scope

- Pre-commit hooks (can be added later)
- Publishing to PyPI
- Release/tag-based workflows
- Client/server package CI (only the main library for now)
