# Makefile — mirrors .github/workflows/ci.yml so local runs match remote CI.
#
#   make ci          run the full pipeline (lint → test → docs)
#   make lint        ruff check + format check
#   make test        pytest with coverage
#   make docs-build  generate API docs, verify, then build the site
#   make docs-serve  live-preview the docs locally
#
# Prerequisites:  uv sync --all-extras   (installs dev + docs extras)

SHELL := /bin/bash
UV    := uv run

# ---------------------------------------------------------------------------
# Composite targets
# ---------------------------------------------------------------------------
.PHONY: ci
ci: lint test test-numpy-compat docs-build  ## Run the full CI pipeline locally

# ---------------------------------------------------------------------------
# Lint  (mirrors: CI → lint job)
# ---------------------------------------------------------------------------
.PHONY: lint
lint:  ## Ruff lint + format check
	$(UV) ruff check .
	$(UV) ruff format --check .

.PHONY: fmt
fmt:  ## Auto-fix lint and format issues
	$(UV) ruff check --fix .
	$(UV) ruff format .

# ---------------------------------------------------------------------------
# Test  (mirrors: CI → test job)
# ---------------------------------------------------------------------------
.PHONY: test
test:  ## Run pytest with coverage
	$(UV) pytest --cov=mechestim

.PHONY: test-numpy-compat
test-numpy-compat:  ## Run NumPy's own tests against mechestim
	$(UV) pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_umath -n auto -q
	$(UV) pytest tests/numpy_compat/ --pyargs numpy._core.tests.test_ufunc -n auto -q
	$(UV) pytest tests/numpy_compat/ --pyargs numpy.linalg.tests.test_linalg -n auto -q

# ---------------------------------------------------------------------------
# Docs  (mirrors: CI → docs job)
# ---------------------------------------------------------------------------
.PHONY: docs-build
docs-build:  ## Generate + verify API docs, then build the site
	$(UV) python scripts/generate_api_docs.py
	$(UV) python scripts/generate_api_docs.py --verify
	$(UV) mkdocs build --strict

.PHONY: docs-serve
docs-serve:  ## Serve docs locally with live reload
	$(UV) mkdocs serve

.PHONY: docs-deploy
docs-deploy:  ## Deploy docs to gh-pages (same as CI)
	git config user.name  "github-actions[bot]"
	git config user.email "github-actions[bot]@users.noreply.github.com"
	$(UV) mkdocs gh-deploy --force

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: install
install:  ## Install all deps (dev + docs) and set up git hooks
	uv sync --all-extras
	git config core.hooksPath .githooks

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
