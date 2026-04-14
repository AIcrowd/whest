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
ci: lint test test-numpy-compat check-sync docs-build  ## Run the full CI pipeline locally

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
test:  ## Run pytest with coverage (fails if < 90%)
	$(UV) pytest --cov=whest --cov-fail-under=90

.PHONY: test-numpy-compat
test-numpy-compat:  ## Run NumPy's own tests against whest
	$(UV) pytest tests/numpy_compat/ -n auto -q \
		--pyargs numpy._core.tests.test_umath \
		          numpy._core.tests.test_ufunc \
		          numpy._core.tests.test_numeric \
		          numpy.linalg.tests.test_linalg \
		          numpy.fft.tests.test_pocketfft \
		          numpy.fft.tests.test_helper \
		          numpy.polynomial.tests.test_polynomial \
		          numpy.random.tests.test_random

# ---------------------------------------------------------------------------
# Docs  (mirrors: CI → docs job)
# ---------------------------------------------------------------------------
.PHONY: docs-build
docs-build:  ## Generate API data and build Docusaurus site
	$(UV) python scripts/generate_api_docs.py
	cd website && npm run build

.PHONY: docs-serve
docs-serve:  ## Serve docs locally with live reload
	cd website && npm start

.PHONY: docs-deploy
docs-deploy:  ## Docs deploy is handled by CI on push to main
	@echo "Docs deploy is handled by CI on push to main"

# ---------------------------------------------------------------------------
# Client-Server Sync
# ---------------------------------------------------------------------------
.PHONY: check-sync
check-sync:  ## Verify client is in sync with core library
	$(UV) python scripts/sync_client.py --check
	$(UV) pytest tests/test_client_server_parity.py tests/test_serialization_parity.py -v

.PHONY: sync-client
sync-client:  ## Regenerate client files from core library
	$(UV) python scripts/sync_client.py

.PHONY: test-integration
test-integration:  ## Run client-server integration tests
	cd whest-client && $(UV) pytest tests/test_full_integration.py -v --tb=short

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
