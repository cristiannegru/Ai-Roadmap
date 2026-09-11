.PHONY: help install install-dev lint format test cov validate links progress clean

PY ?= python3
PIP ?= pip3

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime deps
	$(PIP) install -r requirements.txt

install-dev: ## Install runtime + dev deps + pre-commit
	$(PIP) install -r requirements-dev.txt
	$(PY) -m pre_commit install || true

lint: ## Ruff + Black check + Mypy
	ruff check src scripts tests templates projects
	black --check src scripts tests templates projects
	mypy src || true

format: ## Auto-format code
	ruff check --fix src scripts tests templates projects || true
	black src scripts tests templates projects

test: ## Run test suite
	pytest -q

cov: ## Run tests with coverage report
	pytest -q --cov=src --cov-report=html --cov-report=term-missing

validate: ## Validate repo structure + docs front-matter
	$(PY) scripts/validate_repo.py

links: ## Check YouTube/markdown links (no network in CI-safe mode by default)
	$(PY) scripts/check_links.py --offline

progress: ## Show learning progress tracker
	$(PY) -m ai_roadmap.progress --summary

clean: ## Remove caches and build artefacts
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
