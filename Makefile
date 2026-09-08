.PHONY: help install install-deps lint lint-python lint-tui test test-python test-tui build build-tui check-commits check-pr-title check-large-files ci clean

PYTHON ?= python3
PYTHON_LINT_TARGETS ?= opendde_harness scripts tests
COMMIT_RANGE ?= origin/main..HEAD

help:
	@echo "Targets:"
	@echo "  install        Install Python deps, Node deps, and git hooks"
	@echo "  install-deps   Install Python deps only (CI uses this)"
	@echo "  lint           Run Python and TUI lint gates"
	@echo "  lint-python    Ruff-check the current lint target set"
	@echo "  lint-tui       TypeScript lint + RPC drift check"
	@echo "  test           Run Python and TUI tests"
	@echo "  check-commits  Validate Conventional Commit subjects"
	@echo "  check-pr-title Validate the PR title in PR_TITLE"
	@echo "  check-large-files Validate PR files avoid blocked assets and size bloat"
	@echo "  ci             Run the local CI gate"
	@echo "  clean          Remove generated caches and build output"

install-deps:
	uv sync --frozen --extra dev --dev

install: install-deps
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	npm ci
	npm ci --prefix ui-tui

lint: lint-python lint-tui

lint-python:
	uv run --extra dev ruff check $(PYTHON_LINT_TARGETS)

lint-tui:
	npm run lint --prefix ui-tui
	npm run lint:rpc --prefix ui-tui
	npm run type-check --prefix ui-tui

test: test-python test-tui

test-python:
	uv run pytest -q

test-tui:
	npm test --prefix ui-tui

build: build-tui

build-tui:
	npm run build --prefix ui-tui

check-commits:
	npx commitlint --from origin/main --to HEAD --config commitlint.config.cjs
	PYTHONPATH=. uv run --extra dev python scripts/check_commit_messages.py $(COMMIT_RANGE)

check-pr-title:
	PYTHONPATH=. uv run --extra dev python scripts/check_pr_title.py

check-large-files:
	PYTHONPATH=. uv run --extra dev python scripts/check_large_files.py $(COMMIT_RANGE)

ci: lint test build

clean:
	rm -rf .pytest_cache .ruff_cache .uv-cache .mypy_cache htmlcov dist build
	rm -rf ui-tui/dist ui-tui/coverage ui-tui/.vitest-cache ui-tui/packages/hermes-ink/dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
