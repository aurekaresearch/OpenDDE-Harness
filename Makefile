.PHONY: help install install-deps lint lint-python lint-tui test test-python test-tui build build-tui ci clean

PYTHON_LINT_TARGETS ?= opendde_harness scripts tests

help:
	@echo "Targets:"
	@echo "  install        Install Python deps, Node deps, and git hooks"
	@echo "  install-deps   Install Python deps only (CI uses this)"
	@echo "  lint           Run Python and TUI lint gates"
	@echo "  lint-python    Ruff-check the current lint target set"
	@echo "  lint-tui       TypeScript lint + RPC drift check"
	@echo "  test           Run Python and TUI tests"
	@echo "  ci             Run the local CI gate"
	@echo "  clean          Remove generated caches and build output"

install-deps:
	uv sync --locked --extra dev --dev

install: install-deps
	uv run pre-commit install
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

ci: lint test build

clean:
	rm -rf .pytest_cache .ruff_cache .uv-cache .mypy_cache htmlcov dist build
	rm -rf ui-tui/dist ui-tui/coverage ui-tui/.vitest-cache ui-tui/packages/hermes-ink/dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
