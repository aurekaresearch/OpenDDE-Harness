# AGENTS.md - OpenDDE Harness Agent Guide

This file applies to the whole repository unless a deeper `AGENTS.md` overrides it.

## Core rules

- Check `git status --short` before editing; preserve user changes.
- Make the smallest task-focused change. Do not refactor, rename, or format unrelated files, or add unrequested files.
- Prefer existing dependencies and patterns; avoid speculative abstractions.
- Keep CLI flags, config keys, RPC schemas, output formats, docs, and tests consistent.
- Never commit secrets, private paths, checkpoints, databases, caches, generated outputs, or standalone report/web artifacts. Files over 1 MiB need explicit maintainer approval before commit.

## Repository map

- `opendde_harness/agent/`, `context_engine/`, `memory_engine/`: agent execution, context, and memory.
- `opendde_harness/cli/`, `config/`: `ddeharness` commands and configuration.
- `opendde_harness/plugin/protein_design/`: antibody design workflows and compute backends.
- `opendde_harness/providers/`, `session/`, `sandbox/`, `security/`: model providers, sessions, and execution boundaries.
- `opendde_harness/tui_rpc/`, `tracing/`: TUI communication and tracing dashboard.
- `ui-tui/`: TypeScript/React terminal UI.
- `tests/`: Python tests; `tests/integration/` requires real resources.
- `docs/`, `docker/`: user guides, design examples, and compute image definitions.

## Environment and commands

- Python >=3.12; Node.js >=22 for the TUI.
- Manage Python dependencies with `uv add`, `uv remove`, `uv sync`, and `uv lock`; do not use pip or hand-edit dependency lists or `uv.lock` unless explicitly requested.
- Install development dependencies with `make install`.

```bash
uv run --extra dev ruff check <changed_python_paths>
uv run pytest <relevant_tests> -q
make lint-tui
make test-tui
make build-tui
```

- Run checks relevant to the change. Python tests exclude integration and e2e markers by default; select them explicitly when needed.
- GPU, Docker, model weights, external services, and LLM credentials may be unavailable. State exactly what was not validated.

## Code and tests

- Follow nearby naming, formatting, and logging patterns. Add English comments only for non-obvious logic or constraints.
- Python formatting follows the repository's Ruff configuration. JS/TS formatting follows the repository's Prettier configuration; do not substitute manual formatting or lint-only checks.
- Preserve optional-backend behavior and client/compute boundaries; do not assume CUDA or local model assets exist.
- Keep unit tests under `tests/test_*.py`. CLI changes update the existing `tests/test_cli_<module>_commands.py`; helper and cross-module tests may use descriptive suffixes.
- Integration tests use `tests/integration/test_<scope>_<kind>.py`, with `kind` equal to `e2e`, `smoke`, or `real_<resource>`.
- Do not name tests after phases, versions, or tickets. Report legacy naming violations instead of renaming unrelated files.

## Git and pull requests

- Before creating a branch, confirm the base with the user, fetch its latest tip, and branch before editing. Use `<type>/<short_snake_case_desc>`; the default base is `main`.
- Commit and push only when explicitly requested. Do not amend or rewrite existing commits without authorization.
- Before every commit, format all Python and JS/TS source files included in that commit, including staged changes from earlier work. Run `uv run ruff format <python_paths>` for Python. From `ui-tui/`, run `npx --no-install prettier --write <js_ts_paths>` using paths relative to that directory, including `../` for sources elsewhere in the repository.
- After formatting, run `uv run ruff format --check <python_paths>` and `npx --no-install prettier --check <js_ts_paths>` in the same respective directories, then the relevant lint and tests. Review and stage the formatting changes before committing; do not commit while these checks fail. Respect generated/vendor exclusions and do not format unrelated files.
- Use Conventional Commits: `<type>(<scope>): <subject>`. Scope is an `opendde_harness/` subpackage, or omitted for cross-package changes. Use English ASCII, a lowercase subject, no trailing period, and a header of at most 100 characters.
- Before pushing, fetch the target, check conflicts with `git merge-tree --write-tree HEAD origin/<target>`, rebase onto its latest tip if needed, and rerun relevant tests. Only use `--force-with-lease` on your own feature branch; never force-push protected branches.
- After pushing a new branch, offer to open a PR. Draft an English ASCII title and description covering the problem, changes, actual verification, risks, and related issues; check the text and show it for confirmation before creating the PR.
- PRs are squash-merged: put issue-closing keywords in the PR description, and keep the title plus GitHub's appended PR number within 100 characters.
- When Claude contributes code, include `Co-authored-by: Claude (<actual-model-id>) <noreply@anthropic.com>` in the commit; do not duplicate it in the PR description. Omit generated-by banners and local-only references.

## Handoff

- Review the task diff for unrelated changes and run `git diff --check -- <changed_paths>`.
- Report what changed, validation results, and concrete limitations. Keep this guide concise; avoid duplicate rules, tutorials, and example catalogs.
