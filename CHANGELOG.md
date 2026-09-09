# Changelog

User-facing changes to OpenDDE Harness are documented here.

## [Unreleased]

No changes yet.

## [0.0.2] - 2026-09-09

OpenDDE Harness 0.0.2 improves model configuration, context limits, streaming recovery,
and MCP connection handling. After upgrading, run `ddeharness onboard` again to reconfigure.

### Added

- Explicit protocol selection for OpenAI-compatible endpoints through `wire`: `chat` by
  default, or `responses` for the OpenAI provider. Configure it per provider or per model
  through `modelOverlay`. CLI and TUI configuration screens show the selected protocol.
- Per-model context window and output token limits through `modelOverlay`, with offline
  lookups from built-in data, provider-specific models.dev entries, LiteLLM, and canonical
  models.dev entries. Unknown context windows remain unknown; unknown output limits use
  an estimated 16384-token ceiling.
- `llm_first_token_timeout` and `llm_idle_timeout` to limit the initial wait and gaps
  between streaming events.
- `curator_timeout_seconds` to bound curator planning, with a deterministic fallback
  when planning times out.

### Fixed

- Corrected reasoning parameters for supported models served through relays, including
  DeepSeek, Z.ai, DashScope, OpenRouter, and OpenAI-compatible endpoints.
- Retryable streaming failures now support bounded retries after partial output. The TUI
  clears partial text before retrying, and multi-endpoint configurations can avoid failed
  endpoints when another endpoint is available.
- Codex subscription models now use dedicated built-in limits rather than API model entries:
  272k context and 128k output, with a 128k context window for the spark tier.
- Removed the 65536-token context fallback. History is no longer trimmed against an
  assumed window when the model's context limit is unknown.
- History trimming preserves tool results that belong to protected tool calls.
- Isolated MCP transport failures from the agent task. Tool calls remain subject to
  their configured timeout.
- Preserved reasoning identifiers when replaying Responses API history with tool calls.
- `doctor` and `onboard` recognize managed runtime code that can be prepared at first
  start from cached sources, avoiding unnecessary `compute prepare` prompts for that case.
- Moved synchronous history trimming and curator processing off the event loop to reduce
  terminal stalls.
- Improved protocol-mismatch errors and diagnostics for ambiguous 404 responses.

### Changed

- Updated the configuration format.
- `llm_call_timeout` now applies only to non-streamed calls.
- The curator uses the agent's model unless a separate model is configured.
- Raised the minimum LiteLLM version to 1.100.0.

## [0.0.1] - 2026-09-09

**Introducing OpenDDE Harness - our first public preview!**

We're excited to share OpenDDE Harness, an open-source tool for agentic antibody
design. It brings LLM-guided sequence design and OpenDDE structure prediction into
one workflow, from preparing a target to reviewing candidate sequences and structures.

Describe your design task in natural language, review the plan, and follow the
results as the agent proposes, evaluates, and refines candidates. This first release
supports VHH, scFv, and paired VH/VL binders while preserving configured fixed residues.
A terminal UI guides setup and design, and a tracing dashboard lets you inspect
candidate structures, scores, and progress across iterations.

Alongside this release, we're sharing a
[technical report](https://github.com/aurekaresearch/OpenDDE-Harness/blob/v0.0.1/docs/assets/OpenDDE_harness_tech_report.pdf)
and [design examples](https://github.com/aurekaresearch/OpenDDE-Harness/tree/v0.0.1/docs/examples)
to introduce the approach and help you get started. Install from PyPI or source
using the [installation guide](https://github.com/aurekaresearch/OpenDDE-Harness/blob/v0.0.1/README.md#installation),
then follow the [onboarding guide](https://github.com/aurekaresearch/OpenDDE-Harness/blob/v0.0.1/docs/onboarding.md)
to connect your LLM provider and compute service.

This is an early preview, and you may encounter bugs. Please
[open an issue](https://github.com/aurekaresearch/OpenDDE-Harness/issues) with your
feedback, reproduction steps, and `ddeharness doctor --json` output when relevant.
Thank you for trying OpenDDE Harness and helping us make it better!

[Unreleased]: https://github.com/aurekaresearch/OpenDDE-Harness/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/aurekaresearch/OpenDDE-Harness/releases/tag/v0.0.2
[0.0.1]: https://github.com/aurekaresearch/OpenDDE-Harness/releases/tag/v0.0.1
