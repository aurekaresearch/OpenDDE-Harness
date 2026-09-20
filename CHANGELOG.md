# Changelog

User-facing changes to OpenDDE Harness are documented here.

## [Unreleased]

### Added

- Opt-in PyRosetta FastRelax and InterfaceAnalyzer in isolated, bounded CPU
  subprocesses, with eight interface/energy metrics, relaxed structures, residue
  evidence, and runtime provenance. Requires a separately licensed installation
  in the compute service; the default image remains unchanged.
- Fixed-anchor, bounded grouped loss with explicit ranges, budgets and an audited
  breakdown. The reusable `default-bounded-v1` policy is opt-in and bundled with
  source and wheel distributions; existing linear scoring remains available.
- Raw Min ipAE and ipSAE loss terms, with validated preference directions and
  explicit availability metadata. Uncomputable ipSAE uses a recorded zero fallback;
  other missing required metrics still fail scoring.
- Configurable terminal-refolding parent/sample/survivor limits, and independent
  Age/Cycle colouring in the dashboard with a remembered browser preference.

### Changed

- Terminal selection now enforces hard contact/quality eligibility and the same
  objective ordering as search. LLM ranking is advisory; terminal-stage failure
  cannot be reported as a successfully completed workflow. This is an intentional
  behavior change for newly executed terminal stages, including linear-loss runs.
- Quality assessment uses current candidate sequences and configured masks;
  contradictory High Risk passing verdicts are rejected.
- Dashboard scales, filters and loss details preserve signed values, precision,
  missing data and configured objective semantics. Dashboard tests run in CI.

### Fixed

- Detached workers load the explicitly selected application configuration.
  Managed compute verifies the requested image before reuse and refuses to
  replace a busy worker when code/image settings change.
- Raw interchain PAE mapping, nested loss trace preservation, non-interactive
  Node discovery, and memory tool-call serialization.
- Clean development installs include Biotite for structural-scoring unit tests;
  the licensed PyRosetta dependency remains optional.

### Upgrade notes

Follow [installation](docs/installation.md#enable-pyrosetta-and-bounded-scoring).
Preserve accepted YAML and existing results; use a new configuration/task to opt
into bounded scoring. Let active work finish before changing worker code, images
or dependencies. A dashboard-only static asset update needs a browser reload,
not a compute restart. No scientific results are migrated or rescored on upgrade.

## [0.0.4] - 2026-09-13

OpenDDE Harness 0.0.4 introduces a pi-tui terminal interface, a shared pi-ai
model service, and simpler provider and memory setup.

### Upgrading

Run `ddeharness onboard` after upgrading from 0.0.3, before starting the TUI.
This release changes the configuration format, provider credentials, and memory
settings. Incompatible configuration is rejected with instructions to reconfigure;
it is not migrated automatically.

Onboarding moves incompatible configuration and related compute and memory service
settings into `backup-<timestamp>/`, then creates a new setup. Weights, sessions,
and memory notes are preserved. Sign in to Codex again through the new model
service; MiniMax now uses an API key instead of device sign-in.

Subsequent onboarding runs preserve configuration with a matching `schemaVersion`
and prefill each step with existing values. Use `ddeharness onboard --fresh` to
back up the current configuration and start over.

### Highlights

- Replaced the Ink/React terminal UI with pi-tui for more responsive streaming,
  long transcripts, and window resizing.
- Unified model access through a Node-based pi-ai service shared by the Python
  agent, provider login, and long-term memory.
- Added model discovery for declared OpenAI-compatible endpoints and pi-style
  model selection with saved scopes.
- Simplified onboarding with a shared login flow, compact choices, and clearer
  download and service-start progress.

### Changed

#### Providers and models

- Provider configuration now follows pi-ai's `models.json` structure. Built-in
  providers use their canonical pi IDs; custom providers must declare both
  `baseUrl` and `api`. The protocol is explicit and is never inferred from the URL.
- Renamed provider and model settings: `wire` to `api`, `extraHeaders` to
  `headers`, `apiBase` to `baseUrl`, and `modelOverlay` to entries in `models`.
  Model fields `label`, `contextWindowTokens`, and `maxOutputTokens` are now
  `name`, `contextWindow`, and `maxTokens`. Corresponding CLI flags are
  `--api`, `--headers`, and `--base-url`.
- Credentials use `apiKey` or `"login": "oauth"`. OAuth credentials live in the
  model service's credential store. Providers can also use their standard
  environment variables; endpoints that require no authentication need no key.
  Codex sign-in is available through `ddeharness provider login openai-codex`.
  MiniMax uses `providers.minimax.apiKey` or `providers.minimax-cn.apiKey`.
- Model IDs must include a provider, such as `openai-codex/<model>` or
  `my-vllm/qwen3-32b`. Removed `agents.defaults.provider`; routing and model
  limits now use the same qualified ID. Bare IDs are rejected with instructions
  to correct them.
- `/model` lists models from connected providers, with scoped/all views and
  current/default indicators. `/scoped-models` saves the selection in
  `agents.scopedModels`. In the TUI, connect providers through `/login`; the
  model picker's add-model, remove-model, and disconnect actions are removed.
- Declared OpenAI-compatible endpoints discover models through `/models`.
  Results are cached across restarts and refreshed when the picker opens if the
  endpoint's catalog is at least 24 hours old. Newly declared endpoints are
  queried immediately.
- Model limits, input modalities, and pricing come from the model service's
  catalogs and endpoint metadata. Configure model entries through
  `providers.<id>.models` or `ddeharness provider model set`. Unknown limits
  remain unknown rather than receiving an arbitrary fallback.
- `ddeharness provider list` shows configured providers.
  `ddeharness provider reset <id>` removes the entry and signs out its OAuth
  credential when applicable.
- Removed the legacy Python provider routes, bundled catalog tables, and custom
  OAuth flows. Azure OpenAI uses `azure-openai-responses` with `api-version=v1`;
  Chat Completions-only deployments can use `openai-completions` with the
  deployment URL as `baseUrl`.
- Web search now uses the shared `web_search` tool for every provider: Brave when
  `tools.web.braveApiKey` is configured, otherwise DuckDuckGo. Provider-hosted
  search and its citations are removed. Reasoning embedded in response content
  as `<think>` tags is no longer stripped.
- Temperature is sent only when configured on the model entry, including in
  protein-design tasks. Requests carry the session's cache key for prompt-cache
  reuse across the conversation.
- Transient assistant failures retry with backoff and a visible `turn.retry`
  event, up to `agents.defaults.llmRetries` (default 3). Partial text from a
  failed attempt is replaced on retry; deterministic failures stop immediately.
  Removed `fallbackModels`. Pre-response failures use the turn's retry budget;
  `Retry-After` is no longer honored before the first byte.
- Sessions whose saved model is unavailable use the default model and display
  a one-time notice on the next turn, with instructions to select a replacement
  through `/model <provider>/<model>`.

#### Memory and context

- Upgraded the memory library to 1.3.1. Long-term memory uses the conversation's
  default model through the shared model service, including Codex sign-in.
  Separate memory model, key, and endpoint settings are no longer needed.
- Removed embedding and reranking roles; memory recall uses keywords. Memory
  configuration and service management now live in the memory plugin, and
  onboarding only asks whether to enable memory.
- The memory home is fixed at `~/.opendde_harness/memory`; previously configured
  roots are ignored. The host's memory writer again records episodes and the
  user profile.
- History selection is deterministic. Removed the Curator's model-based planning
  loop and its `context.curatorModel` and `context.curatorProvider` settings.
- Skill selection no longer makes query-rewriting or filtering LLM calls.
  Catalogs with at most 40 eligible skills are listed in a stable order; larger
  catalogs select up to 5 skills using BM25. Skills with unmet `requires` are
  excluded, and bodies are loaded only through `use_skill`.
- Removed `skillForge.injectionMode` and the `rewrite*` and `llmGate*` settings,
  including `rewriteEnabled`, `llmGateEnabled`, and `llmGateModel`. Setting
  `skillForge.enabled=false` or `skillForge.router.enabled=false` now removes
  the skills block from the prompt.

#### Onboarding and protein design

- Onboarding and `/login` share the same provider authentication flow. The
  wizard offers featured providers and an OpenAI-compatible option; additional
  providers are available through `/login`. Advanced protocol configuration,
  including Azure and `--api`, remains available through `ddeharness provider set`.
- Onboarding uses the TUI's branding, palette, translated step labels, and
  compact layout. Short choices appear in a single row with Tab navigation.
  Protein-design setup groups compute settings, folding, and weights into
  sections, and the completion screen uses aligned summaries.
- Service startup displays the current operation and elapsed time, followed by
  one result line. Compute downloads share a progress display; per-file details
  and asset bookkeeping go to logs. Memory startup and recovery choices use the
  same compact layout.
- Protein-design setup defaults to local folding with weights in the compute
  container. Existing folding-mode choices are preserved.
- Protein-design tasks inherit the initiating conversation's model unless the
  task YAML overrides it. Workers no longer load code from the launcher's
  working directory.

#### Terminal UI and updates

- Tools use pi's names and schemas: `bash`, `read`, `write`, `edit`, `ls`,
  `grep`, and `find`.
- `/quiet-tools` collapses tool rows to one line; `Ctrl+O` expands up to 1 KiB
  of a result. `/tasks hide|show` controls the design task bar, `Ctrl+S` saves
  the default model in the picker, and `/tracing` opens the tracing dashboard.
  The busy indicator now shimmers.
- Slash commands report timeouts immediately while the underlying operation
  finishes in the background. The command gateway rejects further commands
  with "another command is still running" until it completes, and late output
  is kept out of the terminal.
- PyPI update notices appear in the TUI status bar, once in the launch
  transcript, and at the top of `ddeharness doctor`. Notices show the appropriate
  upgrade command (`uv tool upgrade` or `pip install --upgrade`); source
  checkouts skip update checks.

### Fixed

- Models discovered with only an ID no longer fail unconditionally with
  `no_max_tokens`. Discovery reads endpoint limit fields such as
  `context_length`, `max_model_len`, and `max_completion_tokens`, and fills
  missing limits from matching pi catalog entries. Supported OpenAI-compatible
  APIs omit an unknown output limit so the server can use its default; APIs
  that require a positive limit still reject unsized models.
- Compute startup errors distinguish an exited container from a service that is
  still unresponsive. Set `OPENDDE_HARNESS_COMPUTE_KEEP=1` to retain the next
  container for inspection with `docker logs`.
- Memory-service startup errors identify exhausted inotify instances and show
  the command to raise the limit alongside the original error.
- Fixed a `CancelledError` after saving a provider key during onboarding: leaving
  the login step no longer cancels the model service's reader.
- Corrected footer colors and prevented the estimated-token marker from
  resetting the rest of the line's color. The first footer line now shows only
  the working directory and Git branch.
- Kept the status area at two rows in both idle and active states, preventing
  layout jumps and separating replies from the editor.
- Added consistent spacing and dim colors to transcript notices, and unified
  keyboard-hint capitalization in `/model` and `/scoped-models`.

## [0.0.3] - 2026-09-10

OpenDDE Harness 0.0.3 adds per-model thinking controls, searchable model lists,
and clearer progress reporting in the terminal.

### Added

- Set per-model thinking levels with `/thinking <level>` (alias `/reasoning`) or
  `ddeharness provider model set <provider> <model> --reasoning-effort <level>`.
  Settings are saved in `modelOverlay` and adapted to the model's supported controls.
- Set the current model's context window with `/context 128k`, or clear the override
  with `/context default`.
- The `/model` picker discovers models from compatible relay and local endpoints,
  puts the current model first, and supports typing to filter providers, models, and endpoints.
- `Ctrl+T` toggles thinking blocks; `Ctrl+O` expands or collapses tool output.
  Both apply to live output and conversation history, and save the display preference.

### Changed

- The default thinking level is now `medium` unless overridden per model. Set
  `agents.defaults.reasoningEffort` to `null` to leave it to the provider;
  Codex subscriptions still default to `medium` and request reasoning summaries.
- Relay and self-hosted models can inherit catalog limits and reasoning support when
  their model name has a unique match. Use `modelOverlay` if the endpoint has lower limits.
- The activity line shows elapsed time, output tokens, retry progress, and how to interrupt.
- Model picker actions use `Ctrl+A` to add, `Ctrl+X` to delete, `Ctrl+E` for endpoints,
  and `Ctrl+D` to disconnect. `q` now filters the list; `Esc` clears the filter before leaving.

### Fixed

- Improved reasoning parameter handling for OpenAI-compatible endpoints, DeepSeek,
  Z.ai, DashScope, and OpenRouter.
- Output token counts now combine usage reported for completed calls with an estimate
  for the call still streaming, replacing that estimate when usage arrives.
- Reduced accidental removal of ordinary reasoning text when filtering echoed status messages.

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

The first public preview of OpenDDE Harness combines LLM-guided antibody sequence
design and OpenDDE structure prediction in one workflow, from target preparation
to candidate review.

Describe your design task in natural language, review the plan, and follow the
results as the agent proposes, evaluates, and refines candidates. This first release
supports VHH, scFv, and paired VH/VL binders while preserving configured fixed residues.
A terminal UI guides setup and design, and a tracing dashboard lets you inspect
candidate structures, scores, and progress across iterations.

This release includes a
[technical report](https://github.com/aurekaresearch/OpenDDE-Harness/blob/v0.0.1/docs/assets/OpenDDE_harness_tech_report.pdf)
and [design examples](https://github.com/aurekaresearch/OpenDDE-Harness/tree/v0.0.1/docs/examples)
to introduce the approach and help you get started. Install from PyPI or source
using the [installation guide](https://github.com/aurekaresearch/OpenDDE-Harness/blob/v0.0.1/README.md#installation),
then follow the [onboarding guide](https://github.com/aurekaresearch/OpenDDE-Harness/blob/v0.0.1/docs/onboarding.md)
to connect your LLM provider and compute service.

This is an early preview, and you may encounter bugs. Please
[open an issue](https://github.com/aurekaresearch/OpenDDE-Harness/issues) with your
feedback, reproduction steps, and `ddeharness doctor --json` output when relevant.

[Unreleased]: https://github.com/aurekaresearch/OpenDDE-Harness/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/aurekaresearch/OpenDDE-Harness/releases/tag/v0.0.4
[0.0.3]: https://github.com/aurekaresearch/OpenDDE-Harness/releases/tag/v0.0.3
[0.0.2]: https://github.com/aurekaresearch/OpenDDE-Harness/releases/tag/v0.0.2
[0.0.1]: https://github.com/aurekaresearch/OpenDDE-Harness/releases/tag/v0.0.1
