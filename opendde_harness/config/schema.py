"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings, SettingsConfigDict

from opendde_harness.config.features import (
    ContextConfig,
    MemoryConfig,
    PluginsConfig,
    RuntimeConfig,
    SkillForgeConfig,
    TracingConfig,
)


class Base(BaseModel):
    """Accepts both camelCase and snake_case keys; rejects keys it does not know.

    A key this release does not define is a typo or one an earlier release
    wrote, and either is an error naming the key, never a value silently
    dropped. Nothing is migrated: the config is the current schema.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


#: pi's thinking-level vocabulary (``ModelThinkingLevel``). Each family is sent
#: the nearest level it takes: OpenAI's effort ladder (``max`` on gpt-5.6 and
#: gpt-6-astra, ``xhigh`` from gpt-5.2), DeepSeek's three steps, a token
#: budget for DashScope, on/off for Z.ai. ``off`` switches thinking off.
ReasoningEffort = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.opendde_harness/workspace"
    model: str = "anthropic/claude-opus-4-5"
    provider: str = "auto"  # Provider name (e.g. "anthropic", "openrouter") or "auto" for auto-detection
    # No maxTokens and no contextWindowTokens here on purpose. A number in a
    # config file cannot be right for every model -- too large is a 400, too
    # small truncates or trims silently -- so both are resolved per model from
    # the model's own overlay and the bundled tables (providers/rates). The
    # place to declare either is `providers.<name>.modelOverlay.<id>`.
    temperature: float = 0.1
    # Wall-clock cap (seconds) on a non-streamed LLM request (main loop and
    # sub-agents). A streamed request is bounded per silence instead: the
    # wait for its first event, then the gap between events. A stalled gateway
    # then surfaces after the idle bound, not after ten minutes, and a reply
    # that keeps arriving is never cut off for being long.
    llm_call_timeout: int = 600
    llm_first_token_timeout: int = 300
    llm_idle_timeout: int = 120
    max_tool_iterations: int = 40
    # Cap on subagent VMs running at once (excess spawns queue). ge=1: a
    # 0/negative cap would deadlock every subagent (Semaphore(0)).
    max_concurrent_subagents: int = Field(default=4, ge=1)
    # Spawn rate limit per session, per rolling hour — the concurrency gate
    # alone can't stop a prompt-injected agent from spawning indefinitely (each
    # finishes, freeing a slot for the next; the cross-turn re-injection loop
    # needs no user input). A rolling window bounds a runaway to N/hour yet
    # auto-recovers, so it never permanently locks out heavy legitimate use.
    # Counted per session so one busy session can't throttle others.
    max_subagent_spawns_per_hour: int = Field(default=30, ge=1)
    # Empty-response recovery: recover turns the model ends with no visible text
    # (post-tool empty / thinking-only) instead of surfacing a dud "no response
    # to give". Budgets are per-turn.
    empty_recovery_enabled: bool = True
    post_tool_empty_max_nudges: int = 1
    thinking_prefill_max_retries: int = 2
    empty_content_max_retries: int = 3
    # How hard every model thinks unless its own overlay says otherwise.
    # pi's DEFAULT_THINKING_LEVEL; None leaves each vendor's default in place.
    # A model the catalogue marks as non-reasoning is sent nothing.
    reasoning_effort: ReasoningEffort | None = "medium"
    # Per-model request-parameter overrides, keyed by a substring of the model
    # name: {"kimi-k2.5": {"temperature": 1.0}}. Some models reject the usual
    # defaults, and hard-coding those quirks in the registry left users unable to
    # adjust them. Entries here win over the registry's built-in defaults.
    # This is also the direct channel for arbitrary sampling/serving params: an
    # unknown top-level key is auto-forwarded into extra_body by LiteLLM for
    # OpenAI-compatible backends (e.g. sglang's repetition_penalty); a nested
    # structure can be written directly as extra_body: {...}.
    model_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ModelOverlay(Base):
    """What the user knows about a model that no catalogue carries.

    A self-hosted deployment serves whatever was put there, and a model released
    since the bundled snapshot is in no table yet, so the picker falls back to
    showing the id and the loop to an unknown window. The id is usually a fine
    name -- it is what the user called their own deployment -- but nothing else
    about such a model can be looked up, and the user is the only source.

    ``context_window_tokens`` and ``max_output_tokens`` are read first by the
    resolvers in ``providers.rates``, ahead of every table: a person describing
    their own deployment is the authority on it. Per model, and only per
    model: one number for every model the session switches to is wrong for
    all but one of them. A deployment configured with a smaller window than
    the model's native one is exactly the case for this: the catalogue's
    figure would be an over-estimate, and an over-estimate is a refused
    request rather than wasted context.

    ``wire`` picks the wire for this one model, over the section's
    ``wire``: one relay can serve different models on different wires
    (models.dev carries 289 such per-model overrides), so a provider-wide
    setting cannot always be right.

    What has no knob is a *price* for an endpoint no catalogue prices; such a
    deployment reports unknown spend rather than borrowing a hosted model's
    rate. Adding one is a separate ask.
    """

    label: str = ""
    description: str = ""
    context_window_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    wire: Literal["responses", "chat"] | None = None
    # This model's thinking level, over ``agents.defaults.reasoningEffort``:
    # a hybrid model one wants fast and a reasoning model one wants deep can
    # share a session without the global default being wrong for one.
    reasoning_effort: ReasoningEffort | None = None


class ProviderEndpoint(Base):
    """One named URL/key group under a provider section.

    ``label`` is not decoration: it is the idempotency key a later stage
    (rotation, failover, per-endpoint health) uses to address one entry across
    edits, so two endpoints in the same list must not share one.
    """

    label: str = Field(min_length=1)
    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None
    # OpenAI-compatible endpoints have two wire protocols: "chat" is
    # POST /v1/chat/completions, "responses" is POST /v1/responses. None means
    # the provider's own default (`ProviderSpec.wire`): Responses for
    # OpenAI proper, Chat Completions for a relay or self-hosted endpoint,
    # which is what most of them implement. Wrong wire is reported as a
    # configuration error naming this field, never routed around silently.
    wire: Literal["responses", "chat"] | None = None
    # Custom headers (e.g. APP-Code for AiHubMix) -- can carry a secret, so
    # display faces redact the values (keys stay visible).
    extra_headers: dict[str, str] | None = Field(default=None, json_schema_extra={"secret": True})
    models: list[str] = Field(default_factory=list)  # User-curated model names for the picker
    # Several full url/key/header groups under one provider section, for a
    # vendor reachable by more than one account or region. Meaningful only for
    # a plain API-key provider reached through the litellm client -- a section
    # whose auth is OAuth, or that needs more than a key and an address (Azure
    # OpenAI, Codex), gets this rejected at `make_provider` construction time
    # (wired in a later stage; this field exists regardless). Set and non-empty,
    # it replaces the flat `api_key` outright rather than merging with it; an
    # entry inherits the flat `api_base`/`extra_headers` for whichever it does
    # not name itself -- see `opendde_harness.providers.endpoints.provider_endpoints` for the
    # one place that resolves which of the two shapes (or Gemini's
    # `api_key_list`) is in effect.
    endpoints: list[ProviderEndpoint] = Field(default_factory=list)

    @field_validator("endpoints")
    @classmethod
    def _unique_endpoint_labels(cls, value: list[ProviderEndpoint]) -> list[ProviderEndpoint]:
        """Reject a duplicate label -- see the class docstring for why one must be unique."""
        seen: set[str] = set()
        for ep in value:
            if ep.label in seen:
                raise ValueError(f"duplicate endpoint label {ep.label!r}: labels must be unique within a provider")
            seen.add(ep.label)
        return value

    # How requests spread across `endpoints` when there is more than one:
    # "sticky" keeps using the first healthy entry until it fails, "round_robin"
    # cycles through all of them. Meaningless with zero or one endpoint.
    endpoint_strategy: Literal["sticky", "round_robin"] = "sticky"
    # Keyed by model id, in any spelling: what the user knows about a model that
    # the catalogues do not. Deliberately additive rather than a change to
    # `models` -- that list already lets a model be added, and what was missing
    # was a way to describe one, so no config has to be rewritten to get it.
    model_overlay: dict[str, ModelOverlay] = Field(default_factory=dict)

    @property
    def effective_api_key(self) -> str:
        """The key to send, which is not always the ``api_key`` field.

        Declared on the base so every call site can ask without knowing which
        providers keep their key somewhere else. Gemini accepts a list, and a
        section holding only that list handed LiteLLM an empty string: the
        request left with no credential and failed at the API, having passed
        every check that only asked whether credentials existed.
        """
        return self.api_key


class AzureProviderConfig(ProviderConfig):
    """Azure OpenAI, whose connection needs more than a key and an address.

    A deployment is a name the tenant gives one model, and it goes into the
    request URL's path. It used to be read off ``agents.defaults.model``, which
    made a model id double as a connection parameter: the id could carry no
    prefix without the prefix landing in the path, so Azure was the one provider
    whose ids had to be spelled differently from everyone else's. Declared here,
    the model id is free to be a model id.

    ``api_version`` was hardcoded in the client, so a tenant on a different one
    had no way to say so.
    """

    deployment: str = ""  # falls back to the model id, for configs written before this field
    api_version: str = "2024-10-21"


class GeminiProviderConfig(ProviderConfig):
    """Gemini, which accepts several keys under one section.

    Example:
        gemini:
          apiKeyList:
            - "key1"
            - "key2"

    A ``vertex`` flag used to sit here, documented as setting
    ``GOOGLE_GENAI_USE_VERTEXAI``. Nothing read it, and it could not have worked:
    that variable belongs to the google-genai SDK, while requests go through
    LiteLLM, which does not read it and reaches Vertex as a separate provider
    (``vertex_ai``) needing ``VERTEXAI_PROJECT`` and ``VERTEXAI_LOCATION``. It was
    settable from the CLI and covered by tests, so it read as a supported feature
    while doing nothing at all. Reaching Vertex is a change to how a request is
    routed, not a boolean on a key.
    """

    #: Several keys may be listed; the first is used. Round-robin rotation was
    #: declared here once and never called -- listing keys and silently using one
    #: is the honest description of what happens.
    api_key_list: list[str] = Field(default_factory=list)

    @property
    def effective_api_key(self) -> str:
        if self.api_key_list:
            return self.api_key_list[0]
        return self.api_key

    @property
    def all_keys(self) -> list[str]:
        """Return all configured API keys."""
        if self.api_key_list:
            return list(self.api_key_list)
        return [self.api_key] if self.api_key else []


def _prefer_set_values(base: dict[str, Any], winner: dict[str, Any]) -> dict[str, Any]:
    """Merge two sections for one provider, letting a set value beat an unset one.

    The current name wins a genuine conflict, but a declared field exists as an
    empty section whether or not it was configured -- so taking it verbatim let a
    placeholder erase the credential the user had written under the provider's
    other spelling.
    """
    merged = dict(base)
    merged.update({k: v for k, v in winner.items() if v not in ("", None, [], {})})
    return merged


def _has_credentials(config: "ProviderConfig", spec: Any, name: str = "") -> bool:
    """Is this section actually usable, or just a placeholder?

    Every declared provider exists as an empty section whether or not the user
    configured it, so "the field is there" says nothing. A spec flag must not
    stand in for evidence either: `is_local` used to answer with no api_base at
    all, and an empty declared section then beat the credentials the user had
    really written under one of that provider's other names.

    The rule itself lives in `providers.auth`, because deciding it here as well
    is what made a Gemini section holding only `api_key_list` invisible to
    routing while `provider list` showed it as configured.

    A vendor OpenDDE Harness carries no spec for reaches this too -- the passthrough route,
    where the section name is all there is -- so the name is passed separately
    rather than read off a spec that may not exist.
    """
    from opendde_harness.providers.auth import credential_status

    return credential_status(name or (spec.name if spec else ""), config, spec=spec).ok


class ProvidersConfig(Base):
    """Configuration for LLM providers.

    Fields below are the providers OpenDDE Harness carries metadata for. Any other key is
    kept as-is and served through :meth:`get`, so a provider LiteLLM supports but
    OpenDDE Harness has no spec for still works from config alone.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _merge_renamed_sections(cls, data: Any) -> Any:
        """Fold a provider's pre-rename section into its current one.

        A file touched by both names holds two half-filled sections -- say the
        credentials under the old name and a model list under the new one.
        Picking either section alone drops the other's fields, so merge with the
        current name winning per field.
        """
        if not isinstance(data, dict):
            return data
        from opendde_harness.providers.registry import PROVIDERS, names_same_provider

        merged = dict(data)

        # Fold any key that spells a declared field differently into that field.
        # Extras are matched spelling-insensitively (`ProvidersConfig.get`), and
        # a declared field exists as an empty section whether or not it was
        # configured -- so without this, "azure-openai" or "OpenRouter" lands in
        # extras where the always-present empty field then wins, and a key the
        # user really wrote reads back as unset. One rule for both kinds.
        for key in [k for k in merged if k not in cls.model_fields]:
            field = next((f for f in cls.model_fields if names_same_provider(key, f)), None)
            if field is None or not isinstance(merged[key], dict):
                continue
            section = dict(merged.pop(key))
            current = merged.get(field)
            if isinstance(current, dict):
                section = _prefer_set_values(section, current)
            merged[field] = section

        for spec in PROVIDERS:
            stale = [merged.pop(a) for a in spec.name_aliases if isinstance(merged.get(a), dict)]
            if not stale:
                continue
            section: dict[str, Any] = {}
            for older in stale:
                section = _prefer_set_values(section, older)
            current = merged.get(spec.name)
            if isinstance(current, dict):
                section = _prefer_set_values(section, current)
            merged[spec.name] = section
        return merged

    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    azure_openai: AzureProviderConfig = Field(default_factory=AzureProviderConfig)  # Azure OpenAI
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    # Z.ai, the vendor's current brand and LiteLLM's name for it. Configs
    # written before the rename say "zhipu"; both keys load.
    zai: ProviderConfig = Field(
        default_factory=ProviderConfig,
        validation_alias=AliasChoices("zai", "zhipu"),
    )
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)  # Alibaba Cloud Tongyi Qianwen
    # LiteLLM's own names for these two, so a model id and a config section are
    # spelled the same. Configs written before the rename keep loading.
    hosted_vllm: ProviderConfig = Field(
        default_factory=ProviderConfig,
        validation_alias=AliasChoices("hosted_vllm", "hostedVllm", "vllm"),
    )
    gemini: GeminiProviderConfig = Field(default_factory=GeminiProviderConfig)  # Google Gemini / Vertex AI
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax_global: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax_cn: ProviderConfig = Field(default_factory=ProviderConfig)
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API gateway
    ollama_chat: ProviderConfig = Field(
        default_factory=ProviderConfig,
        validation_alias=AliasChoices("ollama_chat", "ollamaChat", "ollama"),
    )
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)  # SiliconFlow
    volcengine: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine
    openai_codex: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenAI Codex (OAuth)
    github_copilot: ProviderConfig = Field(default_factory=ProviderConfig)  # Github Copilot (OAuth)

    def get(self, name: str) -> ProviderConfig | None:
        """Return one provider's config, declared field or extra key alike.

        The lookup is spelling-insensitive on both sides. A section key reaches
        here in whichever form its writer used -- LiteLLM's hyphenated vendor
        name, the camelCase this model serializes to, or the underscored field
        name -- and a caller holding a model-id prefix has only one of those. So
        this is the only place a provider name may be resolved to its config;
        reading the attribute directly sees just the one spelling.
        """
        from opendde_harness.providers.registry import canonical_provider_name, names_same_provider

        # A renamed provider keeps answering to its old name, and the declared
        # field wins: a half-migrated config holding both keys must not serve
        # the stale one.
        name = canonical_provider_name(name)
        declared = self.__dict__.get(name)
        if isinstance(declared, ProviderConfig):
            return declared
        extra = (self.model_extra or {}).get(name)
        if extra is None:
            for key, value in (self.model_extra or {}).items():
                if names_same_provider(key, name):
                    extra = value
                    break
        if isinstance(extra, ProviderConfig):
            return extra
        if isinstance(extra, dict):
            return ProviderConfig.model_validate(extra)
        return None

    def model_overlays(self) -> dict[str, ModelOverlay]:
        """Every section's overlays in one map, keyed by ``wire.merge_key``.

        Keyed by identity rather than by the string the user typed, so an
        overlay written against a bare id still matches the qualified id the
        loop runs on. One map for the loop to carry: it switches models at
        runtime and has to find the new model's declaration without holding
        the whole config.
        """
        from opendde_harness.providers.wire import merge_key

        out: dict[str, ModelOverlay] = {}
        sections: dict[str, Any] = {**self.__dict__, **(self.model_extra or {})}
        for name, section in sections.items():
            if isinstance(section, dict):
                try:
                    section = ProviderConfig.model_validate(section)
                except Exception:
                    continue
            overlays = getattr(section, "model_overlay", None) or {}
            for model, overlay in overlays.items():
                out[merge_key(name, model)] = overlay
        return out


class WebSearchConfig(Base):
    """Web search tool configuration."""

    api_key: str = ""  # Serper API key
    max_results: int = 5


class WebToolsConfig(Base):
    """Web tools configuration."""

    proxy: str | None = None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    jina_api_key: str = ""  # Jina Reader API key
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    timeout: int = 60
    path_append: str = ""
    # Extra regex deny-patterns appended to ExecTool's built-in destructive-command
    # defaults. Empty by default. Operators (or eval harnesses running the agent
    # un-sandboxed) can add host-specific blocks, e.g. osascript / `open -a`.
    extra_deny_patterns: list[str] = Field(default_factory=list)


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    tool_timeout: int = 30  # seconds before a tool call is cancelled


class ToolSearchConfig(Base):
    """Progressive tool disclosure.

    When the live tool catalog (built-ins + plugins + MCP) grows past
    ``compaction_threshold``, most tool schemas are withheld from each request and reached
    on demand through the ``tool_search`` / ``tool_call`` meta-tools, so context
    cost stops scaling with tool count and the per-turn tool list (and thus the
    prompt cache) stays stable. At or below the threshold every tool is exposed
    directly (unchanged behavior) and the meta-tools are omitted.
    """

    enabled: bool = False
    compaction_threshold: int = 50
    """Tool-catalog size that triggers compaction: at or below this many tools
    everything is exposed directly; above it, schemas are withheld."""
    search_result_limit: int = 10
    """Default number of hits ``tool_search`` returns per query."""
    always_visible: list[str] = Field(default_factory=list)
    """Extra tool names kept exposed every turn, on top of the core set."""


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    tool_search: ToolSearchConfig = Field(default_factory=ToolSearchConfig)
    disabled_tools: list[str] = Field(default_factory=list)
    """Tool names to unregister after default-tool registration and MCP connect.
    Used by eval harnesses (e.g. BrowseComp-Plus) that need to constrain the
    agent to a specific tool subset. Names match those in ``ToolRegistry``
    (e.g. ``read_file``, ``web_search``, or ``mcp_bcp-search_search``)."""


class CliConfig(Base):
    """CLI surface configuration."""

    turn_summary: bool = True
    """Render a one-line tokens/cost summary after each successful CLI turn."""


class Config(BaseSettings):
    """Root configuration for opendde_harness: the base agent blocks plus the
    feature blocks (:mod:`opendde_harness.config.features`), all parsed from
    the one ``config.json``."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    cli: CliConfig = Field(default_factory=CliConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    # UI language chosen during onboarding. Drives the wizard/CLI copy and the
    # agent's reply language (injected into the system prompt). "en" | "zh".
    language: Literal["en", "zh"] = "en"

    context: ContextConfig = Field(default_factory=ContextConfig)
    # SkillForge subsystem; its RRF routing policy nests at ``skill_forge.router``.
    skill_forge: SkillForgeConfig = Field(default_factory=SkillForgeConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(self, model: str | None = None) -> tuple["ProviderConfig | None", str | None]:
        """Match provider config and its registry name. Returns (config, spec_name)."""
        from opendde_harness.providers.registry import (
            PROVIDERS,
            canonical_provider_name,
            find_by_keywords,
            find_by_name,
            split_model_id,
        )

        forced = self.agents.defaults.provider
        if forced != "auto":
            # Return the canonical name: callers look the spec up by it, and a
            # config still naming the provider the old way would find nothing.
            forced = canonical_provider_name(forced)
            p = self.providers.get(forced)
            return (p, forced) if p else (None, None)

        model_id = model or self.agents.defaults.model
        prefix, _ = split_model_id(model_id)

        # `spec.claims` is the whole prefix-beats-keyword rule: a prefixed id is
        # answered only by the provider it names (so `github-copilot/...codex`
        # cannot match openai_codex, and no vendor's key is posted to another's
        # endpoint), while a bare id falls to keywords in registry order.
        for spec in PROVIDERS:
            if not spec.claims(model_id):
                continue
            p = self.providers.get(spec.name)
            if p and _has_credentials(p, spec):
                return p, spec.name

        # Explicit prefix naming a provider OpenDDE Harness has no spec for: LiteLLM knows
        # the vendor, so credentials under that name are enough to reach it.
        #
        # Only where there is genuinely no spec. A provider that has one has
        # already been offered above and turned down for want of credentials --
        # letting it back in here on `api_key` alone reinstated exactly the
        # material this rejected it for missing: Azure with a key and no address
        # routed here, while display and startup both called it unconfigured.
        if prefix and find_by_name(prefix) is None:
            passthrough = self.providers.get(prefix)
            if passthrough and _has_credentials(passthrough, None, prefix):
                return passthrough, canonical_provider_name(prefix)

        # Fallback: gateways first, then others (follows registry order).
        # OAuth providers are NOT valid fallbacks -- they require explicit model
        # selection.
        #
        # Once an id names a vendor -- by prefix, or by a keyword that only one
        # vendor answers to -- reaching this point means that vendor has no
        # credentials. Only a gateway or a local deployment may answer then,
        # because they route whatever they are handed; a direct vendor would be
        # receiving a competitor's model id along with its own key. Getting here
        # having named nobody ("llama-3.3-70b") carries no such claim, so any
        # credentialed provider is a legitimate guess.
        names_a_vendor = bool(prefix) or find_by_keywords(model_id) is not None
        for spec in PROVIDERS:
            # A prefix is the user naming a provider. A local deployment is not
            # that provider, so "anthropic/claude-sonnet-5" is a misroute rather
            # than a fallback -- while a bare "qwen3-32b", which only matches a
            # vendor keyword, is exactly what a local box is likely serving.
            if spec.is_local and prefix:
                continue
            if names_a_vendor and not (spec.is_gateway or spec.is_local):
                continue
            if spec.is_oauth:
                continue
            # A gateway with no keywords answers to nothing on its own: it is
            # reached by naming it, as a prefix or as agents.defaults.provider.
            # Left in this loop it stood first in registry order and quietly
            # took every model whose own vendor had no credentials, so the
            # picker named one provider while the prompt and a different key
            # went to an endpoint the user never chose for that model. The
            # local deployments below are the same case: they route whatever
            # they are handed, which is only a legitimate guess for an id that
            # names nobody.
            if spec.is_gateway and not spec.keywords:
                continue
            p = self.providers.get(spec.name)
            if p and _has_credentials(p, spec):
                return p, spec.name
        return None, None

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base, extra_headers). Falls back to first available."""
        p, _ = self._match_provider(model)
        return p

    def get_provider_name(self, model: str | None = None) -> str | None:
        """Get the registry name of the matched provider (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get API key for the given model. Falls back to first available key."""
        p = self.get_provider(model)
        return p.effective_api_key if p else None

    def get_api_base(self, model: str | None = None) -> str | None:
        """Get API base URL for the given model. Applies default URLs for gateway/local providers."""
        from opendde_harness.providers.registry import find_by_name

        p, name = self._match_provider(model)
        if p and p.api_base:
            return p.api_base
        # Only gateways get a default api_base here. Standard providers
        # (like Moonshot) set their base URL via the spec's env_extras.
        if name:
            spec = find_by_name(name)
            if spec and spec.usable_default_api_base:
                return spec.usable_default_api_base
        return None

    # The file spells the feature blocks camelCase (``skillForge``); both
    # spellings load, and dumps use the camelCase alias like every nested block.
    model_config = SettingsConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        env_prefix="OPENDDE_HARNESS_",
        env_nested_delimiter="__",
        extra="forbid",
    )


# Written to a fresh file by the onboarding wizard (``config.update``) and
# patched in place afterwards; ``save_config`` leaves them out so a bootstrap
# file stays the shape the wizard seeds.
FEATURE_FIELDS = frozenset({"context", "skill_forge", "runtime", "tracing", "plugins", "memory"})
