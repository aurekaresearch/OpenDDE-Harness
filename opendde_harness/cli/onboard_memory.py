"""Persistent Memory cluster of the onboard wizard (Step 2).

Split out of ``onboard_commands`` because that module had grown past 5000
lines; this file owns Persistent Memory role configuration (llm / embedding / rerank /
multimodal) end to end. Shared wizard UI state (``console``, ``_t``, ``_BACK``,
``_QMARK``, questionary helpers, ...) still lives in ``onboard_commands`` --
this module reaches it via the ``oc`` module reference (not a value import) so
that test monkeypatches on ``onboard_commands`` attributes keep working
whichever module a caller patches through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import typer

from opendde_harness.cli import onboard_commands as oc
from opendde_harness.plugin.memory.longterm._health import BACKEND_NAME, PLUGIN_ID
from opendde_harness.plugin.memory.longterm._library import CONFIG_FILENAME, SERVER_CMDLINE


def _set_memory_backend(backend: Optional[str]) -> None:
    """Set ``memory.backend`` (the bundled backend / ``None``) via the ops layer."""
    from opendde_harness.config.update import set_memory_backend

    set_memory_backend(backend)


def _init_extension_block_defaults() -> None:
    """Seed the memory / plugins / skillForge extension defaults via the ops layer."""
    from opendde_harness.config.update import init_extension_block_defaults

    init_extension_block_defaults()


def _memory_section(section: str) -> dict[str, Any]:
    from opendde_harness.config.update_memory import memory_section

    return memory_section(section)


def _memory_role_configured(section: str) -> bool:
    from opendde_harness.config.update_memory import memory_role_configured

    return memory_role_configured(section)


def _memory_enabled() -> bool:
    """True iff Persistent Memory is both selected AND usable on disk.

    "Usable" means the llm role is configured -- that is the whole requirement.
    embedding is advised but optional: without it the adapter searches lexically
    instead of semantically, which is weaker memory rather than none, and gating
    on it here would tell a user who skipped it that memory is off (and skip the
    import step along with it).
    """
    data = oc._load_raw_config()
    if (data.get("memory") or {}).get("backend") != BACKEND_NAME:
        return False
    slice_ = _recorded_memory_slice(data)
    if slice_.get("owned") is False:
        # A server the user runs. Its models live in a toml opendde promised not
        # to read, and no root is recorded for it, so the address is the whole
        # of what opendde can know -- liveness is the runtime's question.
        return bool(slice_.get("base_url"))
    return _memory_role_configured("llm")


def _recorded_memory_slice(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """``plugins.config["long-term-memory"]`` as raw JSON."""
    data = oc._load_raw_config() if data is None else data
    plugins = data.get("plugins") or {}
    slice_ = (plugins.get("config") or {}).get(PLUGIN_ID) if isinstance(plugins, dict) else None
    return slice_ if isinstance(slice_, dict) else {}


def _configured_target_url() -> str:
    """Where a opendde-managed server is meant to listen.

    A recorded intent rather than a constant. Without one, "the running address
    differs from the target" could not distinguish a port an earlier opendde left
    behind -- which should be converged -- from one the user chose, which
    should not, and the wizard moved both while calling the second a legacy
    port.
    """
    from opendde_harness.plugin.memory.longterm._server import DEFAULT_MEMORY_BASE_URL

    port = _recorded_memory_slice().get("port")
    if isinstance(port, int) and port > 0:
        return f"http://localhost:{port}"
    # Deliberately not falling back to the recorded address. That address is
    # where the service *is*; convergence compares the two, so reading one as
    # the other makes every pre-upgrade install look like it is already where
    # it belongs and the "keep it or move it" question never fires -- leaving
    # the upgrade quietly parked on the old port with the standard one never
    # mentioned. No intent recorded means the default is the target, and the
    # user gets asked.
    return DEFAULT_MEMORY_BASE_URL


# Providers whose main model can be reused as the Persistent Memory LLM: they
# speak the OpenAI chat-completions protocol that Persistent Memory's bare OpenAI client
# requires. OAuth providers (github_copilot / openai_codex) and non-OpenAI
# wire protocols (anthropic / gemini) are excluded.
_OPENAI_COMPATIBLE_PROVIDERS = {"openrouter", "openai", "deepseek", "custom"}

# Fallback OpenAI-compatible base URLs for providers whose registry
# ``default_api_base`` is empty (they rely on the SDK's built-in default,
# which Persistent Memory's bare client doesn't know). Persistent Memory needs an explicit base_url.
_PROVIDER_BASE_URL_FALLBACK = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _resolve_model_provider(model: str) -> Optional[str]:
    """Best-effort: which configured provider does ``model`` belong to?

    Prefixed models (``openrouter/...`` / ``openai/gpt-4o``) read off the head.
    A custom endpoint stores its model as a BARE id (e.g. ``qwen-max``) with no
    prefix, so an unrecognized head falls back to ``"custom"`` when a custom
    provider is actually configured with a key. Returns ``None`` when no match.
    """
    from opendde_harness.providers.registry import split_model_id

    if not model:
        return None
    head, _ = split_model_id(model)
    if head:
        from opendde_harness.config.update_providers import provider_field_specs

        try:
            provider_field_specs(head)
            return head
        except KeyError:
            pass
    # No usable prefix → could be a bare custom-endpoint model.
    from opendde_harness.config.schema import ProviderConfig
    from opendde_harness.providers.auth import credential_status

    custom = (oc._load_raw_config().get("providers") or {}).get("custom") or {}
    if credential_status("custom", ProviderConfig.model_validate(custom)).ok:
        return "custom"
    # A bare id that still matches a known provider head (rare; e.g. a direct
    # provider's bare default before prefixing) — accept the head if known.
    return head if head in _OPENAI_COMPATIBLE_PROVIDERS else None


def _model_is_openai_compatible(model: Optional[str]) -> bool:
    """Heuristic: can the main chat model's provider be reused for memory LLM?

    Persistent Memory's memory LLM uses a bare OpenAI client, so the main model is
    reusable only when its provider speaks the OpenAI chat protocol. Custom
    endpoints are OpenAI-compatible by definition (the wizard only offers
    ``custom`` for OpenAI-compatible endpoints).
    """
    if not model:
        return False
    return _resolve_model_provider(model) in _OPENAI_COMPATIBLE_PROVIDERS


def _resolve_reuse_llm_creds(main_model: str) -> dict[str, Optional[str]]:
    """Map a litellm-style main model to bare Persistent Memory LLM settings.

    Persistent Memory sends its ``[llm].model`` to ``base_url`` via a bare OpenAI
    client, so:
      - strip the provider's litellm prefix to the bare model id the upstream
        endpoint expects (``openrouter/anthropic/claude-x`` → ``anthropic/claude-x``;
        a custom endpoint's bare id is used as-is);
      - resolve the provider's real ``base_url`` (configured ``apiBase`` →
        registry ``default_api_base`` → a known fallback);
      - carry the provider's stored api_key.
    """
    from opendde_harness.providers.registry import find_by_name, normalize_provider_name, split_model_id

    provider = _resolve_model_provider(main_model) or split_model_id(main_model)[0]
    spec = find_by_name(provider)
    # Through the ops library, so a section still stored under the provider's
    # pre-rename name is found -- a raw lookup by the resolved name is not.
    from opendde_harness.config.update_providers import get_provider_config

    # No `if spec` gate: LiteLLM-only vendors have no spec of ours yet their
    # section holds real credentials, and gating on the spec silently handed the
    # probe an empty api_key while the main model was working fine.
    try:
        _resolved = get_provider_config(provider, redact_secrets=False)
    except KeyError:
        _resolved = {}
    prov_cfg = {"apiKey": _resolved.get("api_key"), "apiBase": _resolved.get("api_base")} if _resolved else {}

    # Strip the routing prefix to the bare model id the upstream endpoint
    # expects: litellm consumes it, the raw OpenAI client must not see it. Only
    # a prefix naming this provider is stripped -- a custom endpoint stores a
    # bare id already, and anything else is part of the vendor's own model id.
    bare_model = main_model
    head, rest = split_model_id(main_model)
    known_prefixes = set(spec.route_names) if spec else {normalize_provider_name(provider)}
    if spec:
        known_prefixes.add(normalize_provider_name(spec.model_prefix))
    if head and head in known_prefixes:
        bare_model = rest

    base_url = (
        prov_cfg.get("apiBase")
        or (getattr(spec, "default_api_base", "") if spec else "")
        or _PROVIDER_BASE_URL_FALLBACK.get(provider)
    )
    return {
        "model": bare_model,
        "api_key": prov_cfg.get("apiKey"),
        "base_url": base_url,
    }


def _prompt_text(label: str, *, secret: bool = False, default: str = "", allow_back: bool = False) -> Any:
    """Prompt for free text. With ``allow_back``, an empty submit returns
    ``oc._BACK`` (and a hint is shown); otherwise returns the stripped string."""
    questionary = oc._require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    placeholder = oc._back_placeholder(allow_back)
    if secret:
        value = questionary.password(label, placeholder=placeholder, style=OPENDDE_HARNESS_STYLE, qmark=oc._QMARK).ask()
    else:
        value = questionary.text(
            label, default=default, placeholder=placeholder, style=OPENDDE_HARNESS_STYLE, qmark=oc._QMARK
        ).ask()
    if value is None:
        raise typer.Exit(1)
    value = value.strip()
    if allow_back and value == "":
        return oc._BACK
    return value


def _probe_memory_chat(model: Optional[str], *, api_key: Optional[str], base_url: Optional[str]) -> tuple[bool, str]:
    """Real capability probe for a memory-LLM endpoint: ``POST
    {base_url}/chat/completions`` once and confirm a choice comes back. Unlike a
    bare ``GET /models`` connectivity check, this exercises the picked model, so
    an endpoint that lists models but doesn't serve the chosen id fails here
    instead of reporting a false green. Provider-agnostic; never raises."""
    import httpx

    if not base_url:
        return False, "no base_url configured"
    url = base_url.rstrip("/") + ("/chat/completions" if "/v1" in base_url else "/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, headers=headers, json=body)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
        return False, f"probe failed: {exc}"
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return True, "ok"
    return False, "endpoint returned no completion"


def _verify_memory_llm(
    label: str,
    *,
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    continue_hint: Optional[tuple[str, str]] = None,
) -> bool:
    """Probe the memory LLM with a real chat completion, offering retry/continue on failure."""
    oc.console.print(oc._t(f"  [dim]⏳ Verifying {label}…[/dim]", f"  [dim]⏳ 正在验证 {label}…[/dim]"))
    ok, detail = _probe_memory_chat(model, api_key=api_key, base_url=base_url)
    if ok:
        oc.console.print(oc._t(f"  [green]✓ {label} connected.[/green]", f"  [green]✓ {label} 连接成功。[/green]"))
        return True
    oc.console.print(
        oc._t(
            f"  [yellow]✗ Couldn't verify {label}: {detail}[/yellow]",
            f"  [yellow]✗ 验证失败 {label}:{detail}[/yellow]",
        )
    )
    if continue_hint:
        cont_label = oc._t(f"Continue anyway ({continue_hint[0]})", f"仍然继续({continue_hint[1]})")
    else:
        cont_label = oc._t("Continue anyway", "仍然继续")
    choice = oc._failure_choice(
        [
            (oc._t("Re-enter", "重新填写"), "rekey"),
            (cont_label, "continue"),
        ],
        non_interactive=non_interactive,
    )
    if choice == "rekey":
        return False
    warnings.append(label)
    return True


def _verify_rerank(
    label: str,
    *,
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    rerank_provider: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    continue_hint: Optional[tuple[str, str]] = None,
) -> bool:
    """Probe a rerank endpoint with a provider-specific request, offering retry/continue on failure."""
    oc.console.print(oc._t(f"  [dim]⏳ Verifying {label}…[/dim]", f"  [dim]⏳ 正在验证 {label}…[/dim]"))
    ok, detail = _probe_rerank(model, api_key=api_key, base_url=base_url, rerank_provider=rerank_provider)
    if ok:
        oc.console.print(oc._t(f"  [green]✓ {label} connected.[/green]", f"  [green]✓ {label} 连接成功。[/green]"))
        return True
    oc.console.print(
        oc._t(
            f"  [yellow]✗ Couldn't verify {label}: {detail}[/yellow]",
            f"  [yellow]✗ 验证失败 {label}:{detail}[/yellow]",
        )
    )
    if continue_hint:
        cont_label = oc._t(f"Continue anyway ({continue_hint[0]})", f"仍然继续({continue_hint[1]})")
    else:
        cont_label = oc._t("Continue anyway", "仍然继续")
    choice = oc._failure_choice(
        [
            (oc._t("Re-enter", "重新填写"), "rekey"),
            (cont_label, "continue"),
        ],
        non_interactive=non_interactive,
    )
    if choice == "rekey":
        return False
    warnings.append(label)
    return True


def _probe_rerank(
    model: Optional[str],
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    rerank_provider: Optional[str],
) -> tuple[bool, str]:
    """Real capability probe for a rerank endpoint. Dispatches by provider
    protocol (vllm / deepinfra / dashscope). Never raises."""
    import httpx

    if not base_url or not model:
        return False, "no base_url or model configured"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    headers["Content-Type"] = "application/json"

    try:
        if rerank_provider == "deepinfra":
            url = f"{base_url.rstrip('/')}/{model}"
            body: dict = {"queries": ["ping"], "documents": ["pong"]}
        elif rerank_provider == "dashscope":
            url = f"{base_url.rstrip('/')}/api/v1/services/rerank/text-rerank/text-rerank"
            body = {
                "model": model,
                "input": {"query": "ping", "documents": ["pong"]},
                "parameters": {"return_documents": False, "top_n": 1},
            }
        else:  # vllm / OpenAI-compat
            url = f"{base_url.rstrip('/')}/rerank"
            body = {"model": model, "query": "ping", "documents": ["pong"]}

        with httpx.Client(timeout=15) as client:
            resp = client.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
        return False, f"probe failed: {exc}"

    if rerank_provider == "deepinfra":
        scores = data.get("scores")
        if isinstance(scores, list) and scores:
            return True, "ok"
        return False, "endpoint returned no scores"
    if rerank_provider == "dashscope":
        output = data.get("output")
        results = output.get("results") if isinstance(output, dict) else None
        if isinstance(results, list) and results:
            return True, "ok"
        return False, "endpoint returned no results"
    # vllm
    results = data.get("results")
    if isinstance(results, list) and results:
        return True, "ok"
    return False, "endpoint returned no results"


_REQUIRED_EMBEDDING_DIM = 1024


def _probe_embedding_dim(url: str, headers: dict, model: str) -> int | str:
    """Try embedding with ``dimensions=1024``; fall back to native dim.

    Returns the effective dimension (int) on success, or an error
    description (str) on failure.
    """
    import httpx

    def _try_embed(client: httpx.Client, body: dict) -> int | str:
        try:
            resp = client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                return f"HTTP {resp.status_code}"
            items = resp.json().get("data", [])
            if not items:
                return "empty response"
            first = items[0]
            if not isinstance(first, dict):
                return "unexpected response format"
            return len(first.get("embedding", []))
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            return str(exc)

    with httpx.Client(timeout=15) as client:
        result = _try_embed(
            client, {"model": model, "input": ["dimension check"], "dimensions": _REQUIRED_EMBEDDING_DIM}
        )
        if result == _REQUIRED_EMBEDDING_DIM:
            return result
        return _try_embed(client, {"model": model, "input": ["dimension check"]})


def _verify_embedding_dim(
    *,
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    non_interactive: bool,
) -> bool:
    """Send a test embedding request and verify the vector dimension is 1024.

    Returns True to proceed, False to re-prompt.
    """
    if not base_url or not model:
        return True

    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    while True:
        oc.console.print(
            oc._t(
                "  [dim]⏳ Checking embedding dimension…[/dim]",
                "  [dim]⏳ 正在检测 embedding 维度…[/dim]",
            )
        )
        result = _probe_embedding_dim(url, headers, model)

        if result == _REQUIRED_EMBEDDING_DIM:
            oc.console.print(
                oc._t(
                    f"  [green]✓ Supports {result}-dim.[/green]",
                    f"  [green]✓ 支持 {result} 维。[/green]",
                )
            )
            return True

        if isinstance(result, int) and result < _REQUIRED_EMBEDDING_DIM:
            oc.console.print(
                oc._t(
                    f"  [red]✗ Dimension too small: model outputs {result}-dim, "
                    f"Persistent Memory requires >= {_REQUIRED_EMBEDDING_DIM}. Please pick another model.[/red]",
                    f"  [red]✗ 维度不足：模型输出 {result} 维，"
                    f"Persistent Memory 要求 >= {_REQUIRED_EMBEDDING_DIM} 维，请重新选择。[/red]",
                )
            )
            return False

        if isinstance(result, int) and result > _REQUIRED_EMBEDDING_DIM:
            oc.console.print(
                oc._t(
                    f"  [red]✗ Model outputs {result}-dim and does not support the "
                    f"dimensions parameter to truncate to {_REQUIRED_EMBEDDING_DIM}. "
                    "Please pick another model.[/red]",
                    f"  [red]✗ 模型输出 {result} 维，且不支持 dimensions 参数"
                    f"截断到 {_REQUIRED_EMBEDDING_DIM} 维，请重新选择。[/red]",
                )
            )
            return False

        oc.console.print(
            oc._t(
                f"  [yellow]✗ Couldn't verify dimension: {result}[/yellow]",
                f"  [yellow]✗ 无法验证维度：{result}[/yellow]",
            )
        )
        if non_interactive:
            return False
        choice = oc._failure_choice(
            [
                (oc._t("Retry", "重试"), "retry"),
                (oc._t("Re-enter", "重新选择"), "rekey"),
            ],
            non_interactive=False,
        )
        if choice == "rekey":
            return False


# Curated OpenAI-compatible endpoints for Persistent Memory models. Picking one
# pre-fills its base_url (mirrors the main provider step); everything else is
# reachable via "reuse an existing endpoint" or "custom" (type a base_url).
# These are the providers' documented OpenAI-compatible /v1 endpoints.
_MEMORY_PROVIDERS: list[dict[str, Any]] = [
    {
        "name": "openai",
        "label": "OpenAI",
        "label_zh": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "supports": {"llm", "embedding", "multimodal"},
    },
    {
        "name": "openrouter",
        "label": "OpenRouter",
        "label_zh": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "supports": {"llm", "embedding", "rerank", "multimodal"},
        "rerank_provider": "vllm",
    },
    {
        "name": "deepseek",
        "label": "DeepSeek",
        "label_zh": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "supports": {"llm"},
    },
    {
        "name": "deepinfra",
        "label": "DeepInfra",
        "label_zh": "DeepInfra",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "supports": {"llm", "embedding", "rerank"},
        "rerank_provider": "deepinfra",
        "rerank_base_url": "https://api.deepinfra.com/v1/inference",
    },
    {
        "name": "siliconflow",
        "label": "SiliconFlow",
        "label_zh": "硅基流动 SiliconFlow",
        "base_url": "https://api.siliconflow.cn/v1",
        "supports": {"llm", "embedding", "rerank"},
        "rerank_provider": "vllm",
    },
    {
        "name": "dashscope",
        "label": "DashScope (Alibaba)",
        "label_zh": "阿里百炼 DashScope",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "supports": {"llm", "embedding", "rerank"},
        "rerank_provider": "dashscope",
        "rerank_base_url": "https://dashscope.aliyuncs.com",
    },
]


def _match_provider_by_url(base_url: Optional[str]) -> Optional[str]:
    """Reverse-lookup a curated provider name from its base_url."""
    if not base_url:
        return None
    normalized = base_url.rstrip("/")
    for prov in _MEMORY_PROVIDERS:
        if prov["base_url"].rstrip("/") == normalized:
            return prov["name"]
    return None


# Per-role config: menu/verify label, model-id example, whether optional, and
# whether to run a connectivity probe after configuring (rerank/multimodal use
# non-chat endpoints whose /models probe isn't a reliable health check).
_MEMORY_ROLES: dict[str, dict[str, Any]] = {
    "llm": {
        "label": ("Memory LLM", "记忆 LLM"),
        "example": "gpt-4.1-mini",
        "optional": False,
        "verify": True,
        "purpose": (
            "Reads each conversation to judge what matters and extract the key points.",
            "从对话中判断信息边界、抽取要点。",
        ),
        # Worded as a floor rather than a default: the field is pre-filled with
        # the user's own main model, because a recommended id is only reachable
        # if their key carries it. This tells them how to judge their own.
        "recommendation": (
            "Capability floor: [bold]gpt-4.1-mini[/bold] -- weaker models degrade extraction",
            "能力下限参考 [bold]gpt-4.1-mini[/bold]：低于这个水平会明显影响提取质量",
        ),
        "continue_hint": ("memory extraction may fail", "记忆抽取可能失败"),
    },
    "embedding": {
        "label": ("Memory embedding", "记忆 embedding"),
        "example": "Qwen/Qwen3-Embedding-4B",
        # Optional in the sense that memory still functions without it: the
        # adapter drops to KEYWORD search, which needs no vectors. Strongly
        # advised all the same -- lexical recall misses a memory the moment the
        # user phrases the question differently.
        "optional": True,
        "verify": True,
        "purpose": (
            "Turns text into vectors so memories are found by meaning, not just keywords.",
            "把文字转成向量，让记忆能按「意思」检索，而不只是按关键词。",
        ),
        "tag": (
            "[accent](optional, strongly advised)[/accent]",
            "[accent]（可选，强烈建议配置）[/accent]",
        ),
        "cost": (
            "Without it: rephrase a question and it may miss a memory you have;\n  recall can only match keywords.",
            "不配置：换个说法提问就可能找不到已有记忆，记忆召回时只能使用关键词检索。",
        ),
        "recommendation": (
            "Recommended: [bold]Qwen/Qwen3-Embedding-4B[/bold] -- must be [bold yellow]1024-dim[/bold yellow],\n"
            "  Chinese + English",
            "推荐 [bold]Qwen/Qwen3-Embedding-4B[/bold]，需 [bold yellow]1024 维[/bold yellow]且支持中英文的模型",
        ),
        "continue_hint": ("semantic recall will be unavailable", "语义召回将不可用"),
        "skip_note": (
            "  [yellow]! Skipped: recall will match keywords, not meaning.[/yellow]\n"
            "  [dim]Phrase a question differently and it may miss a memory you have.[/dim]",
            "  [yellow]⚠ 已跳过：召回将按关键词匹配，而非按语义。[/yellow]\n"
            "  [dim]换一种说法提问，就可能找不到已有的记忆。[/dim]",
        ),
    },
    "rerank": {
        "label": ("Memory rerank", "记忆 rerank"),
        "example": "Qwen/Qwen3-Reranker-4B",
        "optional": True,
        "verify": True,
        "purpose": (
            "Re-ranks what semantic search found so the best match comes first, at a small\n  latency cost.",
            "在语义召回一批候选后再精排一遍，让最相关的排在最前，会略增延迟。",
        ),
        "tag": (
            "[accent](optional, advised)[/accent]",
            "[accent]（可选，建议配置）[/accent]",
        ),
        "recommendation": (
            "Recommended: [bold]Qwen/Qwen3-Reranker-4B[/bold]",
            "推荐 [bold]Qwen/Qwen3-Reranker-4B[/bold]",
        ),
        "continue_hint": ("rerank quality may degrade", "rerank 精度可能下降"),
        "skip_note": (
            "  [dim]Skipped rerank; memory retrieval still works.[/dim]",
            "  [dim]已跳过 rerank，记忆检索仍可用。[/dim]",
        ),
    },
    "multimodal": {
        "label": ("Memory multimodal", "记忆多模态"),
        "example": "google/gemini-3-flash-preview",
        "optional": True,
        "verify": True,
        "purpose": (
            "Lets OpenDDE Harness understand and recall images / PDFs / audio as memory.",
            "让 OpenDDE Harness 把图片 / PDF / 音频也作为记忆来理解和检索。",
        ),
        "cost": (
            "Without it: those files stay out of memory. Having such files is not the same\n"
            "  as needing them remembered -- configure it when you do.",
            "不配置：这类文件不进入记忆；有这类文件并不等于需要，确有此需求时再配即可。",
        ),
        "recommendation": (
            "Recommended: [bold]google/gemini-3-flash-preview[/bold]",
            "推荐 [bold]google/gemini-3-flash-preview[/bold]",
        ),
        "skip_note": (
            "  [dim]Skipped; nothing else is affected -- configure it if you come to need\n  multimodal memory.[/dim]",
            "  [dim]已跳过；其余功能不受影响，日后确有把多模态内容纳入记忆的需求时再配即可。[/dim]",
        ),
    },
}


_EMBEDDING_MODEL_PATTERNS = ("embed", "bge", "e5-", "gte-")
_MULTIMODAL_MODEL_PATTERNS = ("vision", "4o", "gemini", "pixtral", "qwen-vl", "qwen2-vl", "qwen2.5-vl")


def _fetch_memory_models(
    base_url: Optional[str],
    api_key: Optional[str],
    *,
    section: str = "llm",
    provider_name: Optional[str] = None,
) -> Optional[list[str]]:
    """Fetch available model ids from a provider endpoint. Never raises.

    For ``section="embedding"``, delegates to per-provider logic because
    each provider exposes embedding models differently.
    """
    if not base_url:
        return None
    if section == "embedding":
        return _fetch_embedding_models(base_url, api_key, provider_name)
    if section == "rerank":
        return _fetch_rerank_models(base_url, api_key, provider_name)
    if section == "multimodal":
        return _fetch_multimodal_models(base_url, api_key, provider_name)
    return _fetch_openai_models(base_url, api_key)


def _fetch_openai_models(
    base_url: str,
    api_key: Optional[str],
    *,
    params: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """``GET {base_url}/models`` with OpenAI-style response parsing."""
    import httpx

    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return None
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    ids = [m.get("id") for m in items if isinstance(m, dict) and m.get("id")]
    return sorted(ids) or None


def _fetch_deepinfra_models(
    api_key: Optional[str],
    reported_type: str,
    *,
    name_contains: Optional[str] = None,
) -> Optional[list[str]]:
    """Fetch DeepInfra models filtered by ``reported_type`` and optional name substring."""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get("https://api.deepinfra.com/models/list", headers=headers)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    items = data if isinstance(data, list) else []
    ids = [
        m.get("model_name")
        for m in items
        if isinstance(m, dict)
        and m.get("reported_type") == reported_type
        and m.get("model_name")
        and (name_contains is None or name_contains in m.get("model_name", ""))
    ]
    return sorted(ids) or None


def _fetch_embedding_models(
    base_url: str,
    api_key: Optional[str],
    provider_name: Optional[str],
) -> Optional[list[str]]:
    """Provider-specific embedding model listing."""
    if provider_name == "openrouter":
        return _fetch_openai_models(base_url.rstrip("/") + "/embeddings", api_key)

    if provider_name == "siliconflow":
        return _fetch_openai_models(base_url, api_key, params={"type": "text", "sub_type": "embedding"})

    if provider_name == "deepinfra":
        return _fetch_deepinfra_models(api_key, "embeddings")

    # OpenAI, DashScope, custom — GET /models + name-based filter.
    ids = _fetch_openai_models(base_url, api_key)
    if ids is None:
        return None
    filtered = [i for i in ids if any(p in i.lower() for p in _EMBEDDING_MODEL_PATTERNS)]
    return filtered or None


def _fetch_rerank_models(
    base_url: str,
    api_key: Optional[str],
    provider_name: Optional[str],
) -> Optional[list[str]]:
    """Provider-specific rerank model listing."""
    if provider_name == "deepinfra":
        # The deepinfra provider hardcodes a Qwen3-Reranker chat template,
        # so only Qwen3-Reranker models are compatible.
        return _fetch_deepinfra_models(api_key, "reranker", name_contains="Qwen3-Reranker")

    if provider_name == "siliconflow":
        return _fetch_openai_models(base_url, api_key, params={"sub_type": "reranker"})

    if provider_name == "dashscope":
        return ["gte-rerank-v2"]

    if provider_name == "openrouter":
        return _fetch_openai_models(base_url, api_key, params={"output_modalities": "rerank"})

    # vllm / custom — no standard rerank listing.
    return None


def _fetch_multimodal_models(
    base_url: str,
    api_key: Optional[str],
    provider_name: Optional[str],
) -> Optional[list[str]]:
    """Provider-specific multimodal (vision) model listing."""
    if provider_name == "openrouter":
        return _fetch_openai_models(base_url, api_key, params={"input_modalities": "image"})

    # OpenAI, custom — GET /models + name-based filter.
    ids = _fetch_openai_models(base_url, api_key)
    if ids is None:
        return None
    filtered = [i for i in ids if any(p in i.lower() for p in _MULTIMODAL_MODEL_PATTERNS)]
    return filtered or None


def _match_memory_default(example: str, models: list[str]) -> str:
    """Find the best match for ``example`` in the fetched model list.

    The example (e.g. ``gpt-4.1-mini``) is a bare model name, while
    ``models`` may carry provider prefixes (``openai/gpt-4.1-mini``).
    Returns the first model whose id ends with ``/example`` or equals
    ``example`` exactly; otherwise selects the first available model.
    """
    lower = example.lower()
    suffix = f"/{lower}"
    for mid in models:
        if mid.lower() == lower or mid.lower().endswith(suffix):
            return mid
    return models[0] if models else ""


def _preferred_memory_model(section: str, main_model: Optional[str], chosen_provider: Optional[str]) -> Optional[str]:
    """The main chat model, when it is a sensible pre-fill for this role.

    Only the llm role -- an embedding / rerank / multimodal endpoint does not
    serve a chat model. Only when the picked provider is the main model's own: no
    other provider carries that id, and pre-filling one it cannot serve turns
    Enter into a verification failure. A custom endpoint has no resolved provider
    and is left alone for the same reason.
    """
    if section != "llm" or not main_model or chosen_provider is None:
        return None
    if chosen_provider != _resolve_model_provider(main_model):
        return None
    return _resolve_reuse_llm_creds(main_model).get("model")


def _memory_pick_model(
    *,
    base_url: Optional[str],
    api_key: Optional[str],
    example: str,
    allow_back: bool,
    section: str = "llm",
    provider_name: Optional[str] = None,
    recommendation: Optional[tuple[str, str]] = None,
    preferred: Optional[str] = None,
) -> Any:
    """Choose a listed memory model, with explicit manual-entry and back options."""

    oc.console.print(oc._t("  [dim]⏳ Loading models…[/dim]", "  [dim]⏳ 正在拉取模型列表…[/dim]"))
    models = _fetch_memory_models(base_url, api_key, section=section, provider_name=provider_name) or []
    default_model = _match_memory_default(preferred or example, models)
    if (
        preferred
        and default_model
        and (default_model.lower() == preferred.lower() or default_model.lower().endswith(f"/{preferred.lower()}"))
    ):
        oc.console.print(
            oc._t(
                f"  [dim]Your main model [bold]{default_model}[/bold] is selected -- press Enter to accept.[/dim]",
                f"  [dim]已默认选中你的主模型 [bold]{default_model}[/bold]，回车确认。[/dim]",
            )
        )
    if recommendation:
        oc.console.print(f"  [dim]{oc._t(*recommendation)}[/dim]")
    if not models:
        oc.console.print(
            oc._t(
                "  [dim]Couldn't list models from this endpoint — select manual entry to continue.[/dim]",
                "  [dim]该端点拉不到模型列表 — 请选择手动输入模型 ID。[/dim]",
            )
        )
    return oc._select_model_id(
        models,
        default_model=default_model or preferred,
        allow_back=allow_back,
    )


def _memory_pick_creds_and_model(
    *,
    section: str,
    example: str,
    main_model: Optional[str],
    non_interactive: bool,
    recommendation: Optional[tuple[str, str]] = None,
) -> Any:
    """Mirror the main provider step for one Persistent Memory model: pick a source
    (curated provider / custom) → API key → model. Returns a dict with
    ``model`` / ``api_key`` / ``base_url`` (plus ``provider`` for rerank), or
    ``oc._BACK`` when the user backs out of the source picker. Empty submit on any
    field rewinds one step."""
    questionary = oc._require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    llm_section = _memory_section("llm")

    # For the LLM role, default to the main chat model's provider.
    # For other roles (embedding/rerank/multimodal), default to whichever
    # provider the LLM step just configured — the user likely has the
    # same API key and only needs to pick a different model.
    if section == "llm":
        default_provider = _resolve_model_provider(main_model or "")
        reuse_source = "main"
    else:
        default_provider = _match_provider_by_url(llm_section.get("base_url"))
        reuse_source = "llm"

    while True:  # source picker — a field-level back rewinds here
        choices: list[Any] = []
        default_choice = None
        for prov in _MEMORY_PROVIDERS:
            if section not in prov.get("supports", set()):
                continue
            is_default = default_provider is not None and prov["name"] == default_provider
            if is_default:
                if reuse_source == "main":
                    label = oc._t(
                        f"{prov['label']} (main model provider, reuse Key)",
                        f"{prov['label_zh']}（主模型服务商，复用 Key）",
                    )
                else:
                    label = oc._t(
                        f"{prov['label']} (memory LLM provider, reuse Key)",
                        f"{prov['label_zh']}（记忆 LLM 服务商，复用 Key）",
                    )
            else:
                label = oc._t(prov["label"], prov["label_zh"])
            choice = questionary.Choice(label, value=("provider", prov))
            choices.append(choice)
            if is_default:
                default_choice = choice.value
        choices.append(
            questionary.Choice(
                oc._t("Other (custom OpenAI-compatible endpoint)", "其他(自定义 OpenAI 兼容端点)"),
                value=("custom",),
            )
        )
        choices.append(questionary.Separator())
        choices.append(questionary.Choice(oc._t("Back", "返回"), value=oc._BACK))

        src = questionary.select(
            oc._t("Pick a provider (or reuse / custom):", "选择服务商(或复用 / 自定义):"),
            choices=choices,
            default=default_choice,
            style=OPENDDE_HARNESS_STYLE,
            qmark=oc._QMARK,
        ).ask()
        if src is None:
            raise typer.Exit(1)
        if src is oc._BACK:
            return oc._BACK
        kind = src[0]

        # Resolve (api_key, base_url) from the chosen source.
        chosen_provider: Optional[str] = None
        if kind == "provider":
            chosen_provider = src[1]["name"]
            base_url = src[1]["base_url"]
            prefilled_key: Optional[str] = None
            if default_provider == src[1]["name"]:
                if reuse_source == "main":
                    prefilled_key = _resolve_reuse_llm_creds(main_model or "").get("api_key")
                else:
                    prefilled_key = llm_section.get("api_key")
            if prefilled_key:
                if reuse_source == "main":
                    oc.console.print(
                        oc._t(
                            "  [dim]API key reused from main chat model.[/dim]",
                            "  [dim]已复用主对话模型的 API Key。[/dim]",
                        )
                    )
                else:
                    oc.console.print(
                        oc._t(
                            "  [dim]API key reused from memory LLM.[/dim]",
                            "  [dim]已复用记忆 LLM 的 API Key。[/dim]",
                        )
                    )
                api_key = prefilled_key
            else:
                api_key = oc._prompt_api_key(src[1]["name"], allow_back=True)
                if api_key is oc._BACK:
                    continue
        else:  # custom
            base_url = _prompt_text(oc._t("Base URL (must include /v1):", "Base URL(需包含 /v1):"), allow_back=True)
            if base_url is oc._BACK:
                continue
            api_key = _prompt_text(oc._t("API key (hidden):", "API Key(隐藏输入):"), secret=True, allow_back=True)
            if api_key is oc._BACK:
                continue

        # Guard against a source that resolved to an empty key / endpoint —
        # set_memory_section drops None values, which would otherwise persist a
        # section with a model but no usable endpoint.
        if not (api_key and base_url):
            oc.console.print(
                oc._t(
                    "  [yellow]✗ Missing API key or Base URL for this source — pick another.[/yellow]",
                    "  [yellow]✗ 该来源缺少 API Key 或 Base URL — 请换一个。[/yellow]",
                )
            )
            continue

        # rerank: resolve service type + override base_url when needed.
        rerank_provider: Optional[str] = None
        if section == "rerank":
            chosen_prov_dict = src[1] if kind == "provider" else None
            if chosen_prov_dict and chosen_prov_dict.get("rerank_provider"):
                rerank_provider = chosen_prov_dict["rerank_provider"]
                if chosen_prov_dict.get("rerank_base_url"):
                    base_url = chosen_prov_dict["rerank_base_url"]
            else:
                rerank_provider = questionary.select(
                    oc._t("Rerank service type:", "rerank 服务类型:"),
                    choices=[
                        questionary.Choice("deepinfra", value="deepinfra"),
                        questionary.Choice("vllm", value="vllm"),
                        questionary.Choice("dashscope", value="dashscope"),
                        questionary.Choice(oc._t("Back", "返回"), value=oc._BACK),
                    ],
                    style=OPENDDE_HARNESS_STYLE,
                    qmark=oc._QMARK,
                ).ask()
                if rerank_provider is None:
                    raise typer.Exit(1)
                if rerank_provider is oc._BACK:
                    continue

        model = _memory_pick_model(
            base_url=base_url,
            api_key=api_key,
            example=example,
            allow_back=True,
            section=section,
            provider_name=chosen_provider,
            recommendation=recommendation,
            preferred=_preferred_memory_model(section, main_model, chosen_provider),
        )
        if model is oc._BACK:
            continue

        result: dict[str, Any] = {"model": model, "api_key": api_key, "base_url": base_url}
        if rerank_provider:
            result["provider"] = rerank_provider
        return result


def _lightweight_memory_model(main_model: str, models: list[str]) -> str:
    tiers = {"pro", "reasoner", "chat", "sonnet", "opus", "mini", "nano", "flash", "haiku", "lite"}
    light = ("mini", "flash", "haiku", "lite", "nano", "chat")

    def family(model: str) -> list[str]:
        return [part for part in model.lower().split("-") if part not in tiers]

    candidates = [
        model
        for model in models
        if isinstance(model, str)
        and family(model) == family(main_model)
        and any(tier in model.lower().split("-") for tier in light)
    ]
    return min(
        candidates,
        key=lambda model: (next(i for i, tier in enumerate(light) if tier in model.lower().split("-")), model),
        default=main_model,
    )


def _auto_configure_memory_llm(main_model: Optional[str], *, skip_test: bool) -> bool:
    from opendde_harness.config.update_memory import set_memory_section

    if not main_model or not _model_is_openai_compatible(main_model) or _memory_role_configured("llm"):
        return False
    creds = _resolve_reuse_llm_creds(main_model)
    if not creds.get("api_key") or not creds.get("base_url") or not creds.get("model"):
        return False
    main = creds["model"]
    models = _fetch_memory_models(creds["base_url"], creds["api_key"]) or []
    preferred = _lightweight_memory_model(main, models)
    for model in dict.fromkeys((preferred, main)):
        if not skip_test:
            ok, _ = _probe_memory_chat(model, api_key=creds["api_key"], base_url=creds["base_url"])
            if not ok:
                continue
        set_memory_section("llm", {**creds, "model": model})
        oc.console.print(
            oc._t(
                f"  ✓ Memory LLM automatically configured: {model} (reusing your main model's provider and API key)",
                f"  ✓ 记忆 LLM：{model}（已自动复用主模型的服务商和 Key）",
            ),
            markup=False,
        )
        return True
    oc.console.print(
        oc._t(
            "  Automatic memory LLM verification failed; configure it below.",
            "  记忆 LLM 自动验证未通过，请在下方配置。",
        )
    )
    return False


def _config_memory_role(
    *, section: str, main_model: Optional[str], non_interactive: bool, warnings: list[str], skip_test: bool = False
) -> Any:
    """Configure one Persistent Memory role (llm / embedding / rerank / multimodal)
    with the unified provider→key→model flow, reuse shortcuts, and a back loop.

    Returns ``None`` normally; returns ``oc._ABORT_MEMORY`` when the user gives up a
    required role (the caller then disables Persistent Memory, leaving no long-term memory)."""
    if section == "llm" and _auto_configure_memory_llm(main_model, skip_test=skip_test):
        return
    questionary = oc._require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE
    from opendde_harness.config.update_memory import clear_memory_section, set_memory_section

    role = _MEMORY_ROLES[section]
    label_en, label_zh = role["label"]
    purpose_en, purpose_zh = role["purpose"]
    optional = role["optional"]
    verify_label = oc._t(label_en, label_zh)

    # Tell the user what this model is for, and what skipping it costs, before
    # asking them to configure it. Header sits on the 2-space info column (bold
    # accent); purpose and cost nest under it, matching the layout used
    # everywhere else.
    #
    # The cost line is dim rather than a warning colour on purpose: this is
    # pre-decision information, and colouring it would cry wolf before the user
    # has chosen anything. The warning comes after, from ``skip_note``.
    #
    # Roles that want to be configured say so in their own ``tag`` -- calling all
    # three merely "optional" flattens the difference between losing semantic
    # recall entirely and losing a little ranking accuracy.
    tag_markup = oc._t(*role["tag"]) if role.get("tag") else oc._t("[dim](optional)[/dim]", "[dim]（可选）[/dim]")
    lines = [f"  [bold][accent]{oc._t(label_en, label_zh)}[/accent][/bold]" + (f" {tag_markup}" if optional else "")]
    lines.append(f"  [dim]{oc._t(purpose_en, purpose_zh)}[/dim]")
    if role.get("cost"):
        lines.append(f"  [dim]{oc._t(*role['cost'])}[/dim]")
    oc.console.print()
    # highlight=False so Rich's default highlighter doesn't tint the dim prose
    # (parens/numbers/words) and make an informational hint read like an error.
    oc.console.print("\n".join(lines), highlight=False)

    while True:  # role-menu loop — a back-out of the source picker returns here
        current = _memory_section(section).get("model") if _memory_role_configured(section) else None
        if current:
            choices = [
                questionary.Choice(oc._t(f"Keep current: {current}", f"沿用当前:{current}"), value="keep"),
                questionary.Choice(oc._t("Reconfigure", "重新配置"), value="redo"),
            ]
            if optional:
                choices.append(questionary.Choice(oc._t("Skip", "跳过"), value="off"))
            action = questionary.select(
                oc._t("Already configured — what now?", "已配置,怎么处理?"),
                choices=choices,
                style=OPENDDE_HARNESS_STYLE,
                qmark=oc._QMARK,
            ).ask()
            if action is None:
                raise typer.Exit(1)
            if action == "keep":
                return
            if action == "off":
                clear_memory_section(section)
                oc.console.print(oc._t(f"  [dim]{label_en} skipped.[/dim]", f"  [dim]已跳过 {label_zh}。[/dim]"))
                return
        elif optional:
            action = questionary.select(
                oc._t("Configure it?", "要配置吗?"),
                choices=[
                    questionary.Choice(oc._t("Configure", "配置"), value="redo"),
                    questionary.Choice(oc._t("Skip", "跳过"), value="skip"),
                ],
                style=OPENDDE_HARNESS_STYLE,
                qmark=oc._QMARK,
            ).ask()
            if action is None:
                raise typer.Exit(1)
            if action == "skip":
                # Printed verbatim rather than wrapped in [dim]: skipping rerank
                # costs ordering, skipping embedding costs semantic recall
                # entirely, and one of those deserves to be seen.
                note_en, note_zh = role.get(
                    "skip_note", (f"  [dim]Skipped {label_en}.[/dim]", f"  [dim]已跳过 {label_zh}。[/dim]")
                )
                oc.console.print(oc._t(note_en, note_zh), highlight=False)
                return
        # A required role with nothing configured falls straight into the picker.

        result = _memory_pick_creds_and_model(
            section=section,
            example=role["example"],
            main_model=main_model,
            non_interactive=non_interactive,
            recommendation=role.get("recommendation"),
        )
        if result is oc._BACK:
            if optional or _memory_role_configured(section):
                # Optional roles offer Skip; a required role already configured
                # falls back to its keep/reconfigure menu. Either way, re-show
                # the role menu rather than forcing the give-up exit.
                continue
            # A required role with nothing configured has no Skip, so backing out
            # of the picker would loop forever. Offer a bounded exit -- keep
            # trying, or leave without long-term memory. Stated in full and in
            # colour: this is the only place the wizard can lose memory
            # altogether, and "no cross-session memory" is a consequence a user
            # should not discover weeks later by noticing the agent forgets
            # everything.
            oc.console.print()
            oc.console.print(
                oc._t(
                    f"  [yellow]⚠ {label_en} is required for long-term memory.[/yellow]\n"
                    "  [dim]Without it OpenDDE Harness has no memory across sessions: every conversation starts\n"
                    "  from nothing, with no recollection of your preferences or of what was done before.[/dim]",
                    f"  [yellow]⚠ {label_zh} 是长期记忆的必需项。[/yellow]\n"
                    "  [dim]放弃后 OpenDDE Harness 没有任何跨会话记忆：每次对话都从零开始，不记得你的偏好，\n"
                    "  也不记得之前做过什么。[/dim]",
                ),
                highlight=False,
            )
            action = questionary.select(
                oc._t("What would you like to do?", "想做什么？"),
                choices=[
                    questionary.Choice(oc._t("Pick a provider / model", "选择服务商 / 模型"), value="retry"),
                    questionary.Choice(
                        oc._t("Give up (no long-term memory)", "放弃（不启用长期记忆）"),
                        value="abort",
                    ),
                ],
                style=OPENDDE_HARNESS_STYLE,
                qmark=oc._QMARK,
            ).ask()
            if action is None:
                raise typer.Exit(1)
            if action == "retry":
                continue
            return oc._ABORT_MEMORY

        if role["verify"] and skip_test:
            oc.console.print(
                oc._t(
                    f"  [dim]Skipping the {verify_label} test call (--skip-test).[/dim]",
                    f"  [dim]已跳过 {verify_label} 的测试调用(--skip-test)。[/dim]",
                )
            )
            ok = True
        elif section == "llm":
            ok = _verify_memory_llm(
                verify_label,
                model=result["model"],
                api_key=result["api_key"],
                base_url=result["base_url"],
                non_interactive=non_interactive,
                warnings=warnings,
                continue_hint=role.get("continue_hint"),
            )
        elif section == "embedding":
            ok = _verify_embedding_dim(
                model=result["model"],
                api_key=result["api_key"],
                base_url=result["base_url"],
                non_interactive=non_interactive,
            )
        elif section == "rerank":
            ok = _verify_rerank(
                verify_label,
                model=result["model"],
                api_key=result["api_key"],
                base_url=result["base_url"],
                rerank_provider=result.get("provider"),
                non_interactive=non_interactive,
                warnings=warnings,
                continue_hint=role.get("continue_hint"),
            )
        elif section == "multimodal":
            ok = _verify_memory_llm(
                verify_label,
                model=result["model"],
                api_key=result["api_key"],
                base_url=result["base_url"],
                non_interactive=non_interactive,
                warnings=warnings,
                continue_hint=role.get("continue_hint"),
            )
        else:
            ok = True
        if not ok:
            continue

        set_memory_section(section, result)
        oc.console.print(
            oc._t(
                f"  [green]✓ {label_en} configured.[/green]",
                f"  [green]✓ 已配置 {label_zh}。[/green]",
            )
        )
        return


_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def _same_address(a: str | None, b: str | None) -> bool:
    """Whether two URLs name the same socket, allowing for spelling.

    The declared address is copied out of the root's toml and the target is
    built from a recorded port, so the same endpoint routinely arrives as
    ``127.0.0.1`` on one side and ``localhost`` on the other. Comparing the
    strings makes a service already on its configured port look like drift, and
    the wizard then offers to move it onto itself.
    """
    from urllib.parse import urlparse

    if not a or not b:
        return False
    pa, pb = urlparse(a), urlparse(b)
    if pa.port != pb.port:
        return False
    ha, hb = (pa.hostname or "").lower(), (pb.hostname or "").lower()
    if ha == hb:
        return True
    return ha in _LOOPBACK_NAMES and hb in _LOOPBACK_NAMES


def _lock_holder(root: Path | str):
    """The process serving ``root``, or None. Indirected so callers can stub it."""
    from opendde_harness.plugin.memory.longterm._server import lock_holder

    return lock_holder(root)


def _stop_for_reload(root: Path | str) -> bool:
    """Stop our own server for ``root`` so a rewritten config is actually read.

    Persistent Memory builds its LLM client in the API lifespan, at startup, so a process
    already running keeps the models it booted with. Without this the wizard
    wrote new models to disk and the closing ``ensure_memory_server`` found the
    address answering and returned -- the reconfiguration was inert until some
    unrelated restart, with nothing on screen saying so.

    Not a decision to put to the user: they asked to reconfigure and filled in
    the models a moment ago, and applying them is what that means. Only a server
    opendde can identify as serving this root is touched.
    """
    from opendde_harness.plugin.memory.longterm._server import StopOutcome, stop_pid

    holder = _lock_holder(root)
    if holder is None:
        return False
    oc.console.print(
        oc._t(
            "  [dim]Restarting the service so it picks up the new configuration...[/dim]",
            "  [dim]正在重启服务以加载新配置...[/dim]",
        )
    )
    # The pid the lock named, not the one the pidfile remembers. Asking the
    # pidfile here would report "not ours" about the very process just
    # identified, which is the state the lock lookup exists to get out of.
    outcome = stop_pid(holder.pid)
    if outcome is not StopOutcome.STOPPED:
        reason = {
            StopOutcome.SIGNAL_FAILED: oc._t("the stop signal could not be delivered", "停止信号发送失败"),
            StopOutcome.STILL_DRAINING: oc._t("it is still finishing memory work", "它还在收尾未完成的记忆任务"),
        }.get(outcome, oc._t("it did not stop", "它没有停下"))
        oc.console.print(
            oc._t(
                f"  [yellow]! The service is still running ({reason}), so it keeps the models it "
                "started with. The new ones take effect the next time it starts.[/yellow]",
                f"  [yellow]⚠ 服务仍在运行（{reason}），它用的还是启动时那套模型。"
                "新模型将在它下次启动时生效。[/yellow]",
            ),
            highlight=False,
        )
        return False
    return True


def _port_is_free(port: int) -> bool:
    """Whether a local TCP port can still be bound."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _ask_managed_port(root: Path | str) -> int:
    """Where a opendde-managed server should listen.

    Silent while the intended port is free, which is nearly always. 18791 is a
    recommendation rather than a fixed address, but turning that into a screen
    on every fresh install asks for a decision almost nobody has to make. The
    case that had no way forward is the occupied one: the start failed and the
    wizard had nothing to offer, so that is where the question belongs.

    The default offered is whatever is already recorded, not the shipped
    constant, so a second run does not quietly undo a port the user moved to by
    inviting them to press Enter on the old one.
    """
    from opendde_harness.plugin.memory.longterm._server import DEFAULT_MEMORY_BASE_URL

    # Creating a root is the other question, and here a recorded address is the
    # best answer available: ignoring it is what "start the memory server at
    # its configured address, not the default" was about.
    slice_ = _recorded_memory_slice()
    recorded = slice_.get("port")
    if not (isinstance(recorded, int) and recorded > 0):
        recorded = urlparse(str(slice_.get("base_url") or "")).port
    current = recorded or urlparse(DEFAULT_MEMORY_BASE_URL).port or 18791
    if _port_is_free(current):
        return int(current)
    # A bind test cannot tell a stranger from our own service. When the holder
    # of this root's lock is the thing listening there, the port is not taken --
    # it is ours, and the step is about to restart into it.
    holder = _lock_holder(root)
    if holder is not None and holder.port == current:
        return int(current)
    oc.console.print(
        oc._t(
            f"  [yellow]! Port {current} is already in use by something else.[/yellow]",
            f"  [yellow]⚠ 端口 {current} 已被别的程序占用。[/yellow]",
        )
    )
    answer = _prompt_text(
        oc._t("Memory service port:", "记忆服务端口:"),
        default=str(current),
    )
    if not str(answer).isdigit():
        oc.console.print(
            oc._t(
                f"  [dim]Not a port ({answer}); keeping {current}.[/dim]",
                f"  [dim]不是合法端口（{answer}），仍用 {current}。[/dim]",
            )
        )
        return int(current)
    return int(answer)


def _retry_or_skip_address() -> str:
    """After a bad address: type another, or give up on memory for now.

    Both ways this is reached -- a port with a typo in it, a server not started
    yet -- are fixed by one more line of input, so ending the step over either
    would send the user back through the whole wizard for a digit. Giving up
    stays available and is what the caller turns into "memory off"; what is not
    on offer is being redirected into a managed setup nobody asked for.
    """
    questionary = oc._require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    choice = questionary.select(
        oc._t("What now?", "怎么办？"),
        choices=[
            questionary.Choice(oc._t("Enter a different address", "重新填写地址"), value="retry"),
            questionary.Choice(
                oc._t("Skip", "跳过"),
                value="skip",
            ),
        ],
        style=OPENDDE_HARNESS_STYLE,
        qmark=oc._QMARK,
    ).ask()
    if choice is None:
        raise typer.Exit(1)
    return str(choice)


def _use_self_managed_memory() -> bool:
    """Point opendde at a Persistent Memory the user runs. Returns False if it is unreachable.

    The whole path is two prompts and one probe. Nothing is scanned: an address
    opendde guessed is an address the user did not confirm, and this is precisely
    the setup where guessing wrong means writing to, or taking the lock on,
    something that is not opendde's.

    No port default. Offering one -- and the only one opendde could offer is its
    own -- invites acceptance by reflex, which would point the self-managed path
    at the managed path's address. Someone who runs their own Persistent Memory knows the
    port; someone who does not should be on the other branch.

    Only the address is recorded. Without a root there is no path on disk for a
    later change to write to by accident, which turns the read-only promise from
    a convention into something the code cannot break.
    """
    from opendde_harness.config.update import set_plugin_config_fields
    from opendde_harness.plugin.memory.longterm._server import ProbeResult, probe_health

    while True:
        host = _prompt_text(oc._t("Host (e.g. 127.0.0.1):", "主机(如 127.0.0.1):"), default="localhost")
        port = _prompt_text(oc._t("Port:", "端口:"))
        if not port.isdigit():
            oc.console.print(oc._t(f"  [red]x Not a port: {port}[/red]", f"  [red]✗ 不是合法端口：{port}[/red]"))
            if _retry_or_skip_address() == "retry":
                continue
            return False

        base_url = f"http://{host}:{port}"
        oc.console.print(oc._t(f"  [dim]Checking {base_url}...[/dim]", f"  [dim]正在检查 {base_url}...[/dim]"))
        result = probe_health(base_url)
        if result is ProbeResult.OK:
            break
        oc.console.print(
            oc._t(
                f"  [red]x No Persistent Memory answered at {base_url} ({result.value}).[/red]",
                f"  [red]✗ {base_url} 上没有 Persistent Memory 响应（{result.value}）。[/red]",
            ),
            highlight=False,
        )
        if _retry_or_skip_address() == "retry":
            continue
        return False

    set_plugin_config_fields(
        PLUGIN_ID,
        {"owned": False, "base_url": base_url},
        # A merging write cannot express "this no longer applies", and a root
        # left behind from a previous managed setup is still exported as
        # the library's root variable -- pointing opendde at a directory it just promised to stop
        # touching. Not recording a path is what makes the promise structural.
        remove=("root", "port"),
    )
    _set_memory_backend(BACKEND_NAME)
    caps = " ".join(_capability_lines(base_url))
    oc.console.print(
        oc._t(
            f"  [green]v OpenDDE Harness will use the Persistent Memory at {base_url}.[/green]\n"
            f"      capability  {caps}\n"
            "  [dim]You run it: OpenDDE Harness never edits its config and never starts or stops it.\n"
            "  If it is down when a session begins, that session has no long-term memory.[/dim]",
            f"  [green]✓ OpenDDE Harness 将使用 {base_url} 上的 Persistent Memory。[/green]\n"
            f"      能力       {caps}\n"
            "  [dim]它由你运行：OpenDDE Harness 不会改它的配置，也不会启停它。\n"
            "  会话开始时它若没在运行，这次会话就没有长期记忆。[/dim]",
        ),
        highlight=False,
    )
    return True


def _record_root(root: Any, owned: bool) -> None:
    """Record which Persistent Memory root is in use and whether opendde may write to it.

    Both are decisions rather than derivable facts, and every later reader --
    the wizard, the runtime, doctor -- consults the record instead of guessing
    again. Recording ``owned`` is what makes read-only reuse enforceable.
    """
    from opendde_harness.config.update import set_plugin_config_fields

    set_plugin_config_fields(PLUGIN_ID, {"root": str(root), "owned": owned})


def _capability_lines(base_url: str) -> list[str]:
    """One line per capability the running server actually built.

    "Running" stopped implying "working" in library 1.2.1: a server whose
    embedding provider failed to build still answers 200 and quietly degrades to
    keyword-only recall.
    """
    from opendde_harness.plugin.memory.longterm._health import (
        DEGRADING_SECTIONS,
        REQUIRED_SECTIONS,
        capability_available,
        probe_capabilities,
    )

    report = probe_capabilities(base_url)
    if not report.reports_capabilities:
        return []
    lines = []
    for section in (*REQUIRED_SECTIONS, *DEGRADING_SECTIONS):
        state = capability_available(report.capabilities, section)
        mark = "[green]✓[/green]" if state is True else "[red]✗[/red]" if state is False else "[dim]?[/dim]"
        lines.append(f"{mark} {section}")
    return lines


def _reuse_unowned_root(state: Any) -> object:
    """Screens for a Persistent Memory the user manages: reuse it, or leave it alone.

    Read-only throughout. opendde records the address and reports what the server
    can do; it does not write the config (those are the user's keys and models to
    set) and does not start it (starting takes the OME jobstore lock
    exclusively, which is theirs to grant).
    """
    questionary = oc._require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    where = state.declared_url or "?"
    oc.console.print()
    oc.console.print(
        oc._t(
            "  [dim]No Persistent Memory service managed by OpenDDE Harness was found.[/dim]",
            "  [dim]未找到 OpenDDE Harness 管理的 Persistent Memory 服务。[/dim]",
        )
    )

    while True:
        if state.serving:
            caps = " ".join(_capability_lines(where))
            oc.console.print(
                oc._t(
                    f"  [green]✓ Found another Persistent Memory service, running[/green]\n"
                    f"      memory dir  {state.root}   [dim]<- you manage this; "
                    f"OpenDDE Harness will not modify it, nor start or stop it[/dim]\n"
                    f"      address     {where}\n"
                    f"      capability  {caps}",
                    f"  [green]✓ 发现另一个 Persistent Memory 服务，正在运行[/green]\n"
                    f"      记忆目录   {state.root}   [dim]<- 由你管理，OpenDDE Harness 不会修改它，也不启停它[/dim]\n"
                    f"      地址       {where}\n"
                    f"      能力       {caps}",
                ),
                highlight=False,
            )
            choice = questionary.select(
                oc._t("Reuse it for OpenDDE Harness's memory?", "要让 OpenDDE Harness 复用它吗？"),
                choices=[
                    questionary.Choice(
                        oc._t(
                            "Reuse (memory and media parsing both use this config)",
                            "复用（记忆和多模态解析都走这一份配置）",
                        ),
                        value="reuse",
                    ),
                    questionary.Choice(
                        oc._t("No, give OpenDDE Harness its own memory", "不用，让 OpenDDE Harness 建一份独立的记忆"),
                        value="own",
                    ),
                ],
                style=OPENDDE_HARNESS_STYLE,
                qmark=oc._QMARK,
            ).ask()
            if choice is None:
                raise typer.Exit(1)
            if choice == "own":
                return _OWN_ROOT_INSTEAD
            _record_root(state.root, owned=False)
            _set_base_url(where)
            _set_memory_backend(BACKEND_NAME)
            oc.console.print(
                oc._t(
                    f"  [green]✓ Recorded: OpenDDE Harness will store and recall memories via {where}.[/green]\n"
                    f"  [dim]You manage this Persistent Memory. To change its models, edit its {CONFIG_FILENAME}\n"
                    "  and restart it -- OpenDDE Harness follows along.[/dim]",
                    f"  [green]✓ 已记录：OpenDDE Harness 将通过 {where} 存取记忆。[/green]\n"
                    f"  [dim]这份 Persistent Memory 由你管理。要调整模型，编辑它的 {CONFIG_FILENAME} 后重启即可，\n"
                    "  OpenDDE Harness 会自动跟随。[/dim]",
                ),
                highlight=False,
            )
            return None

        oc.console.print(
            oc._t(
                f"  [yellow]! Found another Persistent Memory config, but its service is not running[/yellow]\n"
                f"      memory dir  {state.root}   [dim]<- you manage this; "
                f"OpenDDE Harness will not start it[/dim]\n"
                f"      address     {where} [dim](declared in its config, not answering)[/dim]",
                f"  [yellow]⚠ 发现另一份 Persistent Memory 配置，但服务未启动[/yellow]\n"
                f"      记忆目录   {state.root}   [dim]<- 由你管理，OpenDDE Harness 不会替你启动[/dim]\n"
                f"      地址       {where} [dim]（配置中声明，当前无响应）[/dim]",
            ),
            highlight=False,
        )
        choice = questionary.select(
            oc._t("Reuse it?", "要复用它吗？"),
            choices=[
                questionary.Choice(
                    oc._t("I will start it, then probe again", "我去启动它，然后重新探测"),
                    value="retry",
                ),
                questionary.Choice(
                    oc._t("No, give OpenDDE Harness its own memory", "不用，让 OpenDDE Harness 建一份独立的记忆"),
                    value="own",
                ),
            ],
            style=OPENDDE_HARNESS_STYLE,
            qmark=oc._QMARK,
        ).ask()
        if choice is None:
            raise typer.Exit(1)
        if choice == "own":
            return _OWN_ROOT_INSTEAD
        oc.console.print(
            oc._t(
                f"  [dim]Start it in another terminal, for example:[/dim]\n"
                f"      {SERVER_CMDLINE} --root {state.root}",
                f"  [dim]请在另一个终端启动它，例如：[/dim]\n      {SERVER_CMDLINE} --root {state.root}",
            ),
            highlight=False,
        )
        from opendde_harness.plugin.memory.longterm import _discover

        state = _discover._describe(state.root, owned=False)


def _converge_owned_root(state: Any) -> bool:
    """Bring a root opendde owns onto the standard address.

    Returns False when the step should stop (the data is held by something opendde
    cannot identify). Nothing is asked here: the process is opendde's own, Persistent Memory
    replays interrupted strategy runs through crash recovery, and SIGTERM drains
    what is in flight before releasing the lock -- so there is no decision for
    the user to make, only work to report.
    """
    from opendde_harness.plugin.memory.longterm._server import (
        StopOutcome,
        find_recorded_server,
        lock_holder,
        stop_recorded_server,
    )

    target = _configured_target_url()

    if state.serving and _same_address(state.declared_url, target):
        _set_base_url(target)
        return True

    if state.serving and not _same_address(state.declared_url, target):
        # Stopping a healthy service is the one destructive act in this step, so
        # it is the user's call. Adopting is the default because it is the
        # reversible one: the address is recorded, nothing is signalled, and a
        # later run can still move it.
        if _adopt_or_move(state.declared_url, target) == "adopt":
            _adopt_running_address(state.declared_url)
            return True
        outcome = stop_recorded_server(state.root)
        if outcome is not StopOutcome.STOPPED:
            oc.console.print(_stop_failure_line(outcome, keeping=state.declared_url), highlight=False)
            _set_base_url(state.declared_url or target)
            return True
        return _restart_here(state.root, target)

    if state.busy_elsewhere:
        oc.console.print()
        oc.console.print(
            oc._t(
                f"  [yellow]! Something is already serving {state.root}, but not at "
                f"{state.declared_url or 'any declared address'}.[/yellow]\n"
                "  [dim]One memory directory admits one Persistent Memory instance; this is not about "
                "the port.[/dim]",
                f"  [yellow]⚠ 已有进程在使用 {state.root}，但不在它声明的地址 "
                f"{state.declared_url or '（未声明）'} 上。[/yellow]\n"
                "  [dim]一份记忆数据同时只能由一个 Persistent Memory 实例服务，这与端口无关。[/dim]",
            ),
            highlight=False,
        )
        # Ask the OS who has the data before deciding anything. The lock names a
        # holder whatever opendde remembers, and the holder usually names a port
        # -- which turns the one state that used to be a dead end into "it is
        # over there", recoverable without signalling anything.
        holder = lock_holder(state.root)
        if holder is not None and holder.port:
            found_at = f"http://localhost:{holder.port}"
            oc.console.print(
                oc._t(
                    f"  [dim]It is answering at {found_at} (pid {holder.pid}).[/dim]",
                    f"  [dim]它正在 {found_at} 上提供服务（pid {holder.pid}）。[/dim]",
                ),
                highlight=False,
            )
            if _adopt_or_move(found_at, target) == "adopt":
                _adopt_running_address(found_at)
                return True
        elif holder is not None:
            oc.console.print(
                oc._t(
                    f"  [yellow]! pid {holder.pid} holds it but is not serving HTTP:[/yellow]\n"
                    f"  [dim]{holder.cmdline}[/dim]\n"
                    "  [dim]Stop it and re-run `ddeharness onboard`.[/dim]",
                    f"  [yellow]⚠ pid {holder.pid} 占着它，但没有在提供 HTTP 服务：[/yellow]\n"
                    f"  [dim]{holder.cmdline}[/dim]\n"
                    "  [dim]请先停掉它，然后重跑 ddeharness onboard。[/dim]",
                ),
                highlight=False,
            )
            return False
        if find_recorded_server(state.root) is None:
            oc.console.print(
                oc._t(
                    "  [red]x Cannot take over: OpenDDE Harness cannot confirm it started that process.[/red]\n"
                    "  [dim]Stop it and re-run `ddeharness onboard`.[/dim]",
                    "  [red]✗ 无法接管：无法确认占用它的进程是不是 OpenDDE Harness 启动的。[/red]\n"
                    "  [dim]请先停掉它，然后重跑 ddeharness onboard。[/dim]",
                ),
                highlight=False,
            )
            return False
        outcome = stop_recorded_server(state.root)
        if outcome is not StopOutcome.STOPPED:
            # The lock is still held. Walking on to spawn would put a second
            # instance straight into it -- the failure this whole step exists to
            # prevent -- so stop here rather than discard the outcome.
            oc.console.print(_stop_failure_line(outcome, keeping=state.declared_url))
            return False
        return _restart_here(state.root, target)

    # Owned, configured, and simply not running: still converge, or the legacy
    # address survives every future run. ``ensure_memory_server`` would otherwise
    # start a server at the old address and write it straight back into [api].
    return _restart_here(state.root, target)


def _adopt_or_move(running_at: str | None, target: str) -> str:
    """Ask whether to keep the running address or move the service to the target.

    Presented as a question rather than done silently because the two cases the
    wizard cannot tell apart -- a port an old opendde left behind, and a port the
    user deliberately set -- want opposite answers, and only the user knows
    which one this is.
    """
    questionary = oc._require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    oc.console.print()
    oc.console.print(
        oc._t(
            f"  [green]v Found OpenDDE Harness's Persistent Memory service, running at {running_at}[/green]\n"
            f"  [dim]The configured address is {target}.[/dim]",
            f"  [green]✓ 找到 OpenDDE Harness 管理的 Persistent Memory 服务，正在运行于 {running_at}[/green]\n"
            f"  [dim]配置中的地址是 {target}。[/dim]",
        ),
        highlight=False,
    )
    choice = questionary.select(
        oc._t("Which address should OpenDDE Harness use?", "OpenDDE Harness 该用哪个地址？"),
        choices=[
            questionary.Choice(
                oc._t(
                    f"Keep {running_at} (update the setting, nothing restarts)",
                    f"就用 {running_at}（更新配置，不重启）",
                ),
                value="adopt",
            ),
            questionary.Choice(
                oc._t(f"Move it to {target} (restarts the service)", f"迁移到 {target}（会重启服务）"),
                value="move",
            ),
        ],
        style=OPENDDE_HARNESS_STYLE,
        qmark=oc._QMARK,
    ).ask()
    if choice is None:
        raise typer.Exit(1)
    return str(choice)


def _adopt_running_address(running_at: str | None) -> None:
    """Record the address the service is already on as the intended one.

    Both halves matter: ``base_url`` so this session connects, and ``port`` so
    the next run compares against the same intent instead of asking again.
    :func:`_set_base_url` writes both.
    """
    if not running_at:
        return
    _set_base_url(running_at)


def _stop_failure_line(outcome: Any, *, keeping: str | None) -> str:
    """One sentence per stop outcome. Saying the wrong one sends the user
    looking for the wrong thing -- a server draining memory work is not a
    foreign process."""
    from opendde_harness.plugin.memory.longterm._server import StopOutcome

    reason = {
        StopOutcome.NOT_OURS: oc._t(
            "OpenDDE Harness cannot confirm it started this process.",
            "无法确认这个进程是不是 OpenDDE Harness 启动的。",
        ),
        StopOutcome.SIGNAL_FAILED: oc._t("the stop signal could not be delivered.", "停止信号发送失败（权限不足？）。"),
        StopOutcome.STILL_DRAINING: oc._t(
            "it is shutting down but still finishing memory work.",
            "它正在收尾，可能有记忆任务还在跑。",
        ),
    }[outcome]
    tail = oc._t(f"Keeping {keeping}. ", f"继续使用 {keeping}。") if keeping else oc._t("", "")
    return oc._t(
        f"  [yellow]! Could not move it: {reason}[/yellow]\n"
        f"  [dim]{tail}Re-run `ddeharness onboard` once it has stopped.[/dim]",
        f"  [yellow]⚠ 无法迁移：{reason}[/yellow]\n  [dim]{tail}等它停下后重跑 ddeharness onboard 即可。[/dim]",
    )


def _restart_here(root: Any, target: str) -> bool:
    """Start the service at ``target`` and record the address it now serves.

    Convergence is stop -> write -> start, and the last step has to belong to
    the same function as the first two. Leaving it to "whatever runs later"
    meant that a user who picked ``Keep it enabled`` -- the first option, and the
    answer an existing install gives -- left the wizard with the service stopped:
    it had been shut down to move ports and the branch that would have restarted
    it was never reached.
    """
    import asyncio

    from opendde_harness.plugin.memory.longterm._server import ensure_memory_server

    _set_base_url(target)
    oc.console.print(
        oc._t(
            f"  [dim]Starting the service at {target}...[/dim]",
            f"  [dim]正在于 {target} 启动服务...[/dim]",
        )
    )
    try:
        asyncio.run(ensure_memory_server(target))
    except RuntimeError as exc:
        oc.console.print(
            oc._t(
                f"  [red]x Could not start it at {target}: {exc}[/red]",
                f"  [red]✗ 无法在 {target} 启动：{exc}[/red]",
            ),
            highlight=False,
        )
        return False
    oc.console.print(
        oc._t(
            f"  [green]v Persistent Memory service is running at {target}.[/green]",
            f"  [green]✓ Persistent Memory 服务已在 {target} 运行。[/green]",
        )
    )
    return True


def _set_base_url(base_url: str) -> None:
    """Cache the address in opendde's config, and record the port as the intent.

    ``base_url`` is a cache -- the root's config toml is what the server reads,
    and this follows it so the runtime does not have to scan for a root on
    every session. ``port`` is the other half: where a managed server is
    *meant* to listen, which is what convergence compares against.

    Written together on purpose. Left to separate callers, the intent was
    recorded by exactly one branch, so every ordinary converge produced an
    address with no intent beside it and the comparison fell back to the
    shipped constant -- reintroducing the behaviour the intent was added to
    prevent.
    """
    from urllib.parse import urlparse

    from opendde_harness.config.update import set_plugin_config_fields

    fields: dict[str, Any] = {"base_url": base_url}
    port = urlparse(base_url).port
    if port:
        fields["port"] = int(port)
    set_plugin_config_fields(PLUGIN_ID, fields)


_OWN_ROOT_INSTEAD = object()


def _step4_memory(
    *, skip: bool, non_interactive: bool, main_model: Optional[str], warnings: list[str], skip_test: bool = False
) -> object:
    """Step 2 -- Persistent Memory (model sub-screens).

    The bootstrap seeds ``memory.backend`` to the bundled backend (schema default)
    and it is the only memory backend. This step either confirms the seed by configuring
    the llm role, or resolves it back
    to ``None`` on skip / non-interactive / give-up. ``None`` means no long-term
    memory at all, not a fallback to something simpler.

    ``_memory_enabled`` gates on the llm role alone, so a fresh modelless seed
    reads as "not configured yet" and the keep/reconfigure menu only appears once
    that model is actually on disk. Embedding is offered here but never gates:
    skipping it costs semantic recall, not memory itself.
    """
    oc._step_header(2, oc._t("Persistent Memory", "Persistent Memory"))

    import sys

    if sys.platform == "win32":
        oc.console.print(
            oc._t(
                "  [yellow]⚠ Persistent Memory engine does not support native Windows.[/yellow]\n"
                "  [dim]Run OpenDDE Harness inside WSL for full memory support.[/dim]\n"
                "  [dim]Skipping memory configuration.[/dim]",
                "  [yellow]⚠ Persistent Memory 记忆引擎暂不支持 Windows 原生环境。[/yellow]\n"
                "  [dim]在 WSL 中运行 OpenDDE Harness 可获得完整记忆支持。[/dim]\n"
                "  [dim]已跳过记忆配置。[/dim]",
            )
        )
        _set_memory_backend(None)
        return None

    if skip or non_interactive:
        # Never configured the required models here → disable backend-driven
        # memory so runtime doesn't activate Persistent Memory without an llm/embedding.
        # (``_memory_enabled`` already gates on both required models, so an
        # already-enabled+configured setup is preserved.)
        if not _memory_enabled():
            _set_memory_backend(None)
        oc.console.print(
            oc._t(
                "  [dim]Long-term memory stays off.[/dim]\n"
                "  [dim]Run `ddeharness onboard` again whenever you want to configure it.[/dim]",
                "  [dim]长期记忆保持关闭。[/dim]\n  [dim]随时可以重新运行 ddeharness onboard 配置。[/dim]",
            )
        )
        return None

    questionary = oc._require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE
    from opendde_harness.config.update_memory import default_memory_root
    from opendde_harness.plugin.memory.longterm import _discover

    oc.console.print(
        oc._t("  [dim]Looking for an existing memory service...[/dim]", "  [dim]正在查找已有的记忆服务...[/dim]")
    )
    found = _discover.pick(_discover.discover())

    if found is not None and not found.owned:
        # A root the user manages. Read-only from here: record where it is and
        # report what it can do, never configure it and never start or stop it.
        outcome = _reuse_unowned_root(found)
        if outcome is not _OWN_ROOT_INSTEAD:
            return outcome
        # Record the decision here rather than leaving it to the branch that
        # builds the root. Everything downstream -- _memory_enabled(), which
        # reads the recorded root and would still find the user's configured
        # llm, and memory_root() itself -- has to see the new answer, and
        # _memory_enabled() returns from this function before the building
        # branch is ever reached.
        _record_root(default_memory_root(), owned=True)
        found = None

    if found is not None and found.owned:
        _record_root(found.root, owned=True)
        if not _converge_owned_root(found):
            return None

    if found is None and not _memory_enabled():
        source = questionary.select(
            oc._t("Where should long-term memory come from?", "长期记忆从哪来？"),
            choices=[
                questionary.Choice(
                    oc._t(
                        "Let OpenDDE Harness run Persistent Memory for me",
                        "让 OpenDDE Harness 替我运行 Persistent Memory",
                    ),
                    value="managed",
                ),
                questionary.Choice(
                    oc._t(
                        "Skip for now (design still runs; without long-term learning, results may be slightly affected)",
                        "暂时跳过（不影响设计运行；缺少长期经验积累，效果可能略受影响）",
                    ),
                    value="skip",
                ),
            ],
            style=OPENDDE_HARNESS_STYLE,
            qmark=oc._QMARK,
        ).ask()
        if source is None:
            raise typer.Exit(1)
        if source == "skip":
            _set_memory_backend(None)
            return None
        if source == "self":
            if _use_self_managed_memory():
                return None
            # The step ends here. Falling through to the managed path would
            # have walked someone who just said "I run my own Persistent Memory" through
            # configuring four model roles and a key for a setup they did not
            # ask for, and left a root on disk they will not use -- while the
            # line printed a moment earlier told them to start their server and
            # re-run. A refused address is a server not started or a port
            # mistyped, not a change of mind.
            # Nothing printed here: the user picked "skip" one line ago and the
            # option said what that means. Restating it is noise on a screen the
            # step is already leaving.
            _set_memory_backend(None)
            return None

    if _memory_enabled():
        action = questionary.select(
            oc._t(
                "Persistent Memory is already enabled. What would you like to do?",
                "Persistent Memory已启用。想做什么?",
            ),
            choices=[
                questionary.Choice(oc._t("Keep it enabled", "保持启用"), value="keep"),
                questionary.Choice(oc._t("Reconfigure", "重新配置"), value="redo"),
            ],
            style=OPENDDE_HARNESS_STYLE,
            qmark=oc._QMARK,
        ).ask()
        if action is None:
            raise typer.Exit(1)
        if action == "keep":
            return None  # backend already on + models on disk; leave as-is
    else:
        # Wrapped by hand: rich re-wraps at the terminal width and drops the
        # two-space indent on continuation lines, which reads as a stray
        # left-flush sentence under an indented block.
        oc.console.print(
            oc._t(
                "  [dim]Persistent Memory capabilities grow with what you configure:[/dim]\n"
                "  [dim]    memory LLM only    conversations become memories; recall matches keywords[/dim]\n"
                "  [dim]  + memory embedding   recall matches meaning, not wording (strongly advised)[/dim]",
                "  [dim]Persistent Memory 的能力随配置递进：[/dim]\n"
                "  [dim]    仅记忆 LLM       对话会被提炼成记忆存下来，召回按关键词匹配[/dim]\n"
                "  [dim]  + 记忆 embedding   召回按语义匹配，换个问法也能找到（强烈建议配）[/dim]",
            ),
            highlight=False,
        )

    # Ensure the Persistent Memory home directory has its config templates (the
    # config toml + ome.toml) BEFORE writing model sections — set_memory_section merges
    # into the template so default sections (memory/sqlite/lancedb/api) are
    # preserved. Also creates ome.toml which the runtime requires.
    from opendde_harness.config.update_memory import configure_memory_env, ensure_memory_home, owned_memory_root

    # owned_memory_root, not memory_root: after a user declined to share theirs,
    # the recorded root is still theirs, and building there would adopt it.
    root = owned_memory_root()
    _record_root(root, owned=True)
    configure_memory_env(root)
    ensure_memory_home(root)

    # Only when the default is taken. A port screen on every fresh install
    # would be a decision almost nobody has to make; the case that had no way
    # forward is the occupied one, where the start simply failed and the wizard
    # had nowhere to send the user.
    _set_base_url(f"http://localhost:{_ask_managed_port(root)}")

    # Configure required models FIRST, then flip the backend on — so a Ctrl+C
    # mid-configuration leaves backend at its prior (disabled) value rather
    # than an enabled-but-modelless state.
    for _role in ("llm", "embedding"):
        # Each role prints one leading blank before its own header, so no extra
        # separator here — avoids the double blank line between roles.
        outcome = _config_memory_role(
            section=_role,
            main_model=main_model,
            non_interactive=non_interactive,
            warnings=warnings,
            skip_test=skip_test,
        )
        if outcome is oc._ABORT_MEMORY:
            _set_memory_backend(None)
            oc.console.print(
                oc._t(
                    "  [yellow]⚠ Gave up long-term memory: OpenDDE Harness will not remember anything "
                    "between sessions.[/yellow]\n"
                    "  [dim]Run `ddeharness onboard` again whenever you want to configure it.[/dim]",
                    "  [yellow]⚠ 已放弃长期记忆，OpenDDE Harness 不会记住任何跨会话内容。[/yellow]\n"
                    "  [dim]随时可以重新运行 ddeharness onboard 配置。[/dim]",
                )
            )
            return None

    # Verify Persistent Memory server is reachable (auto-starts if needed)
    import asyncio

    from opendde_harness.config.opendde_harness import load_opendde_harness_config
    from opendde_harness.plugin.memory.longterm._health import configured_base_url
    from opendde_harness.plugin.memory.longterm._server import ensure_memory_server

    # The configured address, not the default: the memory backend connects to
    # whatever ``plugins.config`` names, so probing 18791 on a setup that moved
    # the server elsewhere reports on a server nobody uses -- and then spawns a
    # second instance that cannot hold the OME lock.
    base_url = configured_base_url(load_opendde_harness_config())

    # The models were just written; a process already running booted with the
    # old ones and will not re-read them.
    _stop_for_reload(root)

    oc.console.print()
    oc.console.print(
        oc._t(
            "  [dim]Starting Persistent Memory service...[/dim]",
            "  [dim]正在启动 Persistent Memory 服务...[/dim]",
        )
    )
    # A failed start is not a decision to abandon long-term memory. The models
    # are already on disk at this point, so "defer" keeps the whole
    # configuration and lets the runtime start the service on the next session;
    # only the explicit third choice turns memory off.
    while True:
        try:
            asyncio.run(ensure_memory_server(base_url))
            oc.console.print(
                oc._t(
                    "  [green]✓ Persistent Memory service is running.[/green]",
                    "  [green]✓ Persistent Memory 服务已启动。[/green]",
                )
            )
            break
        except RuntimeError as exc:
            oc.console.print(
                oc._t(
                    f"  [red]✗ Persistent Memory service failed to start: {exc}[/red]",
                    f"  [red]✗ Persistent Memory 服务启动失败：{exc}[/red]",
                )
            )
            action = questionary.select(
                oc._t("What to do?", "怎么办？"),
                choices=[
                    questionary.Choice(oc._t("Retry", "重试"), value="retry"),
                    questionary.Choice(
                        oc._t(
                            "Leave it for later (settings kept, OpenDDE Harness retries next start)",
                            "暂时跳过（保留配置，下次启动 OpenDDE Harness 时会再试）",
                        ),
                        value="defer",
                    ),
                    questionary.Choice(
                        oc._t("Turn long-term memory off", "关闭长期记忆"),
                        value="disable",
                    ),
                ],
                style=OPENDDE_HARNESS_STYLE,
                qmark=oc._QMARK,
            ).ask()
            if action is None:
                raise typer.Exit(1) from exc
            if action == "retry":
                continue
            if action == "defer":
                _set_memory_backend(BACKEND_NAME)
                oc.console.print(
                    oc._t(
                        "  [yellow]! Memory settings kept. OpenDDE Harness will try to start the "
                        "service again on the next session.[/yellow]",
                        "  [yellow]⚠ 已保留记忆配置。下次会话启动时 OpenDDE Harness 会再尝试启动服务。[/yellow]",
                    )
                )
                return None
            _set_memory_backend(None)
            oc.console.print(
                oc._t(
                    "  [yellow]! Long-term memory turned off. Run `ddeharness onboard` "
                    "again whenever you want it back.[/yellow]",
                    "  [yellow]⚠ 已关闭长期记忆。随时可以重新运行 ddeharness onboard 开启。[/yellow]",
                )
            )
            return None
    _report_memory_capabilities()
    _set_memory_backend(BACKEND_NAME)
    return None


def _report_memory_capabilities() -> None:
    """Say what the running server can actually do, not just that it answers.

    ``ensure_memory_server`` proves the process is up and nothing more. Since
    library 1.2.1 a server whose embedding provider failed to build still answers
    200 and degrades to keyword-only search, so stopping at "running" would
    print a tick over an install that cannot recall anything. The roles were
    each verified against their provider earlier in this step; what is new here
    is whether the library itself could build them from what got written to
    the config toml.

    Silent on a server too old to report capabilities -- reading that silence as
    "unavailable" would condemn a working install.
    """
    from opendde_harness.config.opendde_harness import load_opendde_harness_config
    from opendde_harness.plugin.memory.longterm._health import (
        DEGRADING_SECTIONS,
        REQUIRED_SECTIONS,
        configured_base_url,
        probe_capabilities,
    )

    # The configured address, not the default: probing the wrong port reports on
    # a server nobody is using, and reads as "not running".
    report = probe_capabilities(configured_base_url(load_opendde_harness_config()))
    if not report.reports_capabilities:
        return
    configured = [s for s in (*REQUIRED_SECTIONS, *DEGRADING_SECTIONS) if _memory_role_configured(s)]
    broken = [s for s in configured if report.available(s) is False]
    if not broken:
        names = " and ".join(configured)
        oc.console.print(
            oc._t(
                f"  [green]✓ {names} {'is' if len(configured) == 1 else 'are'} available.[/green]",
                f"  [green]✓ {names} 均可用。[/green]",
            )
        )
        return
    names = " and ".join(broken)
    oc.console.print(
        oc._t(
            f"  [yellow]⚠ {names} is configured but Persistent Memory could not build it.[/yellow]\n"
            "  [dim]Memory runs degraded until this is fixed.[/dim]\n"
            f"  [dim]Check: {_memory_server_log_hint()}[/dim]",
            f"  [yellow]⚠ {names} 已配置，但 Persistent Memory 未能构建成功。[/yellow]\n"
            "  [dim]在此修复前，记忆能力将处于降级状态。[/dim]\n"
            f"  [dim]请查看：{_memory_server_log_hint()}[/dim]",
        )
    )


def _memory_server_log_hint() -> str:
    from opendde_harness.plugin.memory.longterm._server import server_log_path

    return str(server_log_path())
