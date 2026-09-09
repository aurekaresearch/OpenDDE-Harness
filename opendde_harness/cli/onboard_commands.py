"""Four-step onboarding wizard: LLM provider → memory → protein design → import.

Goal: get a new user from ``pip install`` to a working agent in a few
minutes, without ever opening ``~/.opendde_harness/config.json`` or
the memory root's config toml.

Steps (mirrors ``my_docs/temp/onboard-flow.mermaid``):
  0. Welcome
  1. LLM provider (required; multi-provider, in-step connectivity + test probe)
  2. Persistent Memory (optional; llm required once enabled,
     rerank/multimodal optional)
  3. Protein Design compute and folding configuration
  4. Cold-start import from other AI tools (optional)
  5. Done

All writes go through the ``update_providers`` / ``update`` /
``update_memory`` ops libraries — this module owns the UX layer,
not config-schema knowledge.

Navigation: questionary 2.1.1 has no first-class cross-screen "back", so the
wizard is a screen state machine and back is expressed as a ``0) back``
sentinel choice on the screens that support it (Step 1 <-> language pick);
Steps 2 through 4 are optional (re-run
``onboard`` to change them). Ctrl+C exits at any point, keeping whatever was
already written.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from opendde_harness.cli import onboard_memory
from opendde_harness.cli._helpers import (
    DEFAULT_PROBE_MESSAGE,
    print_probe_troubleshooting,
    send_probe,
)
from opendde_harness.cli._theme import POINTER, QMARK
from opendde_harness.providers.registry import (
    CRED_ENDPOINT,
    CRED_LOCAL,
    CRED_OAUTH,
    credential_kind,
)
from opendde_harness.providers.wire import stored_model_id


class _ThemedConsole(Console):
    """Console that applies the light/dark theme on its first render.

    Theming is deferred to the first ``print`` (never at import) so plain
    ``opendde ...`` commands don't detect or probe the terminal. Because every
    onboard render goes through ``print`` on this instance, there is no
    "push the theme before rendering" ordering constraint and no dependence on
    which entry point (wizard, ``ddeharness deep-research enable``, ...) ran first.
    """

    _themed: bool = False

    def print(self, *args: Any, **kwargs: Any) -> None:
        if not self._themed:
            from opendde_harness.cli._theme import build_rich_theme, detect_scheme

            self.push_theme(build_rich_theme(detect_scheme()))
            self._themed = True
        super().print(*args, **kwargs)


console = _ThemedConsole()

_TOTAL_STEPS = 3

# Sentinel returned by a screen function to ask the runner to go back one
# screen; ``None`` from a picker means Ctrl+C (exit).
_BACK = object()
_MANUAL_MODEL = object()

# Sentinel a required memory role returns when the user chooses to give up long-term memory
# rather than configure it; ``_step4_memory`` then leaves memory disabled.
_ABORT_MEMORY = object()

# Unified prompt chrome (display-only), shared with every other command's
# prompts: a single-space qmark renders as one blank, which -- with
# questionary's own leading space -- puts every prompt line on the same 2-space
# column as our printed help/status lines, so the left edge stays flush instead
# of jittering between 1- and 2-space indents.
_QMARK = QMARK
_POINTER = POINTER

# UI language, chosen on the wizard's first screen. ``_t`` returns the English
# or Chinese variant so every later prompt / message stays bilingual.
_LANG = "en"


def _t(en: str, zh: str) -> str:
    """Return ``zh`` when the user picked Chinese, else ``en``."""
    return zh if _LANG == "zh" else en


# ---------------------------------------------------------------------------
# Curated provider catalogue surfaced in Step 1's picker.
# ---------------------------------------------------------------------------


# Sentinel entries the picker renders as its own step rather than a provider.
_PICK_LITELLM_VENDOR = "__litellm_vendor__"

# Providers offered during onboarding, grouped by authentication method.
_CURATED_GROUPS: list[dict[str, Any]] = [
    {
        "kind": "api_key",
        "providers": [
            {
                "name": "openrouter",
                "label": "OpenRouter",
                "label_zh": "OpenRouter",
            },
            {"name": "openai", "label": "OpenAI", "label_zh": "OpenAI"},
            {"name": "anthropic", "label": "Anthropic", "label_zh": "Anthropic"},
            {"name": "gemini", "label": "Gemini", "label_zh": "Gemini"},
            {
                "name": "minimax",
                "label": "MiniMax",
                "label_zh": "MiniMax",
            },
            {"name": "deepseek", "label": "DeepSeek", "label_zh": "DeepSeek"},
            {"name": "zai", "label": "GLM", "label_zh": "GLM"},
            {"name": "dashscope", "label": "DashScope", "label_zh": "阿里云百炼"},
            {"name": "moonshot", "label": "Kimi", "label_zh": "Kimi"},
            {"name": "volcengine", "label": "VolcEngine", "label_zh": "火山方舟"},
            {"name": "siliconflow", "label": "SiliconFlow", "label_zh": "硅基流动"},
            {"name": "groq", "label": "Groq", "label_zh": "Groq"},
            {"name": "aihubmix", "label": "AiHubMix", "label_zh": "AiHubMix"},
            {"name": "azure_openai", "label": "Azure OpenAI", "label_zh": "Azure OpenAI"},
        ],
    },
    {
        "kind": "oauth",
        "providers": [
            {
                "name": "github_copilot",
                "label": "GitHub Copilot (OAuth)",
                "label_zh": "GitHub Copilot(OAuth 登录)",
            },
            {"name": "openai_codex", "label": "OpenAI Codex (OAuth)", "label_zh": "OpenAI Codex(OAuth 登录)"},
        ],
    },
    {
        "kind": "local",
        "providers": [
            {"name": "ollama_chat", "label": "Ollama (local)", "label_zh": "Ollama(本地)"},
            {"name": "hosted_vllm", "label": "vLLM / self-hosted", "label_zh": "vLLM / 自托管"},
        ],
    },
    {
        "kind": "fallback",
        "providers": [
            {
                "name": _PICK_LITELLM_VENDOR,
                "label": "Another supported vendor (type to search)",
                "label_zh": "其他支持的厂商(输入可搜索)",
            },
            {
                "name": "custom",
                "label": "Self-hosted OpenAI-compatible endpoint",
                "label_zh": "自建 OpenAI 兼容端点",
            },
        ],
    },
]

# Flat view for callers that only need "which providers does the wizard offer".
_CURATED_PROVIDERS: list[dict[str, Any]] = [
    entry for group in _CURATED_GROUPS for entry in group["providers"] if entry["name"] != _PICK_LITELLM_VENDOR
]

_QUESTIONARY_INSTALL_HINT = (
    "[red]Missing dependency:[/red] [accent]questionary[/accent] is required for "
    "interactive onboarding.\n"
    "Install it with: [accent]uv add 'questionary>=2.0,<3.0'[/accent]\n"
    "Or re-run with [accent]--non-interactive[/accent] plus the relevant flags."
)


_PROMPT_THEMED = False


def _theme_questionary(questionary: Any) -> None:
    """Give every ``select`` a consistent pointer and drop questionary's own
    "(Use arrow keys)" hint — the step header already prints the controls.

    Display-only and applied once: we wrap ``questionary.select`` so callers
    that don't pass ``pointer`` / ``instruction`` inherit the unified look,
    while any explicit value still wins (``setdefault``).
    """
    global _PROMPT_THEMED
    if _PROMPT_THEMED:
        return
    import functools

    _orig_select = questionary.select

    @functools.wraps(_orig_select)
    def _themed_select(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("pointer", _POINTER)
        # questionary shows "(Use arrow keys)" when instruction is falsy; a
        # single space is truthy yet visually blank, so it hides that hint
        # (the step header already prints the controls).
        kwargs.setdefault("instruction", " ")
        return _orig_select(*args, **kwargs)

    questionary.select = _themed_select
    _PROMPT_THEMED = True


def _require_questionary() -> Any:
    """Lazy-import :mod:`questionary` so missing-package errors stay scoped here."""
    try:
        import questionary
    except ModuleNotFoundError:
        console.print(_QUESTIONARY_INSTALL_HINT)
        raise typer.Exit(1)
    _theme_questionary(questionary)
    return questionary


def _config_language() -> str:
    """Read the saved UI language from the on-disk config ('en' / 'zh').

    A missing / empty config (fresh install) defaults to 'en'; a malformed one
    raises ConfigReadError (surfaced by the CLI entrypoint) rather than being
    silently read as empty.
    """
    data = _load_raw_config()
    lang = data.get("language")
    return lang if lang in ("en", "zh") else "en"


def _pick_language() -> None:
    """First screen: choose the wizard's language. Updates module-level ``_LANG``.

    Persistence happens later (after bootstrap created the config file), via
    ``set_language`` in :func:`_run_wizard_body`.
    """
    global _LANG
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    # Framed like the other screens (bilingual, since no language is chosen yet)
    # so it reads as the wizard's first step, not a bare floating list.
    console.print()
    console.print(
        Panel(
            "[heading]Let's set up OpenDDE Harness — first, choose your language.[/heading]\n"
            "[dim]开始配置 OpenDDE Harness — 请先选择语言。[/dim]",
            title="[bold][accent]OpenDDE Harness setup[/accent][/bold]",
            title_align="left",
            border_style="border",
            padding=(1, 2),
        )
    )
    console.print("  [dim]↑↓ select · Enter confirm · Ctrl+C quit[/dim]")
    console.print()

    picked = questionary.select(
        "Language / 语言",
        choices=[
            questionary.Choice("English", value="en"),
            questionary.Choice("中文(简体)", value="zh"),
        ],
        default=_LANG,  # preselect the saved language on a re-run
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
    ).ask()
    if picked is None:
        raise typer.Exit(1)
    _LANG = picked


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _step_header(n: int, title: str) -> None:
    # Progress dots: filled for done/current steps, hollow for upcoming ones.
    dots = " ".join("[accent]●[/accent]" if i <= n else "[grey37]○[/grey37]" for i in range(1, _TOTAL_STEPS + 1))
    console.print()
    console.print(
        Panel(
            f"[heading]{title}[/heading]",
            title=f"[bold][accent]{_t('Step', '步骤')} {n}/{_TOTAL_STEPS}[/accent][/bold]",
            title_align="left",
            subtitle=dots,
            subtitle_align="right",
            border_style="border",
            padding=(0, 2),
        )
    )
    console.print()  # breathing room between the header and the step's prompts


def _check_tty_or_die(non_interactive: bool) -> None:
    """Bail when stdout isn't a TTY and the user didn't opt into headless mode."""
    if non_interactive:
        return
    if not sys.stdout.isatty():
        console.print(
            "[red]Non-interactive terminal detected.[/red]\n"
            "Re-run with: "
            "[accent]ddeharness onboard --non-interactive --provider <name> --api-key <key>[/accent]"
        )
        raise typer.Exit(2)


def _load_raw_config() -> dict[str, Any]:
    """Return the parsed on-disk config, or ``{}`` if absent/empty.

    A present-but-unparseable config raises ConfigReadError (surfaced cleanly by
    the CLI entrypoint) instead of being silently treated as empty -- which
    would let onboard misread state and write over a config whose only fault is
    a syntax typo.
    """
    from opendde_harness.config.loader import get_config_path, read_raw_or_raise

    return read_raw_or_raise(get_config_path()) or {}


def _configured_providers() -> list[str]:
    """Names of providers with a usable API key or OAuth token."""
    from opendde_harness.config.update_providers import list_providers

    return [row["name"] for row in list_providers() if row["configured"]]


def _is_config_populated() -> bool:
    """True iff the provider that serves the configured model has its credentials.

    "Populated" for the startup gate means the required step (Step 1) is
    satisfied: a default model plus credentials for whoever answers it. Either
    alone is not enough to talk to a model.

    Which provider answers is not re-derived here. The config already resolves
    it -- honoring an explicit ``agents.defaults.provider``, then prefix over
    keyword, and declining to fall back to an OAuth provider -- and a second
    derivation from the model-id prefix is how this gate came to disagree with
    ``ddeharness status`` about a signed-in provider. What is left to ask is whether
    that provider's credentials are actually on disk, which is the one thing the
    resolver takes on trust for the OAuth families.
    """
    from opendde_harness.config.loader import load_config

    data = _load_raw_config()
    model = (data.get("agents", {}) or {}).get("defaults", {}).get("model")
    if not model:
        return False

    try:
        serving = load_config().get_provider_name(str(model))
    except Exception:
        # A config too damaged to resolve is not a configured one, and the wizard
        # is a better answer here than a traceback. The raw read above already
        # raised on a syntax error, so this is the semantic case.
        return False

    return bool(serving and serving in _configured_providers())


def _handle_existing_config(*, reset: bool, yes: bool, non_interactive: bool) -> None:
    """Guard against silently overwriting an existing config in non-interactive
    runs.

    Interactive runs always fall through into the structured wizard: every step
    defaults to "Keep current" for already-set values, so pressing Enter all the
    way through is equivalent to skipping, and changing any value reconfigures
    just that one. No separate skip/redo/quit screen — it would drop the wizard's
    welcome banner and step framing.
    """
    if reset:
        return
    if not _is_config_populated():
        return

    if non_interactive:
        if yes:
            console.print("[dim]Existing config detected; --yes set, proceeding with overwrite.[/dim]")
            return
        console.print(
            "[red]Existing config detected.[/red] Pass [accent]--reset[/accent] (or "
            "[accent]--yes[/accent]) to overwrite, or edit in place with "
            "[accent]ddeharness provider set[/accent]."
        )
        raise typer.Exit(2)
    # Interactive: fall through to the wizard (per-step "Keep current" handles
    # the existing config gracefully).


def _bootstrap_empty_config() -> None:
    """Make sure ``~/.opendde_harness/config.json`` + workspace dir exist before we patch.

    We seed the user-facing extension defaults (memory / plugins / skillForge),
    including ``memory.backend = "longterm"`` (the schema default). The memory service
    degrades gracefully when its models aren't configured yet (empty recall + a
    warning, never a crash), so an enabled-but-modelless install is safe. The
    wizard's Step 2 — and its skip / non-interactive guard — resolve the backend
    back to ``None`` when the user opts out or never configures the one required
    model (``_memory_enabled`` gates on the llm role being present, not just the
    backend name; embedding and rerank only cost recall quality).

    Seeding runs on EVERY onboard, not just a brand-new config: the writer is
    ``setdefault``-based (non-clobbering), so it backfills these blocks into a
    pre-existing config that predates them without touching any value the user
    already set. The base ``Config()`` is only written when the file is absent —
    overwriting an existing file there would clobber it.
    """
    from opendde_harness.config.loader import get_config_path, load_config, save_config
    from opendde_harness.config.paths import get_workspace_path
    from opendde_harness.utils.helpers import sync_workspace_templates

    path = get_config_path()
    if not path.exists():
        save_config(load_config())  # writes default Config() to disk
    onboard_memory._init_extension_block_defaults()
    workspace = get_workspace_path()
    workspace.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(workspace)


# ---------------------------------------------------------------------------
# Step 1 — provider primitives (reused verbatim from the 3-step wizard)
# ---------------------------------------------------------------------------


def _provider_label(name: str) -> str:
    """Display label for a provider, falling back to the registry's display_name."""
    for entry in _CURATED_PROVIDERS:
        if entry["name"] == name:
            return _t(entry["label"], entry.get("label_zh", entry["label"]))
    try:
        from opendde_harness.providers.registry import find_by_name

        spec = find_by_name(name)
        return spec.label if spec else name
    except Exception:
        return name


def _validate_provider_name(name: str) -> str:
    """Resolve a user-supplied provider name (kebab or snake) to a registry key.

    A vendor LiteLLM routes to but OpenDDE Harness carries no spec for is configurable
    too: the wizard has no default model or OAuth flow to offer it, but it does
    not need one -- the credentials go in under the vendor's name and the model
    list comes from the vendor itself. Callers must therefore treat the spec as
    optional metadata, not as permission.
    """
    from opendde_harness.config.update_providers import provider_field_specs
    from opendde_harness.providers.registry import find_by_name, normalize_provider_name

    candidate = name.replace("-", "_")
    try:
        provider_field_specs(candidate)
    except KeyError as exc:
        raise typer.BadParameter(str(exc))
    spec = find_by_name(candidate)
    if spec is None:
        # No spec, but provider_field_specs above already confirmed LiteLLM
        # routes to it, which is all configuring it takes.
        return normalize_provider_name(candidate)
    # Return the current name: everything downstream compares and stores by it,
    # and a former name would have the wizard reading one section and writing
    # another.
    return spec.name


def _back_placeholder(allow_back: bool, label: Optional[str] = None) -> Any:
    """A faint in-field placeholder telling the user what an empty submit does.

    Rendered greyed inside the input (via prompt_toolkit's ``placeholder``),
    it disappears the moment they type and leaves nothing behind once the
    prompt is answered. Returns ``None`` when back isn't offered. ``label``
    overrides the default "go back" wording for prompts where an empty submit
    means something else (e.g. cancelling rather than rewinding a step).
    """
    if not allow_back:
        return None
    return [("fg:#6c6c6c italic", label or _t("empty ↵ to go back", "留空回车返回上一步"))]


def _collect_fields(prompts: list[Callable[[], Any]]) -> Optional[list[Any]]:
    """Run text-prompt callables in order with empty-submit = back.

    Each callable prompts one field and returns its value, or ``_BACK`` (an
    empty submit) to rewind one field. Backing out of the first field returns
    ``None`` so the caller can rewind to the preceding screen. Returns the list
    of collected values on success.
    """
    values: list[Any] = []
    i = 0
    while i < len(prompts):
        value = prompts[i]()
        if value is _BACK:
            if i == 0:
                return None
            values.pop()
            i -= 1
            continue
        if i < len(values):
            values[i] = value
        else:
            values.append(value)
        i += 1
    return values


def _select_provider_row() -> Optional[str]:
    """Render the grouped provider list once and return the raw choice.

    Separate from `_select_provider` so backing out of the vendor sub-list can
    show this list again rather than unwinding the whole step.
    """
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    choices: list[Any] = []
    for group in _CURATED_GROUPS:
        # A rule between groups, so the API-key providers, the OAuth ones, the
        # local deployments and the two fallbacks read as four decisions rather
        # than one list of twenty.
        if choices:
            choices.append(questionary.Separator())
        for entry in group["providers"]:
            choices.append(
                questionary.Choice(
                    _t(entry["label"], entry.get("label_zh", entry["label"])),
                    value=entry["name"],
                )
            )
    choices.append(questionary.Separator())
    choices.append(questionary.Choice(_t("Back", "返回"), value=_BACK))

    return questionary.select(
        _t("Provider:", "服务商:"),
        choices=choices,
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
    ).ask()  # None on Ctrl+C


def _select_provider() -> Optional[str]:
    """Interactive provider picker built from the curated catalogue.

    Returns the provider name, ``_BACK`` if the user chose the back sentinel,
    or ``None`` on Ctrl+C.
    """
    picked = _select_provider_row()
    while picked == _PICK_LITELLM_VENDOR:
        # Second step rather than a hundred more rows: LiteLLM routes to far more
        # vendors than anyone wants to scroll, and typing the name is how someone
        # who already knows which one they want gets there.
        #
        # Backing out of it returns to this list, not out of the step: the user
        # opened a sub-list, so empty-submit means "close the sub-list". Passing
        # its _BACK straight up sent them to the language screen instead.
        typed = _prompt_litellm_vendor()
        if typed is None:
            return None
        if typed is not _BACK:
            return typed
        picked = _select_provider_row()
    return picked  # _BACK on back, None on Ctrl+C


def _litellm_vendor_choices() -> list[str]:
    """Vendor names for the second step: the ones the picker does not already show.

    Read from the packaged snapshot rather than LiteLLM itself, so offering them
    costs no import on a path that only renders choices.
    """
    from opendde_harness.providers.litellm_provider_names import LITELLM_PROVIDER_NAMES
    from opendde_harness.providers.registry import find_by_name, normalize_provider_name

    # Every name a listed provider answers to, not just the one shown: LiteLLM
    # knows "ollama" and "vllm", which are the pre-rename spellings of two rows
    # already on the list, so matching on the displayed name alone offered them
    # a second time under a name that resolves to the same section.
    already_listed: set[str] = set()
    for entry in _CURATED_PROVIDERS:
        spec = find_by_name(entry["name"])
        already_listed |= set(spec.route_names) if spec else {normalize_provider_name(entry["name"])}
    return sorted(n for n in LITELLM_PROVIDER_NAMES if normalize_provider_name(n) not in already_listed)


def _prompt_litellm_vendor() -> Optional[str]:
    """Ask for a vendor by name, completing against the ones LiteLLM routes to.

    Returns the provider name, ``_BACK`` to rewind to the picker, or ``None`` on
    Ctrl+C. The names come from the packaged snapshot, so offering them costs no
    LiteLLM import.
    """
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE
    from opendde_harness.providers.registry import normalize_provider_name

    choices = _litellm_vendor_choices()

    typed = questionary.autocomplete(
        _t(
            f"Vendor name ({len(choices)} supported - type to search, Tab to complete, empty to go back):",
            f"厂商名(支持 {len(choices)} 家 — 输入可搜索,Tab 补全,留空返回):",
        ),
        choices=choices,
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
        ignore_case=True,
        match_middle=True,
    ).ask()
    if typed is None:
        return None
    typed = typed.strip()
    if not typed:
        return _BACK
    # Validation happens where the name is used, not here: the caller runs it
    # through the same gate the --provider flag goes through, which is what turns
    # a typo into a message instead of a traceback.
    return normalize_provider_name(typed)


def _prompt_api_key(provider: str, *, allow_back: bool = False, back_label: Optional[str] = None) -> Any:
    """Ask for an API key (hidden input). Returns ``_BACK`` on empty submit
    when ``allow_back`` is set, else the key string. ``back_label`` overrides
    the empty-submit hint for callers where it cancels rather than rewinds."""
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    def _validate(v: str) -> Any:
        if allow_back and v == "":
            return True  # a truly-empty submit is the back/cancel signal
        return (
            True
            if len(v.strip()) >= 8
            else _t(
                "API key looks off (empty or too short) — please re-enter (≥ 8 chars).",
                "API Key 看起来不对(过短或为空),请重新输入(至少 8 位)。",
            )
        )

    key = questionary.password(
        _t("Paste your API key:", "粘贴你的 API Key:"),
        validate=_validate,
        placeholder=_back_placeholder(allow_back, back_label),
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
    ).ask()
    if key is None:
        raise typer.Exit(1)
    key = key.strip()
    if allow_back and key == "":
        return _BACK
    if not key:
        raise typer.Exit(1)
    return key


def _prompt_local_api_base(spec: Any, *, current: str = "", allow_back: bool = False) -> Any:
    """Ask a local deployment for its server URL. Returns ``_BACK`` on empty submit.

    A local deployment is reached by address, not by key -- there is nothing to
    authenticate against a server the user is running.

    The field is seeded with the address already configured, falling back to the
    registry default for a first-time setup. Seeding the default unconditionally
    meant reconfiguring a server at some other address offered localhost, and
    pressing Enter to move on replaced a working address with it.
    """
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    def _validate(v: str) -> Any:
        if allow_back and v.strip() == "":
            return True
        return (
            True
            if v.strip().startswith(("http://", "https://"))
            else _t("URL must start with http:// or https://", "地址需以 http:// 或 https:// 开头")
        )

    url = questionary.text(
        _t(f"{spec.label} server URL:", f"{spec.label} 服务地址:"),
        default=current or spec.default_api_base or "",
        validate=_validate,
        placeholder=_back_placeholder(allow_back),
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
    ).ask()
    if url is None:
        # Ctrl+C quits, like the sibling credential prompts. Returning None left
        # each caller to decide what it meant, and they did not agree.
        raise typer.Exit(1)
    url = url.strip()
    if allow_back and not url:
        return _BACK
    return url


def _prompt_base_url(default: str = "https://", *, allow_back: bool = False) -> Any:
    """Ask for an OpenAI-compatible base URL (used by the 'custom' provider).
    Returns ``_BACK`` on empty submit when ``allow_back`` is set."""
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    # With back enabled, don't seed a default — an empty field must be reachable
    # so the user can submit nothing to rewind.
    seed = "" if allow_back else default

    def _validate(v: str) -> Any:
        if allow_back and v == "":
            return True
        return (
            True
            if v.startswith(("http://", "https://"))
            else _t("URL must start with http:// or https://", "地址需以 http:// 或 https:// 开头")
        )

    url = questionary.text(
        _t("Base URL (must include /v1):", "Base URL(需包含 /v1):"),
        default=seed,
        validate=_validate,
        placeholder=_back_placeholder(allow_back),
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
    ).ask()
    if url is None:
        raise typer.Exit(1)
    url = url.strip()
    if allow_back and url == "":
        return _BACK
    if not url:
        raise typer.Exit(1)
    return url


def _prompt_custom_model(*, allow_back: bool = False) -> Any:
    """Select manual model entry for an endpoint without a model catalogue."""
    return _select_model_id([], allow_back=allow_back)


def _select_model_id(
    models: list[str],
    *,
    default_model: Optional[str] = None,
    allow_back: bool = False,
    label: Optional[str] = None,
    provider: Optional[str] = None,
    manual_first: bool = False,
) -> Any:
    """Show an expanded, searchable model list with explicit manual entry.

    Rows are titled with the vendor's own id: the provider was chosen one step
    earlier, so prefixing every row with it says nothing. The value behind each
    row is the qualified id that gets stored.
    """
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE
    from opendde_harness.providers.wire import display_model_id

    models = list(dict.fromkeys(model.strip() for model in models if model.strip()))
    selected_default = default_model if default_model in models else next(iter(models), _MANUAL_MODEL)
    listed = [
        questionary.Choice(display_model_id(provider, model) if provider else model, value=model) for model in models
    ]
    manual = questionary.Choice(_t("Enter a model name", "手动输入模型名称"), value=_MANUAL_MODEL)
    choices: list[Any] = [manual, *listed] if manual_first else [*listed, manual]
    if allow_back:
        choices.append(questionary.Choice(_t("Back", "返回"), value=_BACK))

    while True:
        chosen = questionary.select(
            label or _t(f"Select a model ({len(models)} available):", f"模型(共 {len(models)} 个):"),
            choices=choices,
            default=selected_default,
            use_search_filter=True,
            use_jk_keys=False,
            instruction=_t("(↑/↓ to select, type to filter, Enter to confirm)", "(↑/↓ 选择，输入筛选，回车确认)"),
            style=OPENDDE_HARNESS_STYLE,
            qmark=_QMARK,
        ).ask()
        if chosen is None:
            raise typer.Exit(1)
        if chosen is _BACK:
            return _BACK
        if chosen is not _MANUAL_MODEL:
            return chosen.strip()
        chosen = questionary.text(
            _t("Model name:", "模型名称:"),
            default=default_model or "",
            placeholder=_back_placeholder(True, _t("empty ↵ to return to the list", "留空回车返回模型列表")),
            style=OPENDDE_HARNESS_STYLE,
            qmark=_QMARK,
        ).ask()
        if chosen is None:
            raise typer.Exit(1)
        if chosen.strip():
            return chosen.strip()


def _run_oauth_login(provider: str) -> bool:
    """Dispatch the OAuth login handler registered by ``provider_commands``.

    Returns ``True`` on success. A login that fails (the handler raises
    ``typer.Exit`` or any error) returns ``False`` so the caller can offer a
    retry / back menu instead of tearing the whole wizard down. A genuine
    Ctrl+C (``KeyboardInterrupt``) is left to propagate as a quit.
    """
    from opendde_harness.cli.provider_commands import _LOGIN_HANDLERS
    from opendde_harness.providers.registry import find_by_name

    spec = find_by_name(provider)
    if credential_kind(provider) != CRED_OAUTH:
        console.print(
            _t(
                f"  [red]✗ {provider} is not an OAuth provider.[/red]",
                f"  [red]✗ {provider} 不是 OAuth 服务商。[/red]",
            )
        )
        raise typer.Exit(1)
    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(
            _t(
                f"  [red]✗ No login handler registered for {provider}.[/red]",
                f"  [red]✗ 未为 {provider} 注册登录处理器。[/red]",
            )
        )
        raise typer.Exit(1)
    console.print(
        _t(
            f"  [accent]Starting OAuth login for {spec.label}…[/accent]\n",
            f"  [accent]正在为 {spec.label} 启动 OAuth 登录…[/accent]\n",
        )
    )
    console.print(
        _t(
            "  [dim]A browser window / link will open — finish the sign-in there, "
            "then come back here. This waits until you're done.[/dim]\n",
            "  [dim]会打开浏览器窗口 / 链接 — 在那里完成登录后回到这里;这里会一直等到你完成。[/dim]\n",
        )
    )
    try:
        handler()
    except typer.Exit as exc:
        # Handlers signal a failed login with Exit(1); Exit(0) (if any) is success.
        if exc.exit_code:
            return False
    except Exception as exc:  # network / browser / token errors — recoverable
        console.print(
            _t(
                f"  [yellow]✗ Login didn't complete: {exc}[/yellow]",
                f"  [yellow]✗ 登录未完成:{exc}[/yellow]",
            )
        )
        return False
    return True


def _verify_provider(provider: str, *, skip_test: bool = False) -> tuple[bool, str, Optional[list[str]]]:
    """Hit ``GET /v1/models`` to verify the credentials we just stored.

    Returns ``(ok, status, model_ids)``. ``status`` is one of the ops-library
    failure codes (``invalid_key`` / ``no_credits`` / ``rate_limited`` /
    ``network_error`` / …) and drives the failure submenu's wording.
    """
    from opendde_harness.config.update_providers import test_provider as probe

    # A local deployment has no key to verify -- what is being checked is that
    # the address answers, and saying "API key" there describes a field the user
    # was never asked for.
    if credential_kind(provider) == CRED_LOCAL:
        console.print(_t("  [dim]⏳ Reaching the server…[/dim]", "  [dim]⏳ 正在连接服务…[/dim]"))
    else:
        console.print(_t("  [dim]⏳ Verifying your API key…[/dim]", "  [dim]⏳ 正在验证 API Key…[/dim]"))
    result = probe(provider)
    if result["ok"]:
        models = result.get("models_count")
        suffix = _t(f" ({models} models available)", f"(共 {models} 个可用模型)") if models else ""
        console.print(_t(f"  [green]✓ Connected!{suffix}[/green]", f"  [green]✓ 连接成功!{suffix}[/green]"))
        return True, "valid", result.get("model_ids")

    status = result.get("status", "unknown")
    # Some direct providers (openai / anthropic / deepseek / gemini) ship no
    # base URL and rely on the SDK's built-in endpoint, so there's nothing to
    # hit for a GET /v1/models pre-check. That's NOT a real auth failure: skip
    # the pre-check (the test message sent later exercises real connectivity via
    # litellm) instead of dumping the user into the failure submenu.
    #
    # `no_probe_endpoint` is the probe saying exactly this. It used to say
    # `not_configured` with "api_base" in the text, which is why the old
    # condition read that way -- and a rename this caller does not follow puts
    # every one of those providers into the failure submenu on the first step
    # of onboarding.
    if status == "no_probe_endpoint" or (status == "not_configured" and "api_base" in (result.get("error") or "")):
        if skip_test:
            console.print(
                _t(
                    "  [dim]Skipping the model-list pre-check (this provider has no public /models endpoint); connectivity is not tested (--skip-test).[/dim]",
                    "  [dim]跳过模型列表预检(该服务商无公开 /models 端点);未做连通测试(--skip-test)。[/dim]",
                )
            )
        else:
            console.print(
                _t(
                    "  [dim]Skipping the model-list pre-check (this provider has no public /models endpoint); the test message below will confirm connectivity.[/dim]",
                    "  [dim]跳过模型列表预检(该服务商无公开 /models 端点);稍后的测试消息会验证连通。[/dim]",
                )
            )
        return True, "skipped", None
    hint_map = {
        "invalid_key": _t(
            "Auth failed: the API key is invalid — check for typos / stray spaces.",
            "鉴权失败:API Key 无效 — 检查有无拼写错误或多余空格。",
        ),
        "no_credits": _t(
            "Account out of credits or not provisioned — top up and retry.",
            "账户余额不足或未开通 — 充值后重试。",
        ),
        "rate_limited": _t(
            "Rate limited — wait a bit and retry, or switch provider.",
            "触发限流 — 稍等后重试,或更换服务商。",
        ),
        "network_error": _t(
            "Network error reaching the provider — check network / proxy / VPN.",
            "连接服务商时网络出错 — 检查网络 / 代理 / VPN。",
        ),
        "oauth_token_missing": _t(
            f"Run: ddeharness provider login {provider.replace('_', '-')}",
            f"请运行:ddeharness provider login {provider.replace('_', '-')}",
        ),
    }
    msg = hint_map.get(status, _t(f"Verification failed: {status}", f"验证失败:{status}"))
    console.print(f"  [yellow]✗ {msg}[/yellow]" + (f"  [dim]{result['error']}[/dim]" if result.get("error") else ""))
    return False, status, None


def _load_current_default_model() -> Optional[str]:
    """Read ``agents.defaults.model`` from the on-disk config, if it exists."""
    data = _load_raw_config()
    return (data or {}).get("agents", {}).get("defaults", {}).get("model") or None


def _model_routes_to_provider(model: str, spec: Any) -> bool:
    """True if ``model`` would auto-route to ``spec`` under ``provider='auto'``.

    Defers to the spec so this guard cannot disagree with the routing it guards.
    """
    return bool(model and spec and spec.claims(model))


# How a provider proves who it is. Every decision the wizard makes about a
# provider -- which field to prompt for, what a failure offers to change, what
# "remove" clears, whether a rollback applies -- follows from this one question,
# and it was being answered independently at thirteen sites off two spec flags.
# Each of the last two review rounds found a site that disagreed with the others:
# a rollback that wrote credentials to an OAuth provider and killed the wizard, a
# menu that offered a key prompt to one, a prompt that half-guarded a spec it had
# already dereferenced. Answer it once.


def _format_model_for_provider(provider: str, spec: Any, model_id: str) -> str:
    """Apply the provider's route prefix to a raw ``/v1/models`` id when needed.

    A vendor OpenDDE Harness carries no spec for still needs the prefix, and needs it most:
    the id it returns is bare, and a bare id is routed by keyword and fallback
    rather than to the section the user just configured. Handing one back
    unprefixed sent the request wherever those rules landed -- configuring
    Mistral alongside OpenAI produced "mistral-large-latest", which resolves to
    OpenAI and spends OpenAI's key.

    The rule itself is ``providers.wire.stored_model_id``; deciding it here as
    well is what made the wizard and the TUI write one model two ways.
    """
    return stored_model_id(provider, model_id)


def _pick_model(
    provider: str,
    spec: Any,
    *,
    current_model: Optional[str],
    model_ids: Optional[list[str]],
    probe_status: str,
    user_provided_model: Optional[str],
    non_interactive: bool,
) -> str:
    """Decide the model string to write into ``agents.defaults.model``."""
    # Every exit goes through the formatter, which is idempotent. Applying it
    # only where the candidate list is built covered only the branch that has
    # candidates -- and a vendor OpenDDE Harness carries no spec for reaches the others:
    # the probe cannot pre-check it, so there is no list, so the user types the
    # id. A typed id is bare, and a bare id is routed by keyword and fallback
    # rather than to the provider just configured, which is how
    # "mistral-large-latest" came to be served with OpenAI's key.
    if user_provided_model:
        return _format_model_for_provider(provider, spec, user_provided_model)

    if non_interactive:
        default_model = spec.default_model if spec else ""
        if not default_model:
            raise typer.BadParameter(f"--model is required for provider '{provider}' (no built-in default model).")
        return _format_model_for_provider(provider, spec, default_model)

    if current_model and spec and _model_routes_to_provider(current_model, spec):
        default_value = current_model
    else:
        default_value = (spec.default_model if spec else "") or ""

    # A live fetch is the best answer when there is one; when there is not, the
    # same chain the TUI picker offers beats an empty prompt. Eleven providers
    # carry no curated shortlist, so before this a failed fetch left the user
    # typing a model id from memory.
    if not model_ids:
        from opendde_harness.providers.common_models import common_models_for, litellm_models_for

        known = [*common_models_for(provider), *litellm_models_for(provider)]
        if known:
            # openai / anthropic / deepseek / gemini have no /models endpoint to
            # pre-check, so there is no list on a perfectly healthy run. Saying
            # "couldn't reach the provider" there contradicted the line printed
            # just above it and read as a failure to a user for whom nothing
            # had failed.
            console.print(
                _t(
                    "  [dim]This provider has no model list to fetch - offering the ones we know.[/dim]",
                    "  [dim]该服务商没有可拉取的模型列表,先列出已知的。[/dim]",
                )
                if probe_status == "skipped"
                else _t(
                    "  [dim]Couldn't reach the provider for its model list - offering the ones we know.[/dim]",
                    "  [dim]未能向服务商拉取模型列表,先列出已知的。[/dim]",
                )
            )
            model_ids = known

    if (
        default_value
        and spec
        and default_value == spec.default_model
        and (
            not model_ids
            or any(
                _format_model_for_provider(provider, spec, mid)
                == _format_model_for_provider(provider, spec, default_value)
                for mid in model_ids
            )
        )
    ):
        console.print(
            _t(
                f"  [dim]Default: {default_value} — recommended balance of quality/cost for daily use.[/dim]",
                f"  [dim]默认:{default_value} — 质量/成本均衡,适合日常使用。[/dim]",
            )
        )

    if model_ids:
        choices = [_format_model_for_provider(provider, spec, mid) for mid in model_ids]
        # Dedupe: the chain above already prefixes its ids, and _format_ leaves a
        # correctly-prefixed id alone, so two sources can agree on one model.
        choices = list(dict.fromkeys(choices))
        default_value = _format_model_for_provider(provider, spec, default_value) if default_value else ""
        if default_value not in choices:
            default_value = choices[0]
        prompt_label = _t(
            f"Default model ({len(choices)} available):",
            f"默认模型(共 {len(choices)} 个):",
        )
        # Only the custom provider. Everywhere else the prefix names the vendor
        # the id belongs to, and a list that hides it stops matching what is
        # stored; on a user-described endpoint it is the same word on every row,
        # and that endpoint is also where the fetched list is least likely to
        # carry the model the user means -- so typing one is the first row there.
        own_endpoint = credential_kind(provider) == CRED_ENDPOINT
        chosen = _select_model_id(
            choices,
            default_model=default_value,
            label=prompt_label,
            provider=provider if own_endpoint else None,
            manual_first=own_endpoint,
        )
    else:
        console.print(
            _t(
                "  [dim]Couldn't fetch the model list — select manual entry to continue.[/dim]",
                "  [dim]未能拉取模型列表,请选择手动输入模型 ID。[/dim]",
            )
        )
        chosen = _select_model_id([], default_model=default_value)

    if chosen is None:
        raise typer.Exit(1)  # Ctrl+C
    chosen = chosen.strip()
    if not chosen:
        # Empty submit (e.g. the prefilled default was cleared) falls back to the
        # default rather than tearing down the wizard. The no-default branch
        # validates non-empty, so an empty value only reaches here with a default.
        if default_value:
            # Said out loud: the prompt has already echoed an empty answer, and the
            # next thing on screen is a test message being sent. Without this the
            # model it was sent with appears nowhere.
            console.print(
                _t(
                    f"  [dim]No model entered - using {default_value}.[/dim]",
                    f"  [dim]未输入模型,使用 {default_value}。[/dim]",
                )
            )
            return _format_model_for_provider(provider, spec, default_value)
        raise typer.Exit(1)
    return _format_model_for_provider(provider, spec, chosen)


def _roll_back_provider_fields(provider: str, spec: Any, *, old_key: Optional[str], old_base: Optional[str]) -> None:
    """Undo what this pass wrote, restoring the state read before it started.

    A named function so the behaviour can be driven by a test: the two shapes
    this replaced were both wrong in ways only a test that calls it can hold
    down. Keying off the previous api_key skipped a local deployment entirely,
    leaving a mistyped address where a working one had been; and asking "was it
    configured" cleared both fields for a provider that had held only an
    api_base, erasing an endpoint this pass never touched.

    OAuth providers are skipped: their credentials live in a token file, the ops
    layer refuses to write credential fields for them, and doing it anyway turned
    a failed verification into a dead wizard.
    """
    if credential_kind(provider) == CRED_OAUTH:
        return
    _write_provider_fields(provider, {"api_key": old_key or "", "api_base": old_base})


def _write_provider_fields(provider: str, fields: dict[str, Any]) -> None:
    """Thin wrapper that surfaces ops-library errors with friendly hints."""
    from pydantic import ValidationError

    from opendde_harness.config.update_providers import set_provider_fields

    try:
        set_provider_fields(provider, fields)
    except KeyError as exc:
        console.print(f"  [red]✗[/red] {exc}")
        raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"  [red]✗[/red] {exc}")
        raise typer.Exit(1)
    except ValidationError as exc:
        console.print(_t(f"  [red]✗ Validation failed:[/red]\n{exc}", f"  [red]✗ 校验失败:[/red]\n{exc}"))
        raise typer.Exit(1)


def _persist_default_model(model: Optional[str], provider: str) -> None:
    """Patch ``agents.defaults.model`` and the pin that overrides it.

    Both, always. ``agents.defaults.provider`` wins over whatever a model id
    names, so writing the model alone leaves the wizard's own choice routed to
    whichever provider was pinned before -- with that provider's key. The rule
    for what to pin is ``providers.pin``, the same one the picker and
    ``ddeharness provider use`` ask.
    """
    if not model:
        return
    from opendde_harness.config.loader import load_config
    from opendde_harness.config.update import set_default_model
    from opendde_harness.providers import pin

    try:
        pinned = load_config().agents.defaults.provider or ""
    except Exception:
        pinned = ""
    set_default_model(model, provider=pin.resolve(model, provider=provider, pinned=pinned))


# ---------------------------------------------------------------------------
# Step 1 — connectivity-failure submenu + test probe
# ---------------------------------------------------------------------------


def _failure_choice(options: list[tuple[str, str]], *, non_interactive: bool) -> str:
    """Render a numbered failure submenu, return the chosen value.

    ``options`` is a list of ``(label, value)``. In non-interactive mode the
    last option (always "continue anyway") is auto-chosen so headless runs
    never block.
    """
    if non_interactive:
        return options[-1][1]
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    chosen = questionary.select(
        _t("What would you like to do?", "想做什么?"),
        choices=[questionary.Choice(label, value=value) for label, value in options],
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
    ).ask()
    if chosen is None:
        raise typer.Exit(1)
    return chosen


def _probe_failure(exc: Exception) -> str:
    """Why the test message failed, in words.

    A timeout raised by ``asyncio.wait_for`` carries no message at all, so the
    line read "Test failed:" and stopped, leaving the one setup that fails for
    a benign reason -- a slow model -- with nothing to act on.
    """
    if isinstance(exc, TimeoutError):
        return _t(
            "no reply within 60s. A slow or heavily loaded model can need longer; the credentials are fine if the model list loaded.",
            "60 秒内没有回复。模型较慢或负载较高时会这样；能拉到模型列表说明凭据没问题。",
        )
    return str(exc) or exc.__class__.__name__


def _run_test_probe(
    provider: str,
    *,
    non_interactive: bool,
    warnings: list[str],
    allow_repick: bool = True,
    is_oauth: bool = False,
) -> str:
    """Send a one-shot test message; on failure offer recovery options.

    Returns one of ``"ok"`` / ``"continue"`` / ``"repick"`` / ``"rekey"`` /
    ``"switch"``. A test-message failure can be a wrong model, a bad key, or an
    account/balance issue, so the menu offers all the matching exits (aligning
    with the connectivity-failure menu in ``_resolve_model_with_test``);
    ``allow_repick=False`` drops the model option when selection is unavailable.
    """
    console.print(
        _t(
            f'  [dim]Sending test message: "{DEFAULT_PROBE_MESSAGE}"[/dim]',
            f'  [dim]正在发送测试消息:"{DEFAULT_PROBE_MESSAGE}"[/dim]',
        )
    )
    try:
        text, tokens, elapsed = send_probe()
    except Exception as exc:
        console.print(
            _t(f"  [red]✗ Test failed:[/red] {_probe_failure(exc)}", f"  [red]✗ 测试失败:[/red] {_probe_failure(exc)}")
        )
        console.print(
            _t(
                "  [dim]Run 'ddeharness provider test' to re-check, or confirm the model is served by this provider.[/dim]",
                "  [dim]可运行 'ddeharness provider test' 复查,或确认该模型确由此服务商提供。[/dim]",
            )
        )
        print_probe_troubleshooting(provider)
        options = [(_t("Retry", "重试"), "retry")]
        if allow_repick:
            options.append((_t("Re-pick model", "重新选模型"), "repick"))
        options.append(
            (_t("Sign in again", "重新登录"), "reauth") if is_oauth else (_t("Re-enter key", "重新填 Key"), "rekey")
        )
        options += [
            (_t("Switch provider", "更换服务商"), "switch"),
            (_t("Continue anyway", "仍然继续"), "continue"),
        ]
        choice = _failure_choice(options, non_interactive=non_interactive)
        if choice == "retry":
            return _run_test_probe(
                provider,
                non_interactive=non_interactive,
                warnings=warnings,
                allow_repick=allow_repick,
                is_oauth=is_oauth,
            )
        if choice in ("repick", "rekey", "reauth", "switch"):
            return choice
        warnings.append("provider test message")
        return "continue"

    console.print(f"  [bold]▶ Agent:[/bold] {text}")
    extras: list[str] = []
    if tokens:
        extras.append(f"{tokens} tokens")
    extras.append(f"{elapsed:.1f}s")
    console.print(f"  [green]✓ {', '.join(extras)}[/green]")
    return "ok"


# ---------------------------------------------------------------------------
# Step 1 — add one provider (used by both first-run and the "add" entry)
# ---------------------------------------------------------------------------


def _configure_one_provider(
    *,
    provider: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    skip_test: bool = False,
) -> Optional[dict[str, Any]]:
    """Drive one provider through pick → credentials → verify → model → test.

    Returns ``{"provider", "model"}`` on success, or ``None`` if the user
    chose to go back from the interactive provider picker.
    """
    from opendde_harness.providers.registry import find_by_name

    # Loop so "Switch provider" on a connectivity failure rewinds to the
    # picker instead of tearing the whole wizard down (keeps steps 2/3/4).
    # A provider passed by flag is used once; switching then requires the
    # interactive picker (or, in non-interactive mode, is impossible).
    flag_provider = provider

    def _rewind() -> None:
        """Discard the flag values before the next pass through the picker.

        All of them, not just the provider: they were typed for the provider
        that just failed. A stale --api-key was written to the newly picked
        provider without a prompt, a stale --base-url pointed it at the previous
        provider's machine, and picking a local deployment -- which rejects
        --api-key by design -- ended the whole wizard on a usage error, losing
        the later steps this loop exists to keep.
        """
        nonlocal flag_provider, api_key, base_url, model
        flag_provider = api_key = base_url = model = None

    while True:
        if flag_provider:
            provider = _validate_provider_name(flag_provider)
        else:
            if non_interactive:
                raise typer.BadParameter("--provider is required in non-interactive mode")
            picked = _select_provider()
            if picked is None:
                raise typer.Exit(1)
            if picked is _BACK:
                return None
            # Same gate as the flag path: the vendor step lets the user type a
            # name, and a typo there used to reach the config layer as an
            # uncaught KeyError that tore down the wizard mid-setup.
            try:
                provider = _validate_provider_name(picked)
            except typer.BadParameter as exc:
                console.print(f"  [red]x[/red] {exc}")
                _rewind()
                continue

        spec = find_by_name(provider)
        kind = credential_kind(provider)
        is_oauth = kind == CRED_OAUTH
        is_custom = kind == CRED_ENDPOINT
        # The interactive picker already echoes the chosen provider; only print
        # an explicit confirmation when it came from --provider (no echo then).
        if flag_provider:
            console.print(
                _t(
                    f"  [dim]Provider:[/dim] [accent]{_provider_label(provider)}[/accent]",
                    f"  [dim]服务商:[/dim] [accent]{_provider_label(provider)}[/accent]",
                )
            )

        # Snapshot the stored key before _collect_credentials overwrites it, so a
        # failed re-configuration of an existing provider can be rolled back to
        # its prior working key (rather than left holding the just-typed bad one).
        # Read through the ops library: it folds in a section still stored under
        # the provider's pre-rename name, which a raw lookup by the typed name
        # misses -- and the write below consolidates onto the current name, so a
        # rollback would otherwise restore nothing over a real key.
        from opendde_harness.config.update_providers import get_provider_config

        _prev = get_provider_config(provider, redact_secrets=False)
        old_key = _prev.get("api_key")
        old_base = _prev.get("api_base")

        custom_model = _collect_credentials(
            provider,
            is_oauth=is_oauth,
            is_custom=is_custom,
            is_local=kind == CRED_LOCAL,
            api_key=api_key,
            base_url=base_url,
            model=model,
            non_interactive=non_interactive,
        )
        if custom_model is _BACK:
            # User backed out of the first credential field — rewind to the
            # provider picker (drop the flags so the picker actually shows).
            _rewind()
            continue

        chosen_model = _resolve_model_with_test(
            provider,
            spec,
            is_custom=is_custom,
            custom_model=custom_model,
            user_model_flag=model,
            non_interactive=non_interactive,
            warnings=warnings,
            skip_test=skip_test,
        )
        if chosen_model is None:
            # "Switch provider" — re-run the picker (drop the flags so the second
            # pass prompts rather than reusing the failed values), undoing what
            # this pass wrote.
            #
            # Put back exactly what was there, read before this pass wrote
            # anything. One branch, because "was it configured" is the wrong
            # question twice over: a local deployment is configured by address
            # and has no key, so a rollback keyed off the old key skipped it and
            # a mistyped address replaced a working one for good; and a provider
            # that held only an api_base counts as unconfigured, so clearing
            # both fields for a "new" provider erased an endpoint this pass had
            # never touched.
            #
            # OAuth providers are left alone: their credentials live in a token
            # file, `set_provider_fields` refuses to write credential fields for
            # them at all, and doing so turned a failed verification into a
            # RuntimeError that took the whole wizard down.
            _roll_back_provider_fields(provider, spec, old_key=old_key, old_base=old_base)
            _rewind()
            continue
        _persist_default_model(chosen_model, provider)
        return {"provider": provider, "model": chosen_model}


def _collect_credentials(
    provider: str,
    *,
    is_oauth: bool,
    is_custom: bool,
    is_local: bool = False,
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    non_interactive: bool,
) -> Any:
    """Auth setup: OAuth browser flow or api_key write. Returns the custom
    model id when the provider is ``custom`` (locked in here), ``None`` for a
    non-custom provider, or ``_BACK`` if the user backed out of the first
    interactive credential field, or if the vendor cannot be configured by a
    bare key at all (caller should rewind to the picker either way)."""
    from opendde_harness.providers.auth import key_refusal

    refusal = key_refusal(provider)
    if refusal is not None:
        console.print(f"  [red]x[/red] {refusal}")
        if non_interactive:
            raise typer.Exit(2)
        return _BACK

    if is_oauth:
        if non_interactive:
            console.print(
                "[red]OAuth providers require an interactive browser flow.[/red]\n"
                "Run [accent]ddeharness provider login "
                f"{provider.replace('_', '-')}[/accent] separately, then re-run "
                "onboard."
            )
            raise typer.Exit(2)
        # Loop so a failed login offers retry / back instead of crashing out.
        while True:
            if _run_oauth_login(provider):
                return None
            choice = _failure_choice(
                [
                    (_t("Retry", "重试"), "retry"),
                    (_t("Back (pick another provider)", "返回(改选服务商)"), "back"),
                ],
                non_interactive=non_interactive,
            )
            if choice == "retry":
                continue
            return _BACK

    if is_local:
        # A local deployment authenticates on nothing: it is reached by address.
        # Routing it through the api_key prompt would stop the user at a
        # minimum-length check for a credential that does not exist.
        from opendde_harness.providers.registry import find_by_name

        spec = find_by_name(provider)
        if api_key:
            # Said out loud rather than dropped: a local deployment writes no
            # api_key, so silently ignoring the flag looks like it was accepted.
            raise typer.BadParameter(
                f"{provider} is a local deployment and takes no --api-key; pass --base-url instead"
            )
        if base_url and not base_url.strip().startswith(("http://", "https://")):
            # The interactive prompt validates this; the flag path did not, so a
            # scheme-less address went into the config and failed at first use.
            raise typer.BadParameter(f"--base-url must start with http:// or https:// (got {base_url!r})")
        if not base_url:
            if non_interactive:
                raise typer.BadParameter(f"--base-url is required for {provider} in non-interactive mode")
            from opendde_harness.config.update_providers import get_provider_config

            try:
                stored = get_provider_config(provider, redact_secrets=False).get("api_base") or ""
            except KeyError:
                stored = ""
            base_url = _prompt_local_api_base(spec, current=stored, allow_back=True)
            if base_url is _BACK:
                return _BACK
        _write_provider_fields(provider, {"api_base": base_url})
        return None

    if not api_key:
        from opendde_harness.providers.registry import normalize_provider_name

        # GigaChat's key is not a typical API key -- it is base64(client_id:
        # client_secret) -- and the generic prompt below gives no room to say
        # so, so the wizard would otherwise send someone looking for a plain
        # key straight into a 401.
        if normalize_provider_name(provider) == "gigachat":
            console.print(
                "  [dim]GigaChat's key is base64(client_id:client_secret) from the "
                "GigaChat API console, not a typical API key.[/dim]"
            )

    # Pure interactive path (no creds came from flags): prompt field-by-field
    # with empty-submit = back; backing out of the first field rewinds to the
    # provider picker.
    pure_interactive = not non_interactive and not api_key and (not is_custom or (not base_url and not model))
    if pure_interactive:
        prompts: list[Callable[[], Any]] = [lambda: _prompt_api_key(provider, allow_back=True)]
        if is_custom:
            prompts.append(lambda: _prompt_base_url(allow_back=True))
        collected = _collect_fields(prompts)
        if collected is None:
            return _BACK
        api_key = collected[0]
        if is_custom:
            base_url = collected[1]
    else:
        if not api_key:
            if non_interactive:
                raise typer.BadParameter("--api-key is required in non-interactive mode")
            api_key = _prompt_api_key(provider)
        if is_custom:
            if not base_url:
                if non_interactive:
                    raise typer.BadParameter("--base-url is required when --provider=custom in non-interactive mode")
                base_url = _prompt_base_url()
            if not model and non_interactive:
                raise typer.BadParameter("--model is required when --provider=custom in non-interactive mode")

    fields: dict[str, Any] = {"api_key": api_key}
    custom_model: Optional[str] = None
    if is_custom:
        fields["api_base"] = base_url
        custom_model = model
    elif base_url:
        fields["api_base"] = base_url

    _write_provider_fields(provider, fields)
    return custom_model


def _resolve_model_with_test(
    provider: str,
    spec: Any,
    *,
    is_custom: bool,
    custom_model: Optional[str],
    user_model_flag: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    skip_test: bool = False,
) -> Optional[str]:
    """Verify connectivity → pick the default model → send a test probe.

    On a verify or test-message failure, offers a recovery submenu (retry /
    re-pick model / re-enter key / switch / continue). Custom providers use
    the same model picker after connectivity verification. Only failures stop; success
    auto-advances. Returns the chosen model, or ``None`` to signal "switch
    provider" (the caller rewinds to the picker).
    """
    while True:
        ok, status, model_ids = _verify_provider(provider, skip_test=skip_test)
        if not ok:
            options = (
                [
                    (_t("Retry", "重试"), "retry"),
                    # A local deployment that cannot be reached is usually a
                    # wrong address, and this is the branch it lands in -- so
                    # retry alone left the one thing worth changing unreachable.
                    *(
                        [(_t("Re-enter server URL", "重新填服务地址"), "rebase")]
                        if credential_kind(provider) == CRED_LOCAL
                        else []
                    ),
                    (_t("Continue anyway", "仍然继续"), "continue"),
                ]
                if status == "network_error"
                else [
                    # What to offer depends on what the provider is reached by.
                    # A local deployment has no key to re-enter, so offering that
                    # left a mistyped address with no way back to the field.
                    (
                        (_t("Sign in again", "重新登录"), "reauth")
                        if credential_kind(provider) == CRED_OAUTH
                        else (_t("Re-enter server URL", "重新填服务地址"), "rebase")
                        if credential_kind(provider) == CRED_LOCAL
                        else (_t("Re-enter key", "重新填 Key"), "rekey")
                    ),
                    # Also retry, because this branch takes the failures that
                    # cannot be sorted: a credential the account refused and a
                    # refresh that could not reach the network arrive as the same
                    # thing, and only one of them is fixed by signing in again.
                    (_t("Retry", "重试"), "retry"),
                    (_t("Switch provider", "更换服务商"), "switch"),
                    (_t("Continue anyway", "仍然继续"), "continue"),
                ]
            )
            choice = _failure_choice(options, non_interactive=non_interactive)
            if choice == "retry":
                continue
            if choice == "rekey" and not non_interactive:
                _write_provider_fields(provider, {"api_key": _prompt_api_key(provider)})
                continue
            if choice == "rebase" and not non_interactive:
                from opendde_harness.config.update_providers import get_provider_config

                try:
                    stored = get_provider_config(provider, redact_secrets=False).get("api_base") or ""
                except KeyError:
                    stored = ""
                retyped = _prompt_local_api_base(spec, current=stored)
                _write_provider_fields(provider, {"api_base": retyped})
                continue
            if choice == "reauth" and not non_interactive:
                if _run_oauth_login(provider):
                    continue
                return None
            if choice == "switch":
                return None
            warnings.append("provider connectivity")
            model_ids = None
        break

    if is_custom:
        user_model_flag = custom_model or user_model_flag

    current = _load_current_default_model()
    while True:
        chosen = _pick_model(
            provider,
            spec,
            current_model=current,
            model_ids=model_ids,
            probe_status=status,
            user_provided_model=user_model_flag,
            non_interactive=non_interactive,
        )
        _persist_default_model(chosen, provider)
        if skip_test:
            return chosen
        result = _run_test_probe(
            provider,
            non_interactive=non_interactive,
            warnings=warnings,
            is_oauth=credential_kind(provider) == CRED_OAUTH,
        )
        if result == "switch":
            return None
        if result == "rekey":
            _write_provider_fields(provider, {"api_key": _prompt_api_key(provider)})
            # Re-test the same model with the new key (picker defaults to it).
            current = chosen
            user_model_flag = None
            continue
        if result == "reauth":
            if not _run_oauth_login(provider):
                return None
            current = chosen
            user_model_flag = None
            continue
        if result == "repick":
            current = chosen
            user_model_flag = None
            continue
        return chosen  # ok / continue


def _configure_existing_provider_model(*, non_interactive: bool) -> bool:
    """Choose a model for an already-authenticated provider without re-login."""
    if non_interactive:
        return False
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE
    from opendde_harness.providers.registry import find_by_name

    choices = [questionary.Choice(_provider_label(name), value=name) for name in _configured_providers()]
    if not choices:
        return False
    provider = questionary.select(
        _t("Choose the provider for the default model:", "选择默认模型对应的服务商:"),
        choices=choices,
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
    ).ask()
    if not provider:
        raise typer.Exit(1)
    spec = find_by_name(provider)
    ok, status, model_ids = _verify_provider(provider)
    if not ok:
        return False
    chosen = _pick_model(
        provider,
        spec,
        current_model=None,
        model_ids=model_ids,
        probe_status=status,
        user_provided_model=None,
        non_interactive=False,
    )
    _persist_default_model(chosen, provider)
    result = _run_test_probe(
        provider,
        non_interactive=False,
        warnings=[],
        is_oauth=credential_kind(provider) == CRED_OAUTH,
    )
    if result == "reauth":
        return _run_oauth_login(provider)
    return result in {"ok", "continue"}


# ---------------------------------------------------------------------------
# Step 1 — multi-provider entry (existing-config branch: done / add / edit)
# ---------------------------------------------------------------------------


def _manage_existing_providers(*, non_interactive: bool) -> None:
    """Edit/remove submenu for already-configured providers (interactive only)."""
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE
    from opendde_harness.providers.registry import find_by_name

    while True:
        configured = _configured_providers()
        if not configured:
            return
        choices = [questionary.Choice(_provider_label(n), value=n) for n in configured]
        choices.append(questionary.Choice(_t("Back", "返回"), value=_BACK))
        target = questionary.select(
            _t("Pick a provider to manage:", "选择要管理的服务商:"),
            choices=choices,
            style=OPENDDE_HARNESS_STYLE,
            qmark=_QMARK,
        ).ask()
        if target is None or target is _BACK:
            return

        action = questionary.select(
            _t(
                f"What would you like to do with {_provider_label(target)}?",
                f"对 {_provider_label(target)} 想做什么?",
            ),
            choices=[
                questionary.Choice(_t("Update API key", "更新 API Key"), value="update"),
                questionary.Choice(
                    _t("Remove (clear this provider's key)", "移除(清除该服务商的 Key)"),
                    value="remove",
                ),
                questionary.Choice(_t("Back", "返回"), value=_BACK),
            ],
            style=OPENDDE_HARNESS_STYLE,
            qmark=_QMARK,
        ).ask()
        if action is None or action is _BACK:
            continue
        if action == "update":
            target_spec = find_by_name(target)
            if credential_kind(target) == CRED_OAUTH:
                # Nothing here to update: the credential is a token file, and the
                # ops layer refuses credential writes for these -- so offering the
                # key prompt ended the wizard instead of editing anything.
                console.print(
                    _t(
                        f"  [dim]{_provider_label(target)} signs in through OAuth. "
                        f"Run: ddeharness provider login {target.replace('_', '-')}[/dim]",
                        f"  [dim]{_provider_label(target)} 通过 OAuth 登录。"
                        f"请运行: ddeharness provider login {target.replace('_', '-')}[/dim]",
                    )
                )
                continue
            if credential_kind(target) == CRED_LOCAL:
                # A local deployment holds no key; what there is to update is
                # where it lives. Offering the key prompt wrote a credential into
                # a provider that never reads one, and left the address alone.
                from opendde_harness.config.update_providers import get_provider_config

                try:
                    stored = get_provider_config(target, redact_secrets=False).get("api_base") or ""
                except KeyError:
                    stored = ""
                retyped = _prompt_local_api_base(target_spec, current=stored)
                _write_provider_fields(target, {"api_base": retyped})
            elif credential_kind(target) == CRED_ENDPOINT:
                # A self-hosted endpoint is a key *and* the address it is sent
                # to. Updating only the key left the one field that moves when
                # the user redeploys -- the URL -- unreachable from this menu.
                from opendde_harness.config.update_providers import get_provider_config

                try:
                    stored = get_provider_config(target, redact_secrets=False).get("api_base") or ""
                except KeyError:
                    stored = ""
                retyped_key = _prompt_api_key(target)
                retyped_url = _prompt_base_url(stored or "https://")
                _write_provider_fields(target, {"api_key": retyped_key, "api_base": retyped_url})
            else:
                _write_provider_fields(target, {"api_key": _prompt_api_key(target)})
            console.print(
                _t(
                    f"  [green]✓ Updated {_provider_label(target)}.[/green]",
                    f"  [green]✓ 已更新 {_provider_label(target)}。[/green]",
                )
            )
        elif action == "remove":
            current = _load_current_default_model()
            from opendde_harness.providers.registry import find_by_name, normalize_provider_name, split_model_id

            spec = find_by_name(target)
            if spec is not None:
                was_default_source = bool(current and _model_routes_to_provider(current, spec))
            else:
                # A vendor with no spec of ours is reached by its prefix alone, so
                # that is the whole test. Treating "no spec" as "not the source"
                # skipped the guard and left a default model pointing at a
                # provider whose key had just been removed.
                prefix, _ = split_model_id(current or "")
                was_default_source = bool(current and prefix == normalize_provider_name(target))
            if was_default_source:
                confirm = questionary.confirm(
                    _t(
                        f"The current default model comes from {_provider_label(target)}; "
                        "removing it means you'll need to pick a new default. Remove anyway?",
                        f"当前默认模型来自 {_provider_label(target)};移除后需要重新选择默认模型。仍要移除吗?",
                    ),
                    default=False,
                    style=OPENDDE_HARNESS_STYLE,
                    qmark=_QMARK,
                ).ask()
                if not confirm:
                    continue
            # Clear both: a local deployment counts as configured by its
            # api_base, so clearing only the key reported it removed and left it
            # in the list, still reachable. An OAuth provider has neither field
            # to clear and refuses the write, so it is told where its credential
            # actually lives instead of ending the run.
            target_spec = find_by_name(target)
            if credential_kind(target) == CRED_OAUTH:
                console.print(
                    _t(
                        f"  [dim]{_provider_label(target)}'s credential is an OAuth token, not a config field, "
                        "so there is nothing here to remove.[/dim]",
                        f"  [dim]{_provider_label(target)} 的凭据是 OAuth token,不在配置字段里,"
                        "这里没有可移除的内容。[/dim]",
                    )
                )
                continue
            _write_provider_fields(target, {"api_key": "", "api_base": None})
            if was_default_source:
                # Clear the now-dangling default so step 1's guard forces a
                # re-pick instead of leaving a model whose provider has no key.
                from opendde_harness.config.update import set_default_model

                # The pin goes with it: left behind it would route the next model
                # the user picks to the provider whose key was just removed.
                set_default_model("", provider="auto")
            console.print(
                _t(
                    f"  [green]✓ Removed {_provider_label(target)}'s configuration.[/green]",
                    f"  [green]✓ 已移除 {_provider_label(target)} 的配置。[/green]",
                )
            )


def _step1_provider(
    *,
    provider: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    skip_test: bool = False,
) -> object:
    """Step 1 screen. Returns ``_BACK`` only when the user backs out of the
    first-run picker on the welcome screen (handled by the runner)."""
    _step_header(1, _t("Choose your LLM provider", "选择 LLM 服务商"))
    console.print(
        _t(
            "  [dim]OpenDDE Harness's chat and reasoning are all driven by it.[/dim]",
            "  [dim]OpenDDE Harness 的对话与思考都由它驱动。[/dim]",
        )
    )

    configured = _configured_providers()
    if non_interactive or not configured:
        result = _configure_one_provider(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            non_interactive=non_interactive,
            warnings=warnings,
            skip_test=skip_test,
        )
        if result is None:
            return _BACK
        return None

    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    while True:
        names = ", ".join(_provider_label(n).split(" (")[0] for n in _configured_providers())
        action = questionary.select(
            _t(
                f"LLM provider already configured: {names}. What would you like to do?",
                f"LLM 服务商已配置:{names}。想做什么?",
            ),
            choices=[
                questionary.Choice(_t("Done, continue", "完成,继续"), value="done"),
                questionary.Choice(_t("Choose default model", "选择默认模型"), value="model"),
                questionary.Choice(_t("Add another provider", "新增一个服务商"), value="add"),
                questionary.Choice(_t("Edit / remove a provider", "编辑 / 移除服务商"), value="edit"),
            ],
            style=OPENDDE_HARNESS_STYLE,
            qmark=_QMARK,
        ).ask()
        if action is None:
            raise typer.Exit(1)  # Ctrl+C exits; never treat it as "done"
        if action == "done":
            # Step 1 is required: never advance without at least one provider AND
            # a default model, so deleting every provider can't slip through.
            if not (_configured_providers() and _load_current_default_model()):
                console.print(
                    _t(
                        "  [yellow]At least one provider with a default model is required — add or re-pick one.[/yellow]",
                        "  [yellow]至少需要一个带默认模型的服务商 — 请新增或重新选择一个。[/yellow]",
                    )
                )
                continue
            return None
        if action == "model":
            if _configure_existing_provider_model(non_interactive=False):
                continue
            console.print(
                _t(
                    "  [yellow]Could not configure a default model. Choose a provider and try again.[/yellow]",
                    "  [yellow]无法配置默认模型,请重新选择服务商。[/yellow]",
                )
            )
        if action == "add":
            _configure_one_provider(
                provider=None,
                api_key=None,
                base_url=None,
                model=None,
                non_interactive=False,
                warnings=warnings,
                skip_test=skip_test,
            )
        elif action == "edit":
            _manage_existing_providers(non_interactive=non_interactive)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------


def _protein_design_recap() -> str:
    """Where Protein Design compute runs and whether it answers, for the recap.

    Read from disk and probed the same way ``ddeharness doctor`` does, so the
    recap reports the state the user will find rather than what the step
    intended.
    """
    from opendde_harness.cli.onboard_compute import inspect_compute, load_protein_design_config

    config = load_protein_design_config()
    if not config:
        return _t("[yellow]not configured[/yellow]", "[yellow]未配置[/yellow]")
    report = inspect_compute(config)
    placement = {
        "local_docker": _t("local Docker", "本地 Docker"),
        "remote_service": _t("remote service", "远程服务"),
        "worker_pool": _t("worker pool", "计算节点池"),
    }.get(str(report.get("placement")), str(report.get("placement") or "—"))
    readiness = (
        _t("[green]ready[/green]", "[green]就绪[/green]")
        if report.get("ready")
        else _t("[yellow]unreachable[/yellow]", "[yellow]未就绪[/yellow]")
    )
    return f"{placement}, {readiness}"


def _print_next_steps(*, warnings: list[str], show_next_steps: bool = True) -> None:
    from rich.table import Table

    console.print()
    if warnings:
        console.print(
            Panel(
                _t(
                    "[bold yellow]⚠ Setup finished with warnings[/bold yellow]",
                    "[bold yellow]⚠ 配置完成,但有警告[/bold yellow]",
                )
                + "\n\n"
                + _t(
                    "[dim]These items didn't pass a connectivity test:[/dim] ",
                    "[dim]以下项目未通过连通测试:[/dim] ",
                )
                + f"{', '.join(warnings)}\n"
                + _t(
                    "[dim]Fix them before relying on the related features "
                    "(re-run [/dim][accent]ddeharness onboard[/accent][dim] to reconfigure).[/dim]",
                    "[dim]在依赖相关功能前请先修复(重新运行 [/dim][accent]ddeharness onboard[/accent][dim] 重新配置)。[/dim]",
                ),
                border_style="yellow",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            Panel(
                _t(
                    "[bold green]🎉 Setup complete![/bold green]",
                    "[bold green]🎉 配置完成![/bold green]",
                ),
                border_style="green",
                padding=(0, 2),
            )
        )

    # Recap what was configured (read from disk) so the user has closure.
    provs = ", ".join(_provider_label(n).split(" (")[0] for n in _configured_providers()) or "—"
    mem = (
        _t("Persistent Memory", "长期记忆已启用")
        if onboard_memory._memory_enabled()
        else _t("[yellow]off[/yellow]", "[yellow]未启用[/yellow]")
    )
    recap = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    recap.add_column(style="dim", no_wrap=True)
    recap.add_column()
    recap.add_row(_t("Provider", "服务商"), provs)
    recap.add_row(_t("Default model", "默认模型"), _load_current_default_model() or "—")
    recap.add_row(_t("Memory", "长期记忆"), mem)
    recap.add_row(_t("Protein Design", "蛋白设计"), _protein_design_recap())
    console.print(
        Panel(
            recap,
            title=f"[bold]{_t('Your setup', '你的配置')}[/bold]",
            title_align="left",
            border_style="#8a6d00",
            padding=(1, 2),
        )
    )

    if not show_next_steps:
        # The startup gate runs the wizard with the TUI already on its way in,
        # so a list of commands to try next is answered before it is read.
        console.print(
            _t(
                "  [dim]Setup complete - starting the TUI...[/dim]",
                "  [dim]配置完成,正在进入 TUI...[/dim]",
            )
        )
        return

    table = Table(show_header=False, box=None, padding=(0, 3, 0, 0))
    table.add_column(style="accent", no_wrap=True)
    table.add_column(style="dim")
    table.add_row("ddeharness", _t("start the antibody design TUI", "启动抗体设计 TUI"))
    table.add_row("ddeharness doctor", _t("check configuration and compute readiness", "检查配置与计算就绪状态"))
    table.add_row("ddeharness tracing", _t("open the results dashboard", "打开结果面板"))
    console.print(
        Panel(
            table,
            title=f"[bold]{_t('Get started', '开始使用')}[/bold]",
            title_align="left",
            border_style="border",
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# Wizard runner (screen state machine) + reusable entry point
# ---------------------------------------------------------------------------


def run_wizard(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    skip_memory: bool = False,
    skip_protein_design: bool = False,
    non_interactive: bool = False,
    yes: bool = False,
    reset: bool = False,
    skip_test: bool = False,
    show_next_steps: bool = True,
) -> None:
    """Run the three-step onboarding wizard end-to-end.

    The reusable entry point: the ``onboard`` CLI command and the startup gate
    both call this. Screens form a state machine so a ``0) Back`` choice can
    rewind one step; Ctrl+C exits keeping whatever was already written.

    Internal INFO logs (config writes, etc.) are hushed for the wizard's
    duration so they don't clutter the UI, then restored in ``finally`` —
    display-only; logging elsewhere is unaffected.
    """
    from loguru import logger as _logger

    _logger.disable("opendde_harness")
    try:
        _run_wizard_body(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            skip_memory=skip_memory,
            skip_protein_design=skip_protein_design,
            non_interactive=non_interactive,
            yes=yes,
            reset=reset,
            skip_test=skip_test,
            show_next_steps=show_next_steps,
        )
    finally:
        _logger.enable("opendde_harness")


def _step3_protein_design(*, skip: bool, non_interactive: bool, warnings: list[str]) -> object:
    """Step 3 -- Protein Design compute settings."""
    _step_header(3, _t("Protein Design", "蛋白设计配置"))
    if skip or non_interactive:
        return None
    from opendde_harness.cli.onboard_protein_design import configure_protein_design

    configure_protein_design()
    return None


def _run_wizard_body(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    skip_memory: bool = False,
    skip_protein_design: bool = False,
    non_interactive: bool = False,
    yes: bool = False,
    reset: bool = False,
    skip_test: bool = False,
    show_next_steps: bool = True,
) -> None:
    global _LANG
    _check_tty_or_die(non_interactive)
    _LANG = _config_language()  # start from the saved language (default "en")
    if not non_interactive:
        _pick_language()  # may change _LANG (persisted after bootstrap below)
    _handle_existing_config(reset=reset, yes=yes, non_interactive=non_interactive)
    _bootstrap_empty_config()
    if not non_interactive:
        from opendde_harness.config.update import set_language

        set_language(_LANG)  # persist now that config.json exists

    console.print()
    console.print(
        Panel(
            _t(
                "[bold][accent]✨ Welcome to the OpenDDE Harness setup wizard[/accent][/bold]\n\n"
                "[dim]We'll configure, in order:[/dim]\n"
                "  [accent]①[/accent] LLM      [accent]②[/accent] Persistent Memory      "
                "[accent]③[/accent] Protein Design\n\n"
                "[dim]↑↓ select · Enter confirm · Ctrl+C quit anytime — anything already written is kept.[/dim]",
                "[bold][accent]✨ 欢迎使用 OpenDDE Harness 配置向导[/accent][/bold]\n\n"
                "[dim]我们将依次配置:[/dim]\n"
                "  [accent]①[/accent] LLM      [accent]②[/accent] Persistent Memory      "
                "[accent]③[/accent] 蛋白设计\n\n"
                "[dim]↑↓ 选择 · Enter 确认 · 随时 Ctrl+C 退出 — 已写入的配置会保留。[/dim]",
            ),
            border_style="border",
            padding=(1, 2),
        )
    )

    warnings: list[str] = []

    # Screen state machine. Each screen returns ``_BACK`` to rewind or anything
    # else to advance. Step 1 is required; backing out of it from the first
    # screen is a no-op (there's no earlier screen).
    screens: list[Callable[[], object]] = [
        lambda: _step1_provider(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            non_interactive=non_interactive,
            warnings=warnings,
            skip_test=skip_test,
        ),
        lambda: onboard_memory._step4_memory(
            skip=skip_memory,
            non_interactive=non_interactive,
            main_model=_load_current_default_model(),
            warnings=warnings,
            skip_test=skip_test,
        ),
        lambda: _step3_protein_design(
            skip=skip_protein_design,
            non_interactive=non_interactive,
            warnings=warnings,
        ),
    ]

    index = 0
    while index < len(screens):
        result = screens[index]()
        if result is _BACK:
            if index == 0:
                # The language picker ran before the state machine, so Step 1
                # is the first *numbered* screen but not the first screen the
                # user saw. Backing out of it returns to the language picker:
                # re-pick (persisting the choice) and then re-display Step 1 in
                # the chosen language. Step 1 stays required -- we never skip
                # past it, which would leave provider/model unwritten and
                # re-trip the startup gate into an infinite loop.
                _pick_language()
                from opendde_harness.config.update import set_language

                set_language(_LANG)
            else:
                index -= 1
        else:
            index += 1

    _print_next_steps(warnings=warnings, show_next_steps=show_next_steps)


def ensure_ready_to_start(*, interactive: bool) -> bool:
    """Whether the TUI can start; on an interactive first run, the wizard runs first.

    Two different things fail the startup check. A config with no usable provider
    at all is a first run, and the wizard is the answer -- it configures more
    subsystems besides this one. A config whose default model happens to name a
    provider that has gone unusable is not: the wizard restarts at the language
    screen to fix one line, over a session that has other providers ready. Say
    which line, and let the user fix it where models are chosen.

    Without a terminal the wizard cannot run, so the caller is told to run
    ``ddeharness onboard`` instead. The wizard writes as it goes, so a cancelled
    run is re-checked rather than assumed to have failed.
    """
    if _is_config_populated():
        return True

    if _configured_providers():
        model = (_load_raw_config().get("agents", {}) or {}).get("defaults", {}).get("model")
        # Says what was found, not why: the provider it resolves to may have no
        # credentials, or the id may resolve to a provider that never served it (a
        # deployment name carrying another vendor's keyword does that). Naming a cause
        # we have not established sends the user to fix the wrong thing.
        console.print(
            _t(
                f"  [yellow]No usable provider resolves the default model ({model}).[/yellow]",
                f"  [yellow]默认模型({model})解析不到可用的服务商。[/yellow]",
            )
        )
        console.print(
            _t(
                "  [dim]Choose one that works: `ddeharness tui` then /model. Or `ddeharness onboard` to set this up again.[/dim]",
                "  [dim]换一个能用的:`ddeharness tui` 后按 /model。或用 `ddeharness onboard` 重新配置。[/dim]",
            )
        )
        return True

    if interactive:
        try:
            run_wizard(show_next_steps=False)
        except (typer.Exit, KeyboardInterrupt):
            pass
        return ensure_ready_to_start(interactive=False)

    console.print("OpenDDE Harness is not configured. Run `ddeharness onboard`.")
    return False


# ---------------------------------------------------------------------------
# Typer entry point
# ---------------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    """Attach the ``onboard`` command to ``app``."""

    @app.command()
    def onboard(
        provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider name (skips Step 1's prompt)"),
        api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the chosen provider"),
        base_url: Optional[str] = typer.Option(
            None,
            "--base-url",
            help="Server URL: required for a local deployment (ollama_chat / hosted_vllm), or a custom OpenAI-compatible endpoint",
        ),
        model: Optional[str] = typer.Option(None, "--model", help="Default model id (e.g. 'openai/gpt-4o-mini')"),
        skip_memory: bool = typer.Option(False, "--skip-memory", help="Skip Step 2 (Persistent Memory)"),
        skip_protein_design: bool = typer.Option(
            False, "--skip-protein-design", help="Skip Step 3 (Protein Design compute settings)"
        ),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            help="Run without prompts (requires flags for any missing field)",
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip all confirm prompts"),
        reset: bool = typer.Option(
            False,
            "--reset",
            help="Re-run the wizard over an existing config (does not erase it; each step keeps current values as defaults)",
        ),
        skip_test: bool = typer.Option(
            False,
            "--skip-test",
            help="Skip the one-shot test message (avoids a billed call; connectivity is still checked)",
        ),
    ) -> None:
        """Configure LLM, memory and Protein Design compute."""
        run_wizard(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            skip_memory=skip_memory,
            skip_protein_design=skip_protein_design,
            non_interactive=non_interactive,
            yes=yes,
            reset=reset,
            skip_test=skip_test,
        )


__all__ = ["ensure_ready_to_start", "register", "run_wizard"]
