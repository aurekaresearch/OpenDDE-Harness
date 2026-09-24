"""Four-step onboarding wizard: LLM provider → memory → protein design → import.

Goal: get a new user from ``pip install`` to a working agent in a few
minutes, without ever opening ``~/.opendde_harness/config.json`` or
the memory root's config toml.

Steps:
  0. Language
  1. LLM provider (required; the TUI's ``/login`` at a terminal, then the
     connectivity check, the default model and a test message)
  2. Long-term memory (optional; one question -- it runs on the default model)
  3. Protein Design compute and folding configuration
  4. Web search (optional)
  5. Done

Step 1 writes through the gateway's own handlers (``tui_rpc.methods.model``),
which is what makes it the same flow as the TUI's; every other write goes
through the ``update_providers`` / ``update`` ops libraries. This module owns
the UX layer, not config-schema knowledge. The long-term memory step is the
memory plugin's own (``plugin.memory.longterm.onboard``); this module only
calls it.

Navigation: questionary 2.1.1 has no first-class cross-screen "back", so the
wizard is a screen state machine and back is expressed as a ``Back`` row on
the screens that support it; Steps 2 through 4 are optional (re-run
``onboard`` to change them). Ctrl+C exits at any point, keeping whatever was
already written.
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional
from uuid import uuid4

import typer
from rich.console import Console

from opendde_harness.cli import _choice
from opendde_harness.cli import _chrome as chrome
from opendde_harness.cli._helpers import (
    DEFAULT_PROBE_MESSAGE,
    print_probe_troubleshooting,
    send_probe,
)
from opendde_harness.cli._theme import POINTER, QMARK
from opendde_harness.providers import login_flow, model_id, pi_ids
from opendde_harness.providers.auth import CRED_ENDPOINT, CRED_KEY, CRED_OAUTH, credential_kind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from concurrent.futures import Future
    from typing import Coroutine, Iterator


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

_TOTAL_STEPS = 4

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
# Step 1 -- pi's /login, at the terminal.
#
# The flow is the TUI's ``/login`` (``ui-tui/src/selectors/authSelector.ts``),
# which is pi's: the authentication method, then the provider, then the sign-in
# or the key. It runs against the gateway's own handlers
# (``tui_rpc.methods.model``), so what the wizard can connect is what the TUI
# can, by construction; and its words are :mod:`providers.login_flow`'s, which
# is where they are written once for this side and held to the TUI's.
#
# Models are not chosen here, as they are not in ``/login``. What the wizard
# adds after a provider is connected is its own: a check that the provider
# answers, the default model, and a test message -- the things a first run has
# to settle before the TUI can open at all.
# ---------------------------------------------------------------------------

#: The provider list's one row that is not a provider: declaring an endpoint
#: the gateway has never heard of. Ours, and the one thing in the flow that is
#: not pi's; it sits under the key method because an endpoint is reached by one.
_DECLARE_ENDPOINT = object()

_QUESTIONARY_INSTALL_HINT = (
    "[error]Missing dependency:[/error] [accent]questionary[/accent] is required for "
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

    # The lockup the TUI opens with, then the ask. Bilingual, since no language
    # is chosen yet.
    chrome.lockup(console)
    chrome.caption(console, "Setup · 配置向导")
    console.print()

    picked = _choice.row(
        "Language / 语言",
        [("English", "en"), ("中文(简体)", "zh")],
        default=_LANG,  # preselect the saved language on a re-run
    )
    if picked is None:
        raise typer.Exit(1)
    _LANG = picked


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _step_header(n: int, title: str) -> None:
    chrome.step(console, number=n, total=_TOTAL_STEPS, title=title, word=_t("Step", "步骤"))


def _check_tty_or_die(non_interactive: bool) -> None:
    """Bail when stdout isn't a TTY and the user didn't opt into headless mode."""
    if non_interactive:
        return
    if not sys.stdout.isatty():
        console.print(
            "[error]Non-interactive terminal detected.[/error]\n"
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

    Which provider answers is not re-derived here. The model id's prefix is the
    whole rule and the config is what applies it, so the config is asked --
    re-splitting the id in this gate is how it came to disagree with
    ``ddeharness status`` about a signed-in provider. What is left to ask is
    whether that provider's credentials are actually on disk, which is the one
    thing routing takes on trust for a sign-in.
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


def _reusable_config() -> bool:
    """Whether the configuration on disk is one this build can carry forward.

    The rule is the shape, not the release. A build that did not change
    ``config.json``'s shape has nothing to regenerate, and a run of the wizard
    to fix one step should not cost the other three -- which is what starting
    clean every time cost. A shape this build does not know is not migrated
    (this program never migrates): it is moved aside and the run starts clean.

    Unreadable counts as not reusable. A file too damaged to parse cannot be
    reused field by field, and the wizard is a better answer than a traceback.
    """
    from opendde_harness.config.loader import get_config_path
    from opendde_harness.config.schema import CONFIG_SCHEMA_VERSION

    if not get_config_path().exists():
        return False
    try:
        written = _load_raw_config().get("schemaVersion")
    except Exception:  # noqa: BLE001 - the read path raises on a damaged file
        return False
    return written == CONFIG_SCHEMA_VERSION


def _retire_existing_setup() -> Optional[Path]:
    """Move every configuration file aside so the wizard starts from nothing.

    The wizard writes every section it walks, so a run over old files left
    behind whatever those files carried and this run did not touch -- a retired
    field, a provider shape from an earlier release, a default model naming an
    entry that was removed, a checkpoint chosen by a build before the first
    release -- and the next load refused the whole file for it, or a later step
    quietly kept the value. Nothing is inherited now: the main config, the
    migration log, the prepared-assets record, the compute service's record of
    its container, and the memory service's own settings all move into one
    ``backup-<timestamp>/`` beside them, and a memory server running on the old
    settings is stopped first. Weights, the workspace (sessions, memory notes),
    task records and sign-in grants are not configuration and stay where they
    are.
    """
    import shutil
    from datetime import datetime

    from opendde_harness.config.loader import get_config_path
    from opendde_harness.config.paths import get_data_dir
    from opendde_harness.plugin.memory.longterm import _library, _server
    from opendde_harness.plugin.memory.longterm.settings import memory_root

    data = get_data_dir()
    memory = memory_root()
    candidates = [
        get_config_path(),
        data / "config.migrations.json",
        data / "compute-assets.json",
        data / "compute" / "local.json",
        data / "compute" / "local.lock",
        memory / _library.CONFIG_FILENAME,
        memory / _library.OME_CONFIG_FILENAME,
    ]
    present = [path for path in candidates if path.exists()]
    if not present:
        return None
    if any(path.parent == memory for path in present):
        try:
            _server.stop_recorded_server(memory)
        except Exception:  # noqa: BLE001 - a server that will not stop is not a reason to keep its settings
            pass
    backup = data / f"backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    moved: list[str] = []
    for path in present:
        relative = path.relative_to(data) if path.is_relative_to(data) else Path(path.name)
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        moved.append(relative.as_posix())
    console.print(
        _t(
            f"  [dim]Previous configuration moved to {backup.name}/ ({', '.join(moved)}); this run starts clean.[/dim]",
            f"  [dim]原有配置已移到 {backup.name}/（{', '.join(moved)}）；本次从空白开始。[/dim]",
        )
    )
    return backup


def _bootstrap_empty_config() -> None:
    """Make sure ``~/.opendde_harness/config.json`` + workspace dir exist before we patch.

    We seed the user-facing extension defaults (memory / plugins / skillForge).
    The wizard's memory step then sets ``memory.backend`` to the bundled backend
    or to ``None``, per the one question it asks.

    Seeding runs on EVERY onboard, not just a brand-new config: the writer is
    ``setdefault``-based (non-clobbering), so it backfills these blocks into a
    pre-existing config that predates them without touching any value the user
    already set. The base ``Config()`` is only written when the file is absent —
    overwriting an existing file there would clobber it.
    """
    from opendde_harness.config.loader import get_config_path, load_config, save_config
    from opendde_harness.utils.helpers import sync_workspace_templates

    path = get_config_path()
    if not path.exists():
        save_config(load_config())  # writes default Config() to disk
    from opendde_harness.config.update import init_extension_block_defaults

    init_extension_block_defaults()
    workspace = load_config().workspace_path
    workspace.mkdir(parents=True, exist_ok=True)
    # Silent: the wizard is mid-screen, and which template files it copied is
    # its own bookkeeping rather than something the user chose.
    sync_workspace_templates(workspace, silent=True)


# ---------------------------------------------------------------------------
# Step 1 — provider primitives
# ---------------------------------------------------------------------------


def _provider_label(name: str) -> str:
    """What to call this provider on screen: pi's own name for it, else the id.

    Nothing is invented and nothing is translated -- a provider id has one
    spelling worldwide and so does a vendor's name, so both languages show the
    same word. A provider this config declares has no pi name, and its id is
    the name the user chose for it, which is the right thing to show.
    """
    return pi_ids.display_name(name)


def _validate_provider_name(name: str) -> str:
    """Resolve a provider name typed by hand to the id its entry is written under.

    There is nothing to normalize: a pi provider id has exactly one spelling,
    and a name that is not one names a provider this config declares -- which is
    a perfectly good answer, since that is how a self-hosted server is reached.
    The one case worth stopping for is a near-miss of a pi id: written as given
    it would declare a provider nobody serves, and then be refused for having no
    address, which is a true sentence about the wrong problem.
    ``auth.key_refusal`` is what says which name that is.
    """
    from opendde_harness.providers.auth import key_refusal

    candidate = (name or "").strip()
    removed = pi_ids.removed_message(candidate)
    if removed:
        raise typer.BadParameter(removed)
    refusal = key_refusal(candidate)
    if refusal:
        raise typer.BadParameter(refusal)
    if not candidate or "/" in candidate or any(c.isspace() for c in candidate):
        raise typer.BadParameter(
            f"{name!r} is not a provider id: an id carries no slash and no spaces, "
            "because it is the prefix every one of that provider's model ids starts with."
        )
    return candidate


def _stored_entry(provider: str) -> Optional[dict[str, Any]]:
    """This provider's entry as flat plaintext fields, or ``None`` when it has none.

    "No entry" and "an entry with nothing in it" are different states: an entry
    exists because somebody wrote it, so a provider without one has never been
    configured -- and that is exactly what a rollback has to be able to put back.
    """
    from opendde_harness.config.update_providers import get_provider_config

    stored = _load_raw_config().get("providers")
    if not isinstance(stored, dict) or provider not in stored:
        return None
    entry = get_provider_config(provider, redact_secrets=False)
    # The flat view names the models and drops what each row declares. A
    # rollback has to put a row back as it stood, so the list comes from the
    # file itself -- writing back the flattened names would quietly turn every
    # row into a bare id and lose the context window somebody typed.
    models = stored[provider].get("models")
    if isinstance(models, list):
        entry["models"] = models
    return entry


def _stored_kind(provider: str) -> str:
    """How this provider is reached, derived from what is stored for it.

    One question, asked of the entry rather than of a table of ours:
    ``auth.credential_kind`` reads the entry's own ``login`` and whether pi
    ships the id, and every branch the wizard takes about an already-configured
    provider -- which field to re-prompt, what a failure offers, what "remove"
    clears -- follows from it. Answering it independently at each of those sites
    is what used to make them disagree.
    """
    return credential_kind(provider, _stored_entry(provider))


def _is_declared(provider: str) -> bool:
    """Does this config say where the provider is, rather than pi?

    A different question from :func:`_stored_kind`, which answers what credential
    to ask for. ``baseUrl`` is what tells the two kinds apart -- the schema's own
    rule -- and what follows from it is: the address is ours to re-prompt, and
    the entry's ``models`` list is that provider's whole catalogue, so a model
    chosen here has to join it. That stays true of the one built-in id
    configured this way, Azure, whose resource belongs to the tenant and whose
    deployment names pi has never heard of.
    """
    return bool((_stored_entry(provider) or {}).get("base_url"))


def _split_ids(raw: str) -> list[str]:
    """A comma-separated answer as a list of ids, in the order it was typed."""
    return list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


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

    Rows are titled with the id the endpoint serves: the provider was chosen one
    step earlier, so repeating its name on every row says nothing. The value
    behind each row is the qualified id that gets stored.
    """
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    models = list(dict.fromkeys(model.strip() for model in models if model.strip()))
    selected_default = default_model if default_model in models else next(iter(models), _MANUAL_MODEL)
    listed = [
        questionary.Choice(model_id.display(provider, model) if provider else model, value=model) for model in models
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
            placeholder=_placeholder(_t("empty ↵ to return to the list", "留空回车返回模型列表")),
            style=OPENDDE_HARNESS_STYLE,
            qmark=_QMARK,
        ).ask()
        if chosen is None:
            raise typer.Exit(1)
        if chosen.strip():
            return chosen.strip()


def _verify_provider(provider: str, *, skip_test: bool = False) -> tuple[bool, str, Optional[list[str]]]:
    """Ask the provider for a few tokens, to verify what was just stored.

    Returns ``(ok, status, model_ids)``. ``status`` is one of the ops-library
    failure codes (``not_configured`` / ``auth`` / ``timeout`` / …) and drives
    the failure submenu's wording; ``model_ids`` is what the model layer serves
    for this provider -- bare ids, as the endpoint serves them -- which the model
    step offers as suggestions.

    It used to be a free ``GET /v1/models``, which told the truth about a key and
    nothing about the path a turn takes: seven of the vendors publish no such
    route, and the wizard had a whole branch for saying so and skipping. This
    asks the model layer the question the wizard is actually asking -- can this
    provider answer -- and pays a handful of tokens for it.
    """
    from opendde_harness.config.update_providers import test_provider as probe

    # Worded by what the provider is reached by. A declared endpoint may hold no
    # key at all, and "verifying your API key" there describes a field the user
    # was never asked for; a sign-in's credential is not in this file.
    kind = _stored_kind(provider)
    if kind == CRED_ENDPOINT:
        label = _t("Reaching the endpoint…", "正在连接端点…")
    elif kind == CRED_OAUTH:
        label = _t("Checking the sign-in…", "正在检查登录状态…")
    else:
        label = _t("Verifying your API key…", "正在验证 API Key…")
    with chrome.working(console, label):
        result = probe(provider)
    if result["ok"]:
        models = result.get("models_count")
        suffix = _t(f" ({models} models available)", f"（共 {models} 个可用模型）") if models else ""
        chrome.done(console, _t(f"Connected!{suffix}", f"连接成功！{suffix}"))
        return True, "valid", result.get("model_ids")

    status = result.get("status", "unknown")
    if status in {"no_node", "no_bundle"} and skip_test:
        # The model layer is not runnable here, and the user asked for no
        # connectivity test. Refusing a key that may be perfectly good, over a
        # check they declined, is the wizard failing on its own machinery.
        console.print(
            _t(
                "  [dim]Skipping the connectivity check (the model service is not runnable here) (--skip-test).[/dim]",
                "  [dim]跳过连通性检查(此机器无法运行模型服务)(--skip-test)。[/dim]",
            )
        )
        return True, "skipped", None
    hint_map = {
        "auth": _t(
            "Auth failed: the API key was refused — check for typos / stray spaces.",
            "鉴权失败:API Key 被拒绝 — 检查有无拼写错误或多余空格。",
        ),
        "not_configured": _t(
            "No credential reached the provider — the key was not stored where it is read.",
            "服务商没有收到凭据 — Key 未写入被读取的位置。",
        ),
        "not_served": _t(
            "This entry is not complete enough to be used — it needs an address and a model.",
            "该配置还不完整 — 需要地址和模型。",
        ),
        "no_model": _t(
            "No model is configured for this provider — name one and retry.",
            "该服务商尚未配置模型 — 填写一个后重试。",
        ),
        "timeout": _t(
            "No answer in time — check network / proxy / VPN.",
            "超时未响应 — 检查网络 / 代理 / VPN。",
        ),
        "oauth": _t(
            f"Run: ddeharness provider login {provider}",
            f"请运行:ddeharness provider login {provider}",
        ),
        "no_node": _t(
            "The model service needs Node — run `ddeharness doctor`.",
            "模型服务需要 Node — 运行 `ddeharness doctor`。",
        ),
        "no_bundle": _t(
            "The model service is not built — run `npm run build` in ui-tui/.",
            "模型服务尚未构建 — 在 ui-tui/ 中运行 `npm run build`。",
        ),
    }
    msg = hint_map.get(status, _t(f"Verification failed: {status}", f"验证失败:{status}"))
    console.print(f"  [warn]✗ {msg}[/warn]" + (f"  [dim]{result['error']}[/dim]" if result.get("error") else ""))
    return False, status, None


# ---------------------------------------------------------------------------
# Step 1 -- the flow itself
# ---------------------------------------------------------------------------


def _flow(text: login_flow.Text, **fields: Any) -> str:
    """One of the flow's sentences, in the wizard's language, filled in."""
    return _t(*text).format(**fields)


class _Gateway:
    """The gateway's handlers, on a loop of their own beside the prompts.

    The handlers are coroutines, and a prompt blocks the thread it runs on. So
    the handlers get a loop on a worker thread and the prompts keep the main
    thread -- where Ctrl+C lands, and where every other prompt of the wizard
    runs. A sign-in's steps cross over through :meth:`pushed`, the way
    ``login.step`` frames reach the TUI, and the thread ends the model service
    it started before it stops, on the loop that owns the child's pipes.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="onboard-gateway", daemon=True)
        self._frames: queue.Queue[Any] = queue.Queue()
        #: The handlers started here, so what is still running can be ended.
        self._started: list[Future[Any]] = []
        self._thread.start()

    def __enter__(self) -> "_Gateway":
        return self

    def __exit__(self, *_exc: Any) -> None:
        from opendde_harness.providers.pi_service import shutdown_service

        # A handler still running -- a sign-in left at Ctrl+C -- is ended
        # first, which is what stops a device-code poll at the vendor. Only
        # those: the loop also carries the model service's own tasks, and a
        # shutdown that finds its reader cancelled under it ends cancelled too.
        for running in self._started:
            running.cancel()
        try:
            self.call(shutdown_service())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop.close()

    def call(self, coro: "Coroutine[Any, Any, Any]") -> Any:
        """Run one handler to its answer, on the loop, and wait here for it."""
        return self.start(coro).result()

    def start(self, coro: "Coroutine[Any, Any, Any]") -> "Future[Any]":
        """Start one handler on the loop; its steps arrive through :meth:`pushed`."""
        started = asyncio.run_coroutine_threadsafe(coro, self._loop)
        self._started.append(started)
        return started

    async def push(self, frame: dict[str, Any]) -> None:
        """What the gateway would send the TUI: kept for the main thread to show."""
        self._frames.put(frame)

    def pushed(self, running: "Future[Any]") -> "Iterator[dict[str, Any]]":
        """Every frame pushed while ``running`` runs, in order, and then no more."""
        running.add_done_callback(lambda _done: self._frames.put(None))
        while True:
            frame = self._frames.get()
            if frame is None:
                return
            yield frame


def _refusal(exc: Exception) -> str:
    """What a refused handler said: its own sentence, never its payload."""
    from opendde_harness.tui_rpc.errors import RpcError

    return exc.detail if isinstance(exc, RpcError) and exc.detail else str(exc)


def _connect_interactive(named: Optional[str] = None) -> tuple[str, list[str]]:
    """pi's ``/login`` at the terminal: a method, a provider, a sign-in or a key.

    Returns the connected provider and, for an endpoint declared here, the
    models it was found to serve; every other provider's list is the model
    layer's to report at the check that follows. ``named`` is ``--provider``,
    which goes straight to that provider the way ``/login <provider>`` does.
    """
    from opendde_harness.tui_rpc.methods.model import model_options

    with _Gateway() as gateway:
        with chrome.working(console, _flow(login_flow.LOADING_PROVIDERS)):
            rows = gateway.call(model_options({"include_catalog": False}))["providers"]

        if named:
            outcome = _connect_named(gateway, rows, named)
            if outcome is not _BACK:
                return outcome

        while True:
            method = _ask_method()
            while True:
                picked = _select_provider(rows, method)
                if picked is _BACK:
                    break
                if picked is _DECLARE_ENDPOINT:
                    outcome = _declare_endpoint(gateway)
                elif method == CRED_OAUTH:
                    outcome = _sign_in(gateway, picked)
                else:
                    outcome = _connect_with_key(gateway, picked)
                if outcome is not _BACK:
                    return outcome


def _connect_named(gateway: _Gateway, rows: list[dict[str, Any]], named: str) -> Any:
    """``/login <provider>``: straight to the one it names.

    One way in goes straight to it; both ask pi's question first, titled with
    the provider's name and worded with its own label where pi wrote one. A
    name no row carries falls through to the list, as the TUI's does.
    """
    wanted = named.strip().lower()
    row = next((row for row in rows if wanted in (row["slug"].lower(), row["name"].lower())), None)
    if row is None:
        return _BACK
    methods = login_flow.login_methods(row)
    if not methods:
        return _BACK
    method = methods[0]
    if len(methods) > 1:
        method = _choice.row(
            _flow(login_flow.SELECT_METHOD_FOR, name=row["name"]),
            [
                (row.get("login_label") or _flow(login_flow.SIGN_IN_WITH_ACCOUNT), CRED_OAUTH),
                (_flow(login_flow.SIGN_IN_WITH_API_KEY), CRED_KEY),
            ],
            default=CRED_OAUTH,
        )
        if method is None:
            raise typer.Exit(1)
    return _sign_in(gateway, row) if method == CRED_OAUTH else _connect_with_key(gateway, row)


def _ask_method() -> str:
    """pi's first question: which way in, before which provider."""
    picked = _choice.row(
        _flow(login_flow.SELECT_METHOD),
        [(_flow(label), method) for method, label in login_flow.METHODS],
        default=CRED_OAUTH,
    )
    if picked is None:
        raise typer.Exit(1)
    return picked


def _row_title(row: dict[str, Any], method: str) -> str:
    """One provider row: the name, and pi's status marker when it says something.

    The TUI's row carries the id and the marker on every line. Here the config
    is a clean one, so "• unconfigured" on forty lines says nothing forty
    times, and the id repeats the name beside it; what is worth a column is a
    key the environment already supplies, or a credential of the other kind.
    """
    marker = login_flow.status_marker(row, method)
    if marker == login_flow.UNCONFIGURED:
        return str(row["name"])
    return f"{row['name']}  ·  {_flow(marker)}"


def _select_provider(rows: list[dict[str, Any]], method: str) -> Any:
    """pi's "Select provider to configure:", filtered to the method chosen.

    The featured few and nothing else, as a grid of dots: a wizard screen is
    read at a glance, and the forty rows the TUI keeps behind a filter would
    bury the handful almost everybody picks. The line above says where the
    rest are. Under the key method the last option is ours, the endpoint the
    gateway has never heard of. Escape is the TUI's own way back, to the
    method question. Returns a row, :data:`_DECLARE_ENDPOINT`, or :data:`_BACK`.
    """
    featured = login_flow.offered(rows, method)
    if not featured and method == CRED_OAUTH:
        chrome.caption(console, _t("No providers available", "没有可用的服务商"))
        return _BACK

    chrome.hint(
        console,
        _t(
            "Every other provider is a /login away once the setup is done.",
            "其他服务商在配置完成后，于 TUI 内用 /login 接入。",
        ),
    )
    options: list[tuple[str, Any]] = [(_row_title(row, method), row) for row in featured]
    if method == CRED_KEY:
        options.append((_flow(login_flow.ENDPOINT_ROW), _DECLARE_ENDPOINT))
    picked = _choice.row(_flow(login_flow.SELECT_PROVIDER), options, back=_BACK)
    if picked is None:
        raise typer.Exit(1)
    return picked


def _placeholder(text: str) -> Any:
    """A faint in-field placeholder: gone the moment they type, and gone from
    the record once the prompt is answered. In the palette's disabled grey."""
    return [("class:disabled", text)]


def _prompt_text(
    label: str,
    *,
    instruction: str,
    default: str = "",
    placeholder: str = "",
    back: bool = False,
    optional: bool = False,
) -> Any:
    """One text field of a form: the label, the hint beside it, the value.

    An empty submit is :data:`_BACK` on the field that offers it -- the first
    of a form, where the TUI's Esc would leave -- an answer on an optional one,
    and refused everywhere else.
    """
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    def _validate(value: str) -> Any:
        if value.strip() or back or optional:
            return True
        return _t("Required.", "必填。")

    value = questionary.text(
        label,
        default=default,
        validate=_validate,
        instruction=instruction,
        placeholder=_placeholder(placeholder or _t("empty ↵ to go back", "留空回车返回"))
        if placeholder or back
        else None,
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
    ).ask()
    if value is None:
        raise typer.Exit(1)
    value = value.strip()
    if back and not value:
        return _BACK
    return value


def _prompt_secret(label: str, *, instruction: str, back: bool = False) -> Any:
    """One masked field. Judged as the TUI judges it (``validateSecret``): the
    value is trimmed, and one with control characters in it is refused -- a
    key with a line break pasted into it would be sent broken and stored
    broken. ``back`` gives an empty submit the meaning the TUI's Esc has."""
    questionary = _require_questionary()
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

    def _validate(value: str) -> Any:
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value.strip()):
            return _flow(login_flow.CONTROL_CHARACTERS)
        return True

    value = questionary.password(
        label,
        validate=_validate,
        instruction=instruction,
        placeholder=_placeholder(_t("empty ↵ to go back", "留空回车返回")) if back else None,
        style=OPENDDE_HARNESS_STYLE,
        qmark=_QMARK,
    ).ask()
    if value is None:
        raise typer.Exit(1)
    value = value.strip()
    if back and not value:
        return _BACK
    return value


def _pick_option(message: str, options: list[dict[str, Any]]) -> Optional[str]:
    """pi's own menu inside a sign-in, answered with the option's id, as a row
    of dots the way the wizard asks every short question. pi's first option is
    its default."""
    labels = [(str(option.get("label") or option.get("id")), str(option.get("id"))) for option in options]
    return _choice.row(message, labels, default=labels[0][1])


def _link(url: str, label: Optional[str] = None) -> None:
    """A link the way the TUI's login view draws one: the address as a
    hyperlink, and the click hint under it."""
    console.print(f"{chrome.INDENT}[accent][link={url}]{label or url}[/link][/accent]")
    chrome.hint(console, _flow(login_flow.CLICK_TO_OPEN))


def _show_login_step(step: dict[str, Any]) -> None:
    """One step that is only there to be read, as the TUI's login view shows it.

    pi's lines, in pi's order, with pi's words: the URL on its own line, the
    code as "Enter code: <code>" and the wait after it; a sign-in URL with its
    instructions, and the browser opened for it -- as pi's dialog does, and not
    for a device code, whose URL is meant for another machine.
    """
    kind = step.get("type")
    if kind == "device_code":
        _link(str(step.get("verificationUri") or ""))
        console.print(f"{chrome.INDENT}[warn]{_flow(login_flow.ENTER_CODE, code=step.get('userCode') or '')}[/warn]")
        chrome.caption(console, _flow(login_flow.WAITING_FOR_AUTHENTICATION))
        return
    if kind == "auth_url":
        from opendde_harness.cli.provider_commands import ConsoleInteraction

        url = str(step.get("url") or "")
        _link(url)
        if step.get("instructions"):
            console.print(f"{chrome.INDENT}[warn]{step['instructions']}[/warn]")
        ConsoleInteraction().open_url(url)
        return
    if kind == "info":
        console.print(f"{chrome.INDENT}{step.get('message') or ''}")
        for link in step.get("links") or ():
            if isinstance(link, dict) and link.get("url"):
                _link(str(link["url"]), link.get("label"))
        return
    chrome.caption(console, str(step.get("message") or ""))


def _sign_in(gateway: _Gateway, row: dict[str, Any]) -> Any:
    """Run pi's own sign-in here and now, showing its steps as they arrive.

    The gateway holds the flow (``model.login``) and pushes every step it
    reports; the two it stops on are answered from here -- the login-method
    menu with pi's own labels, and the authorization code a browser login
    falls back to asking for. The request settles when a credential is stored,
    which is as long as the person takes. A failure is said the way pi says
    it, and the list is where it leaves them: entering the provider is what
    starts a sign-in, so there is nothing to retry from but the list.
    """
    from opendde_harness.tui_rpc.methods.model import model_login, model_login_answer, model_login_cancel

    name, slug = row["name"], row["slug"]
    chrome.heading(console, _flow(login_flow.SIGN_IN_TO, name=name))
    chrome.caption(console, _flow(login_flow.SIGN_IN_SUBTITLE))
    chrome.caption(console, _flow(login_flow.STARTING_SIGN_IN))
    # Chosen here so it is known before the first step arrives, as the TUI
    # chooses it: a step carrying another id is not this sign-in's to show.
    login_id = uuid4().hex
    running = gateway.start(model_login({"provider": slug, "login_id": login_id}, send_frame=gateway.push))
    try:
        for frame in gateway.pushed(running):
            params = frame.get("params") or {}
            step = params.get("step") or {}
            if params.get("login_id") != login_id:
                continue
            if step.get("type") == "select":
                answer = _pick_option(str(step.get("message") or ""), list(step.get("options") or ()))
            elif step.get("type") == "manual_code":
                answer = _prompt_text(str(step.get("message") or ""), instruction=str(step.get("placeholder") or ""))
            else:
                _show_login_step(step)
                continue
            if answer is None:
                gateway.call(model_login_cancel({"login_id": login_id}))
                raise typer.Exit(1)
            gateway.call(model_login_answer({"login_id": login_id, "answer": answer}))
        result = running.result()
    except Exception as exc:  # noqa: BLE001 - the gateway's own sentence, then the list
        console.print(f"{chrome.INDENT}[warn]{_flow(login_flow.FAILED_LOGIN, name=name, why=_refusal(exc))}[/warn]")
        return _BACK
    chrome.done(console, _flow(login_flow.LOGGED_IN, name=result["provider"]["name"]))
    return slug, []


def _connect_with_key(gateway: _Gateway, row: dict[str, Any]) -> Any:
    """pi's key form: one masked field, judged by the gateway.

    What the field says beside it is the row's own answer -- a variable that
    already supplies the key, a key this provider requires, or one it can do
    without -- and what happens to the value is the gateway's: it stores the
    key and answers with the row as it now stands, connected or not.
    """
    from opendde_harness.tui_rpc.methods.model import model_save_key

    name, slug = row["name"], row["slug"]
    env = row.get("key_env")
    required = bool(row.get("needs_api_key")) and not env
    chrome.heading(console, _flow(login_flow.CONNECT, name=name))
    chrome.caption(
        console, _flow(login_flow.KEY_SUBTITLE_REPLACES if row.get("authenticated") else login_flow.KEY_SUBTITLE)
    )
    if env:
        instruction = _flow(login_flow.KEY_HINT_ENV, env=env)
    elif required:
        instruction = _flow(login_flow.KEY_HINT_REQUIRED)
    else:
        instruction = _flow(login_flow.KEY_HINT_OPTIONAL)

    while True:
        key = _prompt_secret(_flow(login_flow.API_KEY), instruction=instruction, back=required)
        if key is _BACK:
            return _BACK
        try:
            with chrome.working(console, _flow(login_flow.WORKING)):
                connected = gateway.call(model_save_key({"slug": slug, "api_key": key}))["provider"]
        except Exception as exc:  # noqa: BLE001 - shown on the form, as the TUI shows it
            console.print(f"{chrome.INDENT}[warn]{_refusal(exc)}[/warn]")
            continue
        if connected.get("authenticated"):
            chrome.done(console, _flow(login_flow.LOGGED_IN, name=connected["name"]))
            return slug, []
        console.print(f"{chrome.INDENT}[warn]{connected.get('warning') or _flow(login_flow.KEY_REFUSED)}[/warn]")


def _declare_endpoint(gateway: _Gateway) -> Any:
    """The four fields of the TUI's endpoint form, asked one after another.

    The id, the address, a key only if the server wants one, and the model ids
    only for an endpoint that publishes none: the gateway asks the endpoint what
    it serves the moment it is declared, the way pi reads OpenRouter's list. A
    refusal is the gateway's own sentence, and the form is asked again with
    what was typed kept.
    """
    from opendde_harness.tui_rpc.methods.model import model_declare_provider

    chrome.heading(console, _flow(login_flow.ADD_ENDPOINT))
    chrome.caption(console, _flow(login_flow.ENDPOINT_SUBTITLE))
    typed = {"provider": "", "base_url": "", "model": ""}
    while True:
        provider = _prompt_text(
            _flow(login_flow.PROVIDER_ID),
            instruction=_flow(login_flow.PROVIDER_ID_HINT),
            default=typed["provider"],
            placeholder=login_flow.PROVIDER_ID_PLACEHOLDER,
            back=True,
        )
        if provider is _BACK:
            return _BACK
        base_url = _prompt_text(
            _flow(login_flow.BASE_URL),
            instruction=_flow(login_flow.BASE_URL_HINT),
            default=typed["base_url"],
            placeholder=login_flow.BASE_URL_PLACEHOLDER,
        )
        api_key = _prompt_secret(_flow(login_flow.API_KEY), instruction=_flow(login_flow.ENDPOINT_KEY_HINT))
        model = _prompt_text(
            _flow(login_flow.MODEL_IDS),
            instruction=_flow(login_flow.MODEL_IDS_HINT),
            default=typed["model"],
            placeholder=_flow(login_flow.MODEL_IDS_PLACEHOLDER),
            optional=True,
        )
        typed = {"provider": provider, "base_url": base_url, "model": model}
        try:
            with chrome.working(console, _flow(login_flow.WORKING)):
                declared = gateway.call(
                    model_declare_provider(
                        {"provider": provider, "base_url": base_url, "api_key": api_key, "model": model}
                    )
                )["provider"]
        except Exception as exc:  # noqa: BLE001 - shown on the form, as the TUI shows it
            console.print(f"{chrome.INDENT}[warn]{_refusal(exc)}[/warn]")
            continue
        # Bare, as the model step stores them: the row's ids carry the prefix
        # that routes them, and the step joins it back on when it writes.
        served = [model_id.bare(model) for model in declared["models"]]
        count = len(served)
        chrome.done(
            console,
            _flow(
                login_flow.LOGGED_IN_ENDPOINT,
                name=declared["name"],
                count=count,
                plural="" if count == 1 else "s",
                url=base_url,
            ),
        )
        return declared["slug"], served


def _connect_headless(
    provider: str, *, api_key: Optional[str], base_url: Optional[str], model: Optional[str]
) -> tuple[str, list[str]]:
    """The same two handlers, fed from flags rather than prompts.

    One of pi's own takes a key (or the variable that supplies it) and nothing
    else; a name pi does not ship is an endpoint declared here, and needs its
    address. A provider that is only signed in to has no headless way in, and a
    built-in whose address is the tenant's own -- Azure -- is written with
    ``ddeharness provider set``, which takes every field the entry has.
    """
    from opendde_harness.providers.auth import env_key_name
    from opendde_harness.tui_rpc.methods.model import model_declare_provider, model_save_key

    provider = _validate_provider_name(provider)
    with _Gateway() as gateway:
        try:
            if pi_ids.is_builtin(provider):
                if base_url:
                    raise typer.BadParameter(
                        f"pi carries {provider}'s address, so --base-url does not apply to it. An address of "
                        f"your own is written with `ddeharness provider set {provider} --base-url <url> --api "
                        "<wire> --api-key <key>`."
                    )
                if not api_key and not env_key_name(provider) and provider not in pi_ids.AMBIENT:
                    if provider in pi_ids.OAUTH:
                        console.print(
                            "[error]A sign-in requires an interactive browser flow.[/error]\n"
                            f"Run [accent]ddeharness provider login {provider}[/accent] separately, then re-run onboard."
                        )
                        raise typer.Exit(2)
                    raise typer.BadParameter("--api-key is required in non-interactive mode")
                gateway.call(model_save_key({"slug": provider, "api_key": api_key or ""}))
                return provider, []
            if not base_url:
                raise typer.BadParameter(f"--base-url is required for {provider} in non-interactive mode")
            declared = gateway.call(
                model_declare_provider(
                    {"provider": provider, "base_url": base_url, "api_key": api_key or "", "model": model or ""}
                )
            )["provider"]
        except Exception as exc:
            from opendde_harness.tui_rpc.errors import RpcError

            if isinstance(exc, RpcError):
                raise typer.BadParameter(_refusal(exc)) from exc
            raise
    # The ids typed lead: ``--model`` is the catalogue of a declared provider,
    # and its first entry is the default with nobody at the terminal.
    return provider, _split_ids(model) if model else [model_id.bare(served) for served in declared["models"]]


def _reconnect(provider: str, *, sign_in: bool) -> bool:
    """Back into the key form or the sign-in for a provider already picked --
    what a failed check offers. True once it is connected again."""
    from opendde_harness.tui_rpc.methods.model import model_options

    with _Gateway() as gateway:
        rows = gateway.call(model_options({"include_catalog": False, "slug": provider}))["providers"]
        if not rows:
            return False
        outcome = _sign_in(gateway, rows[0]) if sign_in else _connect_with_key(gateway, rows[0])
    return outcome is not _BACK


def _memory_step(*, skip: bool, non_interactive: bool, warnings: list[str]) -> object:
    """The memory plugin's own wizard step. Imported here rather than at the
    top: the step uses this module's console and prompts, so the two would
    import each other."""
    from opendde_harness.plugin.memory.longterm import onboard as memory_onboard

    return memory_onboard.step(skip=skip, non_interactive=non_interactive, warnings=warnings)


def _memory_enabled() -> bool:
    from opendde_harness.plugin.memory.longterm import onboard as memory_onboard

    return memory_onboard.memory_enabled()


def _load_current_default_model() -> Optional[str]:
    """Read ``agents.defaults.model`` from the on-disk config, if it exists."""
    data = _load_raw_config()
    return (data or {}).get("agents", {}).get("defaults", {}).get("model") or None


def _recommended_model(provider: str) -> str:
    """The bare model id to preselect for this provider, or "" when we know none.

    The curated shortlist's first entry, which is the one this project would
    recommend for daily use. It is the only recommendation left: there is no
    per-provider spec carrying a default model any more, because a model the
    vendor has retired is a fact that goes stale in a table and never goes stale
    in :mod:`providers.common_models`, which the picker reads too.
    """
    from opendde_harness.providers.common_models import common_models_for

    shortlist = common_models_for(provider)
    return model_id.bare(shortlist[0]) if shortlist else ""


def _pick_model(
    provider: str,
    *,
    current_model: Optional[str],
    model_ids: Optional[list[str]],
    probe_status: str,
    user_provided_model: Optional[str],
    non_interactive: bool,
) -> str:
    """Decide the model string to write into ``agents.defaults.model``.

    Every exit goes through ``model_id.join``, which is the one rule for what a
    stored id looks like: "<provider id>/<model id>", the prefix naming the
    provider because nothing else does any more. The ids reaching here are bare
    -- the model service is asked for what the endpoint serves -- so qualifying
    them is not optional: a bare id names no provider and is refused by the
    schema.
    """
    if user_provided_model:
        return model_id.join(provider, user_provided_model)

    if non_interactive:
        recommended = _recommended_model(provider)
        if not recommended:
            raise typer.BadParameter(f"--model is required for provider '{provider}' (we have no default for it).")
        return model_id.join(provider, recommended)

    recommended = _recommended_model(provider)
    if current_model and model_id.provider_of(current_model) == provider:
        default_value = model_id.bare(current_model)
    else:
        default_value = recommended

    # The verification above already asked the model service what this provider
    # serves, and that answer is `model_ids`. When it has none -- the check was
    # skipped, or it failed and the user chose to continue -- the same source the
    # TUI picker reads is asked again here rather than a second list of our own,
    # so the wizard and the picker cannot offer different models for the same
    # provider. `models_for_provider` falls back to the curated shortlist when
    # the service cannot be reached at all; a provider neither answers for is
    # typed by hand, which is how a self-hosted server was always named.
    if not model_ids:
        from opendde_harness.config.loader import load_config
        from opendde_harness.providers.common_models import models_for_provider

        try:
            config = load_config()
        except Exception:  # noqa: BLE001 - an unreadable config just has no suggestions
            config = None
        known = [str(row.get("id") or "") for row in models_for_provider(config, provider) if row.get("id")]
        if known:
            # "skipped" is the one status that is nobody's failure: the model
            # service is not runnable on this machine and the user asked for no
            # connectivity check, so nothing was asked and nothing refused.
            # Saying "couldn't reach the provider" there contradicted the line
            # printed just above it and read as a failure to a user for whom
            # nothing had failed.
            console.print(
                _t(
                    "  [dim]The model service isn't runnable here - offering the models we know.[/dim]",
                    "  [dim]此机器无法运行模型服务,先列出已知的模型。[/dim]",
                )
                if probe_status == "skipped"
                else _t(
                    "  [dim]Couldn't reach the provider for its model list - offering the ones we know.[/dim]",
                    "  [dim]未能向服务商拉取模型列表,先列出已知的。[/dim]",
                )
            )
            model_ids = known

    if default_value and default_value == recommended and (not model_ids or default_value in model_ids):
        console.print(
            _t(
                f"  [dim]Default: {default_value} — recommended balance of quality/cost for daily use.[/dim]",
                f"  [dim]默认:{default_value} — 质量/成本均衡,适合日常使用。[/dim]",
            )
        )

    if model_ids:
        choices = list(dict.fromkeys(model_ids))
        if default_value not in choices:
            default_value = choices[0]
        prompt_label = _t(
            f"Default model ({len(choices)} available):",
            f"默认模型(共 {len(choices)} 个):",
        )
        # Only a provider this config declares. Everywhere else the list came
        # from pi's catalogue and holds the models that provider really serves;
        # on an endpoint the user described, the list is that entry's own
        # `models` and is the least likely to already carry the model they mean
        # -- so typing one is the first row there.
        own_endpoint = _is_declared(provider)
        chosen = _select_model_id(
            choices,
            default_model=default_value,
            label=prompt_label,
            provider=provider,
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
            return model_id.join(provider, default_value)
        raise typer.Exit(1)
    return model_id.join(provider, chosen)


def _roll_back_provider(provider: str, before: Optional[dict[str, Any]]) -> None:
    """Undo what this pass wrote, restoring the state read before it started.

    Two cases, and the difference is the one the old shape could not express: a
    provider this pass *created* is put back by removing the entry, because an
    entry emptied of its fields is not the same thing as never having been
    configured -- for a declared provider it is not even a valid entry. A
    provider that already had one is restored to it, rows and all.

    An entry naming a sign-in is left alone. Its credential is a grant in the
    model service's store, removing the entry would sign the user out of an
    account they have just signed in to, and the ops layer refuses credential
    writes for it anyway -- which is how a failed verification once took the
    whole wizard down with it.
    """
    if _stored_kind(provider) == CRED_OAUTH:
        return
    from opendde_harness.config.update_providers import reset_provider

    if before is None:
        reset_provider(provider)
        return
    _write_provider_fields(provider, before)


def _write_provider_fields(provider: str, fields: dict[str, Any]) -> None:
    """Thin wrapper that surfaces ops-library errors with friendly hints."""
    from pydantic import ValidationError

    from opendde_harness.config.update_providers import set_provider_fields

    try:
        set_provider_fields(provider, fields)
    except KeyError as exc:
        console.print(f"  [error]✗[/error] {exc}")
        raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"  [error]✗[/error] {exc}")
        raise typer.Exit(1)
    except ValidationError as exc:
        console.print(_t(f"  [error]✗ Validation failed:[/error]\n{exc}", f"  [error]✗ 校验失败:[/error]\n{exc}"))
        raise typer.Exit(1)


def _persist_default_model(model: Optional[str], provider: str) -> None:
    """Write ``agents.defaults.model`` as ``"<provider id>/<model id>"``.

    One field, because the id's prefix is the only thing that names the provider
    now. The field that used to pin one beside it is gone, and with it the case
    it caused: a stale pin outranked what an id said, so the model the wizard
    had just chosen was answered by whichever provider was pinned before, with
    that provider's key.
    """
    if not model:
        return
    from opendde_harness.config.update import set_default_model

    set_default_model(model, provider=provider)


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
            _t(
                f"  [error]✗ Test failed:[/error] {_probe_failure(exc)}",
                f"  [error]✗ 测试失败:[/error] {_probe_failure(exc)}",
            )
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
    console.print(f"  [ok]✓ {', '.join(extras)}[/ok]")
    return "ok"


# ---------------------------------------------------------------------------
# Step 1 — add one provider (used by both first-run and the "add" entry)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1 -- after the provider is connected: the check, the model, the test
# ---------------------------------------------------------------------------


def _resolve_model_with_test(
    provider: str,
    *,
    declared_models: Optional[list[str]],
    user_model_flag: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    skip_test: bool = False,
) -> Optional[str]:
    """Verify connectivity → pick the default model → send a test probe.

    On a verify or test-message failure, offers a recovery submenu (retry /
    re-pick model / sign in again or re-enter the key / switch / continue).
    Only failures stop; success auto-advances. Returns the chosen model, or
    ``None`` to signal "switch provider" (the caller rewinds to the picker).
    """
    while True:
        ok, status, model_ids = _verify_provider(provider, skip_test=skip_test)
        if not ok:
            signs_in = _stored_kind(provider) == CRED_OAUTH
            if status in {"timeout", "not_served", "no_model"}:
                # The failures a credential cannot fix: nothing answered in time,
                # or the entry is not complete enough to be used at all. Offering
                # the key form for those sends the user to the wrong field.
                options = [
                    (_t("Retry", "重试"), "retry"),
                    (_t("Continue anyway", "仍然继续"), "continue"),
                ]
            else:
                options = [
                    (_t("Sign in again", "重新登录"), "reauth")
                    if signs_in
                    else (_t("Re-enter key", "重新填 Key"), "rekey"),
                    # Also retry, because this branch takes the failures that
                    # cannot be sorted: a credential the account refused and a
                    # refresh that could not reach the network arrive as the same
                    # thing, and only one of them is fixed by signing in again.
                    (_t("Retry", "重试"), "retry"),
                    (_t("Switch provider", "更换服务商"), "switch"),
                    (_t("Continue anyway", "仍然继续"), "continue"),
                ]
            choice = _failure_choice(options, non_interactive=non_interactive)
            if choice == "retry":
                continue
            if choice in ("rekey", "reauth") and not non_interactive:
                # The same form or sign-in the provider was connected with:
                # backing out of it is choosing another provider.
                if _reconnect(provider, sign_in=choice == "reauth"):
                    continue
                return None
            if choice == "switch":
                return None
            warnings.append("provider connectivity")
            model_ids = None
        break

    # A declaration named its own catalogue a moment ago, and nothing knows
    # better than that what the provider serves: those ids were typed for this
    # endpoint. They are also what the connectivity check could not report when
    # it could not be run, which is where a self-hosted server that is not up
    # yet always lands, so they stand in for its answer.
    #
    # The flag is overwritten rather than read a second time: for a declared
    # provider ``--model`` *is* the catalogue, so re-using the raw answer as one
    # model id wrote "my-vllm/qwen3-32b,qwen3-8b" as the default.
    if declared_models:
        if not model_ids:
            model_ids = list(declared_models)
        # One id is the default outright, and so is the first of several with
        # nobody at the terminal to choose. Several and a terminal is a choice.
        user_model_flag = declared_models[0] if len(declared_models) == 1 or non_interactive else None

    current = _load_current_default_model()
    while True:
        chosen = _pick_model(
            provider,
            current_model=current,
            model_ids=model_ids,
            probe_status=status,
            user_provided_model=user_model_flag,
            non_interactive=non_interactive,
        )
        # A provider this config declares serves what its own list names, so a
        # model picked or typed here has to join that list or the model service
        # will not carry it. Idempotent, and the id is stored bare.
        if _is_declared(provider):
            from opendde_harness.config.update_providers import add_provider_model

            add_provider_model(provider, chosen)
        _persist_default_model(chosen, provider)
        if skip_test:
            return chosen
        result = _run_test_probe(
            provider,
            non_interactive=non_interactive,
            warnings=warnings,
            is_oauth=_stored_kind(provider) == CRED_OAUTH,
        )
        if result == "switch":
            return None
        if result in ("rekey", "reauth"):
            if not _reconnect(provider, sign_in=result == "reauth"):
                return None
            # Re-test the same model with the new credential (picker defaults to it).
            current = chosen
            user_model_flag = None
            continue
        if result == "repick":
            current = chosen
            user_model_flag = None
            continue
        return chosen  # ok / continue


def _step1_provider(
    *,
    provider: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    skip_test: bool = False,
) -> None:
    """Step 1: connect a provider the way ``/login`` does, then settle the model.

    The flags connect without prompts when they carry a credential or an
    address; ``--provider`` alone goes straight to that provider the way
    ``/login <provider>`` does. "Switch provider" from a failed check undoes
    what that pass wrote and comes back to the flow; the flags are used once.
    """
    _step_header(1, _t("LLM provider", "LLM 服务商"))
    chrome.caption(
        console,
        _t(
            "OpenDDE Harness's chat and reasoning are all driven by it.",
            "OpenDDE Harness 的对话与思考都由它驱动。",
        ),
    )

    while True:
        # Read before anything is written, so a failed pass can be undone: the
        # entry as it stood, or ``None`` when this provider had none at all.
        stored = _load_raw_config().get("providers") or {}
        before = {name: _stored_entry(name) for name in stored if isinstance(stored, dict)}

        if non_interactive or (provider and (api_key or base_url)):
            if not provider:
                raise typer.BadParameter("--provider is required in non-interactive mode")
            connected, declared_models = _connect_headless(provider, api_key=api_key, base_url=base_url, model=model)
        else:
            connected, declared_models = _connect_interactive(provider)

        chosen_model = _resolve_model_with_test(
            connected,
            declared_models=declared_models,
            user_model_flag=model,
            non_interactive=non_interactive,
            warnings=warnings,
            skip_test=skip_test,
        )
        if chosen_model is None:
            # "Switch provider" -- back to the flow, undoing what this pass
            # wrote and dropping the flags: they were typed for the provider
            # that just failed.
            _roll_back_provider(connected, before.get(connected))
            provider = api_key = base_url = model = None
            continue
        _persist_default_model(chosen_model, connected)
        return


def _protein_design_recap() -> str:
    """Where Protein Design compute runs and whether it answers, for the recap.

    Read from disk and probed the same way ``ddeharness doctor`` does, so the
    recap reports the state the user will find rather than what the step
    intended.
    """
    from opendde_harness.cli.onboard_compute import inspect_compute, load_protein_design_config

    config = load_protein_design_config()
    if not config:
        return _t("[warn]not configured[/warn]", "[warn]未配置[/warn]")
    report = inspect_compute(config)
    placement = {
        "local_docker": _t("local Docker", "本地 Docker"),
        "remote_service": _t("remote service", "远程服务"),
        "worker_pool": _t("worker pool", "计算节点池"),
    }.get(str(report.get("placement")), str(report.get("placement") or "—"))
    readiness = (
        _t("[ok]ready[/ok]", "[ok]就绪[/ok]")
        if report.get("ready")
        else _t("[warn]unreachable[/warn]", "[warn]未就绪[/warn]")
    )
    return f"{placement}, {readiness}"


def _print_next_steps(*, warnings: list[str], show_next_steps: bool = True) -> None:
    console.print()
    if warnings:
        console.print(f"  [warn][bold]![/bold] {_t('Setup finished with warnings', '配置完成,但有警告')}[/warn]")
        console.print()
        chrome.caption(
            console,
            _t("These items didn't pass a connectivity test: ", "以下项目未通过连通测试: ") + ", ".join(warnings),
        )
        chrome.hint(
            console,
            _t(
                "Fix them before relying on the related features — ddeharness onboard reconfigures.",
                "在依赖相关功能前请先修复 — 重新运行 ddeharness onboard 可重新配置。",
            ),
        )
    else:
        console.print(f"  [ok][bold]✓[/bold] {_t('Setup complete', '配置完成')}[/ok]")

    # Recap what was configured (read from disk) so the user has closure.
    provs = ", ".join(_provider_label(n) for n in _configured_providers()) or "—"
    mem = _t("on", "已启用") if _memory_enabled() else _t("[warn]off[/warn]", "[warn]未启用[/warn]")
    chrome.heading(console, _t("Your setup", "你的配置"))
    chrome.fields(
        console,
        [
            (_t("Provider", "服务商"), provs),
            (_t("Default model", "默认模型"), _load_current_default_model() or "—"),
            (_t("Memory", "长期记忆"), mem),
            (_t("Protein design", "蛋白设计"), _protein_design_recap()),
            (
                _t("Web search", "网页搜索"),
                _t("Brave Search API", "Brave Search API")
                if _web_search_configured()
                else _t("DuckDuckGo (no key)", "DuckDuckGo（无 key）"),
            ),
        ],
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

    chrome.heading(console, _t("Get started", "开始使用"))
    chrome.fields(
        console,
        [
            ("ddeharness", _t("start the antibody design TUI", "启动抗体设计 TUI")),
            ("ddeharness doctor", _t("check configuration and compute readiness", "检查配置与计算就绪状态")),
            ("ddeharness tracing", _t("open the results dashboard", "打开结果面板")),
        ],
        key_style="accent",
    )
    console.print()


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
    brave_api_key: Optional[str] = None,
    non_interactive: bool = False,
    yes: bool = False,
    skip_test: bool = False,
    show_next_steps: bool = True,
    fresh: bool = False,
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
            brave_api_key=brave_api_key,
            non_interactive=non_interactive,
            yes=yes,
            skip_test=skip_test,
            show_next_steps=show_next_steps,
            fresh=fresh,
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


def _web_search_configured() -> bool:
    """Is a Brave key in effect -- configured under either spelling, or in the environment?"""
    from opendde_harness.agent.tools.web import brave_key

    web = _load_raw_config().get("tools", {}).get("web", {})
    return bool(brave_key(web.get("braveApiKey") or web.get("brave_api_key")))


def _step4_web_search(
    *, brave_api_key: Optional[str], non_interactive: bool, warnings: list[str], skip_test: bool
) -> object:
    """Step 4 -- Web search (optional).

    Without a key ``web_search`` queries DuckDuckGo, which needs none, and a
    provider with a hosted search (the Codex login, OpenAI, Anthropic) uses
    that regardless. A Brave Search API key (free tier: 2,000 queries a
    month) makes the keyless path a documented API instead of a scraped page.
    The key comes from the flag when given, else from a prompt when there
    is a terminal; then one path verifies (unless ``skip_test``) and saves.
    """
    from opendde_harness.agent.tools.web import header_safe
    from opendde_harness.config.update import set_web_search_key

    _step_header(4, _t("Web search", "网页搜索"))
    console.print(
        _t(
            "  [dim]Without a key, web_search uses DuckDuckGo (no key needed); a provider with its own "
            "hosted search (Codex login, OpenAI, Anthropic) uses that either way.[/dim]\n"
            "  [dim]A Brave Search API key makes the keyless path a proper API: free tier 2,000 queries/month at "
            "[/dim][accent]https://api-dashboard.search.brave.com/app/keys[/accent]",
            "  [dim]不配置 key 时，web_search 走 DuckDuckGo（无需 key）；自带托管搜索的服务商（Codex 登录、OpenAI、"
            "Anthropic）始终用自己的搜索。[/dim]\n"
            "  [dim]配置 Brave Search API key 后，无托管搜索的模型改走正式 API：免费档每月 2,000 次，申请地址 "
            "[/dim][accent]https://api-dashboard.search.brave.com/app/keys[/accent]",
        )
    )
    key = brave_api_key
    if key is None and not non_interactive:
        questionary = _require_questionary()
        from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE

        key = questionary.password(
            _t("Brave Search API key (Enter to skip):", "Brave Search API key（直接回车跳过）:"),
            placeholder=_t("skip: DuckDuckGo, no key", "跳过：使用 DuckDuckGo，无需 key"),
            style=OPENDDE_HARNESS_STYLE,
            qmark=_QMARK,
        ).ask()
        if key is None:
            raise typer.Exit(1)
    key = (key or "").strip()
    if not key:
        if not non_interactive:
            console.print(
                _t(
                    "  [dim]Skipped: web_search keeps its Brave key.[/dim]"
                    if _web_search_configured()
                    else "  [dim]Skipped: web_search uses DuckDuckGo. Run `ddeharness onboard` again to add a key.[/dim]",
                    "  [dim]已跳过：web_search 沿用已配置的 Brave key。[/dim]"
                    if _web_search_configured()
                    else "  [dim]已跳过：web_search 使用 DuckDuckGo。随时重新运行 ddeharness onboard 添加 key。[/dim]",
                )
            )
        return None
    if not header_safe(key):
        # Refused before it is saved or sent: the HTTP stack's own rejection
        # would quote the whole key.
        console.print(
            _t(
                "  [warn]⚠ The key holds characters a header cannot carry (a line break?); not saved.[/warn]",
                "  [warn]⚠ key 中含有 HTTP 头无法携带的字符（换行？），未保存。[/warn]",
            )
        )
        warnings.append(_t("Web search (Brave)", "网页搜索（Brave）"))
        return None
    if not skip_test:
        ok, reason = _verify_brave_key(key)
        if not ok:
            console.print(
                _t(
                    f"  [warn]⚠ Brave did not accept the key: {reason}[/warn]",
                    f"  [warn]⚠ Brave 未接受该 key：{reason}[/warn]",
                )
            )
            warnings.append(_t("Web search (Brave)", "网页搜索（Brave）"))
    set_web_search_key(key)
    console.print(_t("  [ok]✓ Brave Search configured.[/ok]", "  [ok]✓ Brave Search 已配置。[/ok]"))
    return None


def _verify_brave_key(key: str) -> tuple[bool, str]:
    """One query against Brave's API; the reason on failure.

    ``_brief`` renders the failure: the reason is printed, and the raw text of
    a header the HTTP stack refused locally quotes the key itself.
    """
    import asyncio

    from opendde_harness.agent.tools.web import WebSearchTool, _brief

    try:
        hits = asyncio.run(WebSearchTool(api_key=key)._brave("OpenDDE", 1))
    except Exception as exc:
        return False, _brief(exc)
    return (True, "") if hits else (False, "no results")


def _run_wizard_body(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    skip_memory: bool = False,
    skip_protein_design: bool = False,
    brave_api_key: Optional[str] = None,
    non_interactive: bool = False,
    yes: bool = False,
    skip_test: bool = False,
    show_next_steps: bool = True,
    fresh: bool = False,
) -> None:
    global _LANG
    _check_tty_or_die(non_interactive)
    _LANG = _config_language()  # start from the saved language (default "en")
    if not non_interactive:
        _pick_language()  # may change _LANG (persisted after bootstrap below)
    if fresh or not _reusable_config():
        _retire_existing_setup()
    else:
        console.print()
        chrome.caption(
            console,
            _t(
                "Keeping your configuration; every step starts from what is already there.",
                "保留现有配置；每一步都以已有的值作为起点。",
            ),
        )
    _bootstrap_empty_config()
    if not non_interactive:
        from opendde_harness.config.update import set_language

        set_language(_LANG)  # persist now that config.json exists

    console.print()
    chrome.caption(console, _t("We'll configure, in order:", "我们将依次配置:"))
    chrome.steps(
        console,
        [
            _t("LLM provider", "LLM 服务商"),
            _t("Long-term memory", "长期记忆"),
            _t("Protein design", "蛋白设计"),
            _t("Web search", "网页搜索"),
        ],
    )
    console.print()
    chrome.hint(
        console,
        _t(
            "Ctrl+C quits at any point — the previous config is backed up and this run starts clean.",
            "随时 Ctrl+C 退出 — 原有配置已备份，本次从空白开始。",
        ),
    )

    warnings: list[str] = []

    # Screen state machine. Each screen returns ``_BACK`` to rewind or anything
    # else to advance. Step 1 is required and never rewinds: the language
    # screen before it is answered again by running the wizard again.
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
        lambda: _memory_step(skip=skip_memory, non_interactive=non_interactive, warnings=warnings),
        lambda: _step3_protein_design(
            skip=skip_protein_design,
            non_interactive=non_interactive,
            warnings=warnings,
        ),
        lambda: _step4_web_search(
            brave_api_key=brave_api_key,
            non_interactive=non_interactive,
            warnings=warnings,
            skip_test=skip_test,
        ),
    ]

    index = 0
    while index < len(screens):
        result = screens[index]()
        index += -1 if result is _BACK and index else 1

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
                f"  [warn]No usable provider resolves the default model ({model}).[/warn]",
                f"  [warn]默认模型({model})解析不到可用的服务商。[/warn]",
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
        provider: Optional[str] = typer.Option(
            None,
            "--provider",
            help="A pi provider id (e.g. 'openai-codex', 'anthropic'), or a name of your own for an "
            "OpenAI-compatible endpoint this config declares. Alone it goes straight to that provider, "
            "as /login <provider> does; with --api-key or --base-url it connects without prompts",
        ),
        api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the chosen provider"),
        base_url: Optional[str] = typer.Option(
            None,
            "--base-url",
            help="Address of an OpenAI-compatible endpoint this config declares (a relay, a self-hosted "
            "server); pi carries the address of every provider it ships",
        ),
        model: Optional[str] = typer.Option(
            None,
            "--model",
            help="Default model id (e.g. 'openai/gpt-4o-mini'); for a declared provider, the id(s) it serves",
        ),
        skip_memory: bool = typer.Option(False, "--skip-memory", help="Skip Step 2 (Persistent Memory)"),
        skip_protein_design: bool = typer.Option(
            False, "--skip-protein-design", help="Skip Step 3 (Protein Design compute settings)"
        ),
        brave_api_key: Optional[str] = typer.Option(
            None,
            "--brave-api-key",
            help="Step 4: Brave Search API key for web_search (without one, DuckDuckGo is used)",
        ),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            help="Run without prompts (requires flags for any missing field)",
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip all confirm prompts"),
        skip_test: bool = typer.Option(
            False,
            "--skip-test",
            help="Skip the one-shot test message (avoids a billed call; connectivity is still checked)",
        ),
        fresh: bool = typer.Option(
            False,
            "--fresh",
            help="Move the existing configuration aside and start from nothing, even when this build "
            "understands its shape",
        ),
    ) -> None:
        """Configure LLM, memory, Protein Design compute and web search."""
        run_wizard(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            skip_memory=skip_memory,
            skip_protein_design=skip_protein_design,
            brave_api_key=brave_api_key,
            non_interactive=non_interactive,
            yes=yes,
            skip_test=skip_test,
            fresh=fresh,
        )


__all__ = ["ensure_ready_to_start", "register", "run_wizard"]
