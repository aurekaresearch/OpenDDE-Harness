"""``ddeharness doctor`` — health check (static + optional --probe).

The static checks are zero-network and millisecond-fast. Memory and, when
Protein Design is configured, the compute service are probed over HTTP;
without a Protein Design configuration doctor touches neither Docker nor the
network. ``--probe`` sends one chat exchange via
:func:`opendde_harness.cli._helpers.send_probe`.

Exit codes:
  0  — all green (and probe ok if requested)
  1  — static check failed (config missing / schema invalid / unresolved routing)
  2  — static checks ok but ``--probe``, memory or compute readiness failed
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console

from opendde_harness import __logo__
from opendde_harness.cli._helpers import print_probe_troubleshooting, send_probe

if TYPE_CHECKING:
    from opendde_harness.config.opendde_harness import OpenDDEHarnessConfig

console = Console()


@dataclass
class PathsInfo:
    config_path: str
    config_exists: bool
    config_valid: bool = False
    config_invalid_reason: str = ""
    workspace_path: str = ""
    workspace_exists: bool = False


@dataclass
class RoutingInfo:
    model: str
    provider: Optional[str]
    max_tokens: int
    context_window_tokens: Optional[int]


@dataclass
class FeaturesInfo:
    skill_forge_enabled: bool = False


@dataclass
class MemoryInfo:
    """What the memory backend is, and what it can actually do.

    ``configured`` comes from the config files; ``capabilities`` from a running
    server's ``/health``. Keeping both is the point: from library 1.2.1 a server
    whose embedding provider failed to build still answers 200 and degrades to
    keyword-only search, so the two can disagree, and that disagreement is the
    fault worth reporting.
    """

    backend: Optional[str] = None
    root: Optional[str] = None
    owned: bool = True
    address: Optional[str] = None
    server_running: bool = False
    reports_capabilities: bool = False
    configured: list[str] = field(default_factory=list)
    capabilities: dict[str, bool] = field(default_factory=dict)
    retrieval: Optional[str] = None
    #: Set when memory exists on disk but the runtime will not use it.
    disabled_reason: Optional[str] = None

    @property
    def unbuilt(self) -> list[str]:
        """Roles the user configured that the server could not build."""
        from opendde_harness.plugin.memory.longterm._health import capability_available

        return [s for s in self.configured if capability_available(self.capabilities, s) is False]

    @property
    def broken(self) -> list[str]:
        """Unbuilt roles that memory cannot work without at all.

        Separate from :attr:`unbuilt` because the others cost quality, not
        function: without embedding the adapter searches lexically instead of
        semantically, and that is a worse memory rather than no memory. Only this
        list decides the exit code.
        """
        from opendde_harness.plugin.memory.longterm._health import REQUIRED_SECTIONS

        return [s for s in self.unbuilt if s in REQUIRED_SECTIONS]


@dataclass
class ProbeResult:
    ok: bool
    text: Optional[str] = None
    tokens: Optional[int] = None
    elapsed_s: Optional[float] = None
    error: Optional[str] = None


@dataclass
class DoctorReport:
    version: int = 1
    config_loaded: bool = False
    config_error: Optional[str] = None
    paths: Optional[PathsInfo] = None
    routing: Optional[RoutingInfo] = None
    features: Optional[FeaturesInfo] = None
    memory: Optional[MemoryInfo] = None
    probe: Optional[ProbeResult] = None
    compute: Optional[dict] = None

    def exit_code(self) -> int:
        if self.paths is None or not self.paths.config_exists:
            return 1
        if not self.paths.config_valid:
            return 1
        if not self.config_loaded:
            return 1
        if self.routing is None or self.routing.provider is None:
            return 1
        if self.probe is not None and not self.probe.ok:
            return 2
        # A role the user configured that the server could not build is a real
        # fault, not a warning: recall silently returns nothing.
        if self.memory is not None and self.memory.broken:
            return 2
        if self.compute is not None and not self.compute.get("ready"):
            return 2
        return 0


def _gather_static_checks() -> DoctorReport:
    """Inspect config / routing / features. Strictly zero-network."""
    from opendde_harness.config.loader import ConfigSchemaError, get_config_path, load_config

    config_path = get_config_path()
    paths = PathsInfo(
        config_path=str(config_path),
        config_exists=config_path.exists(),
    )
    report = DoctorReport(paths=paths)

    if not paths.config_exists:
        return report

    # Classify config validity with load_config's eyes: a syntax error, an
    # empty file, and a non-object top level all mean no settings were read.
    # Inspect the file directly -- load_config swallows syntax errors into
    # defaults, and read_raw_or_raise folds the last two cases into {} for
    # its read-modify-write callers, so neither can classify all three.
    try:
        text = config_path.read_text(encoding="utf-8")
        data = json.loads(text) if text.strip() else None
    except (OSError, UnicodeDecodeError, ValueError):
        paths.config_invalid_reason = "invalid JSON"
    else:
        if not text.strip():
            paths.config_invalid_reason = "empty"
        elif not isinstance(data, dict):
            paths.config_invalid_reason = "not a JSON object"
    paths.config_valid = not paths.config_invalid_reason

    try:
        config = load_config()
    except ConfigSchemaError as exc:
        report.config_error = str(exc)
        return report
    except Exception:
        return report
    report.config_loaded = True

    workspace = config.workspace_path
    paths.workspace_path = str(workspace)
    paths.workspace_exists = workspace.exists()

    from opendde_harness.providers.rates import resolve_max_output_tokens

    defaults = config.agents.defaults
    report.routing = RoutingInfo(
        model=defaults.model,
        provider=config.get_provider_name(),
        # What a request will actually carry, resolved the same way the
        # provider resolves it -- doctor reporting a configured number that
        # no longer exists would be reporting a setting, not the behaviour.
        max_tokens=resolve_max_output_tokens(defaults.model),
        context_window_tokens=defaults.context_window_tokens,
    )

    try:
        skill_forge_on = bool(config.skill_forge.enabled)
    except Exception:
        skill_forge_on = False
    report.features = FeaturesInfo(skill_forge_enabled=skill_forge_on)
    return report


def _probe_memory(config: "OpenDDEHarnessConfig") -> MemoryInfo:
    """Ask the memory server what it can do. Local HTTP only, never raises.

    Deliberately not part of ``_gather_static_checks``: that stays zero-network.
    This one talks to localhost, which is cheap enough to run unconditionally --
    unlike ``--probe``, it spends no tokens and reaches no third party.
    """
    backend = config.memory.backend
    info = MemoryInfo(backend=backend)
    from opendde_harness.config.update_memory import memory_owned, memory_role_configured, memory_root
    from opendde_harness.plugin.memory.longterm._health import BACKEND_NAME

    if backend != BACKEND_NAME:
        # Memory can sit on disk and still be off: the wizard records a managed
        # root and starts a service, then leaves the backend unset while a
        # required role has no credentials. Recall answers zero hits forever
        # and every other surface looks healthy, so this is the one place that
        # can say why.
        if backend is None and memory_owned() and memory_root().is_dir():
            from opendde_harness.plugin.memory.longterm._health import REQUIRED_SECTIONS

            missing = [s for s in REQUIRED_SECTIONS if not memory_role_configured(s)]
            info.root = str(memory_root())
            info.disabled_reason = f"no credentials for {', '.join(missing)}" if missing else "turned off in config"
        return info
    from opendde_harness.plugin.memory.longterm._health import (
        DEGRADING_SECTIONS,
        REQUIRED_SECTIONS,
        configured_base_url,
        probe_capabilities,
    )

    # Which memories, and whose. Neither was reachable from any command before:
    # the wizard printed the path once while converging and nothing showed it
    # again, so "where are my memories" had no answer short of reading
    # config.json by hand. This is the place that question gets asked.
    info.owned = memory_owned()
    info.address = configured_base_url(config)
    report = probe_capabilities(configured_base_url(config))
    info.server_running = report.reachable
    info.reports_capabilities = report.reports_capabilities
    info.capabilities = dict(report.capabilities)

    if info.owned:
        info.root = str(memory_root())
        info.configured = [s for s in (*REQUIRED_SECTIONS, *DEGRADING_SECTIONS) if memory_role_configured(s)]
        # Recall quality is decided by the embedding role in the user-level
        # memory config: with it recall matches meaning, without it only keywords.
        info.retrieval = "semantic" if "embedding" in info.configured else "keyword-only"
        return info

    # A root the user runs. Nothing here may come from the local filesystem:
    # no root is recorded for it, so ``memory_root()`` would answer with the
    # fallback -- a directory that is not theirs and holds none of their
    # memories -- and the roles read out of that directory's toml would
    # describe an install nobody is using. Reading their toml is not an option
    # either; not touching it is the promise. What the server says about itself
    # is the only honest source, and when it is down there is no source at all.
    info.root = None
    # Every section the server has an opinion about -- built or failed. Taking
    # only the built ones made ``unbuilt`` (the failed subset of this list)
    # structurally empty, so ``broken`` and the exit code could never fire and
    # a server that could not build its LLM reported healthy. OpenDDE Harness cannot read
    # their toml to learn what they configured, and does not need to: a section
    # the server reports as unavailable is one it tried to build and could not.
    info.configured = [s for s in (*REQUIRED_SECTIONS, *DEGRADING_SECTIONS) if report.available(s) is not None]
    info.retrieval = None
    if report.reports_capabilities:
        info.retrieval = "semantic" if report.available("embedding") is True else "keyword-only"
    return info


def _run_llm_probe(timeout_s: int) -> ProbeResult:
    """Wrap :func:`send_probe` so failures become a structured ProbeResult."""
    try:
        text, tokens, elapsed = send_probe(timeout_s=timeout_s)
        return ProbeResult(ok=True, text=text, tokens=tokens, elapsed_s=elapsed)
    except Exception as exc:
        return ProbeResult(ok=False, error=str(exc) or exc.__class__.__name__)


def _render_memory_capabilities(memory: MemoryInfo) -> None:
    """Report the running server's capabilities, or say why they are unknown.

    "Server running" and "server can recall" stopped being the same statement in
    library 1.2.1, so they are printed as separate lines rather than one tick.
    """
    from opendde_harness.plugin.memory.longterm._health import BACKEND_NAME, capability_available

    if memory.disabled_reason:
        console.print("\n[bold]Memory[/bold]")
        console.print(f"  Memories:   {memory.root}")
        console.print(f"  [yellow]Disabled:   {memory.disabled_reason}; recall returns nothing.[/yellow]")
        console.print("  [dim]Run ddeharness onboard to finish memory setup.[/dim]")
        return
    if memory.backend != BACKEND_NAME:
        return
    if memory.root:
        console.print(f"  Memories:   {memory.root}")
    if not memory.owned:
        console.print(
            "  [dim]Managed by you -- OpenDDE Harness reads it at the address below and never writes,\n"
            "  starts or stops it, so it does not track where on disk it keeps them.[/dim]"
        )
    console.print(f"  Address:    {memory.address}")
    if not memory.server_running:
        console.print("  Server:     [dim]not running  (starts on demand)[/dim]")
        if memory.configured:
            console.print(f"  Configured: {', '.join(memory.configured)}")
        return
    console.print("  Server:     [green]running[/green]")
    if not memory.reports_capabilities:
        console.print("  [dim]This server does not report capabilities (library < 1.2.1).[/dim]")
        if memory.configured:
            console.print(f"  Configured: {', '.join(memory.configured)}")
        return
    from opendde_harness.plugin.memory.longterm._health import DEGRADING_SECTIONS, REQUIRED_SECTIONS

    for section in (*REQUIRED_SECTIONS, *DEGRADING_SECTIONS):
        label = f"  {section + ':':<12}"
        if section not in memory.configured:
            console.print(f"{label}[dim]not configured{_degradation_note(section)}[/dim]")
            continue
        state = capability_available(memory.capabilities, section)
        if state is True:
            console.print(f"{label}[green]✓[/green]")
        elif state is False:
            console.print(f"{label}[red]✗ configured, but the server could not build it[/red]")
        else:
            console.print(f"{label}[dim]not reported[/dim]")
    if memory.unbuilt:
        console.print()
        if memory.broken:
            console.print(
                f"  [yellow]⚠ Memory needs {' and '.join(memory.broken)} and cannot work until this is fixed.[/yellow]"
            )
        else:
            # "memory", not "recall": an unbuilt multimodal llm costs ingest of
            # images / PDFs / audio, which recall never sees either way.
            console.print(
                f"  [yellow]⚠ {' and '.join(memory.unbuilt)} is configured but unavailable, so memory "
                "runs degraded.[/yellow]"
            )
        console.print(f"  [dim]Check the server log: {_server_log_hint()}[/dim]")


def _degradation_note(section: str) -> str:
    """What is lost by leaving an optional role unconfigured.

    Stated per role rather than as one blanket "optional": they degrade
    differently, and a user deciding whether to configure embedding needs to know
    it costs semantic recall specifically.
    """
    return {
        "embedding": "  (recall matches keywords, not meaning)",
        "rerank": "  (agent-track recall uses the LLM lane instead of a cross-encoder)",
        "multimodal": "  (images, PDFs and audio stay out of memory)",
    }.get(section, "")


def _server_log_hint() -> str:
    from opendde_harness.plugin.memory.longterm._server import server_log_path

    return str(server_log_path())


def _render_human_output(report: DoctorReport) -> None:
    console.print(f"\n{__logo__} OpenDDE Harness Doctor\n")

    paths = report.paths
    assert paths is not None  # _gather_static_checks always populates this
    console.print("[bold]Paths[/bold]")
    if not paths.config_exists:
        console.print(f"  Config:    {paths.config_path}  [red]✗  (not found)[/red]")
    elif not paths.config_valid:
        reason = paths.config_invalid_reason or "invalid JSON"
        console.print(f"  Config:    {paths.config_path}  [yellow]⚠  {reason} (running on defaults)[/yellow]")
    else:
        console.print(f"  Config:    {paths.config_path}  [green]✓[/green]")
    if paths.config_exists:
        mark = "[green]✓[/green]" if paths.workspace_exists else "[red]✗[/red]"
        console.print(f"  Workspace: {paths.workspace_path}  {mark}")

    if not paths.config_exists:
        console.print(
            "\n[yellow]⚠ OpenDDE Harness is not configured.[/yellow] Run [cyan]ddeharness onboard[/cyan] to set it up."
        )
        return

    if not report.config_loaded:
        if paths.config_valid:
            console.print(
                "\n[red]✗ Config schema invalid.[/red] Run [cyan]ddeharness onboard --reset[/cyan] to recreate it."
            )
            # The unknown-key line, without pydantic's full listing.
            for line in (report.config_error or "").splitlines():
                if line.startswith("unknown key(s):"):
                    console.print(f"  {line}")
        else:
            reason = paths.config_invalid_reason or "invalid JSON"
            console.print(f"\n[yellow]⚠ Config file is {reason}; the checks above ran on built-in defaults.[/yellow]")
            console.print(f"Fix [cyan]{paths.config_path}[/cyan] or run [cyan]ddeharness onboard --reset[/cyan].")
        return

    routing = report.routing
    if routing is not None:
        console.print("\n[bold]Routing[/bold]")
        console.print(f"  Model:        {routing.model}")
        if routing.provider:
            console.print(f"  Routes to:    {routing.provider}")
        else:
            console.print("  Routes to:    [red]<unresolved>[/red]")
        console.print(f"  Max tokens:   {routing.max_tokens}")
        console.print(f"  Context win:  {routing.context_window_tokens if routing.context_window_tokens else 'auto'}")

    features = report.features
    if features is not None:
        console.print("\n[bold]Features[/bold]")
        sf_label = "enabled" if features.skill_forge_enabled else "[dim]disabled[/dim]"
        console.print(f"  Skill forge: {sf_label}")

    memory = report.memory
    if memory is not None and memory.disabled_reason:
        _render_memory_capabilities(memory)
    elif memory is not None and memory.backend:
        console.print("\n[bold]Memory[/bold]")
        console.print(f"  Backend:    {memory.backend}")
        if memory.retrieval == "semantic":
            console.print("  Retrieval:  semantic")
        elif memory.retrieval:
            console.print("  Retrieval:  [dim]keyword-only  (no embedding key)[/dim]")
        elif not memory.owned:
            console.print("  Retrieval:  [dim]unknown  (the server you run is not answering)[/dim]")
        _render_memory_capabilities(memory)

    if report.probe is not None:
        console.print("\n[bold]LLM Probe[/bold]")
        if routing:
            console.print(f"  → {routing.model}")
        if report.probe.ok:
            console.print(f'  [green]✓ Response:[/green] "{report.probe.text}"')
            extras: list[str] = []
            if report.probe.tokens:
                extras.append(f"{report.probe.tokens} tokens")
            if report.probe.elapsed_s is not None:
                extras.append(f"{report.probe.elapsed_s:.1f}s")
            if extras:
                console.print(f"  [green]✓ {', '.join(extras)}[/green]")
        else:
            console.print(f"  [red]✗ Failed:[/red] {report.probe.error}")
            print_probe_troubleshooting(routing.provider if routing else None)

    if report.compute is not None:
        _render_compute(report.compute)

    console.print()
    code = report.exit_code()
    if code == 0:
        if report.probe is None:
            console.print("[green]✓ Configuration looks healthy.[/green]")
            console.print("Run [cyan]doctor --probe[/cyan] to send a test message and verify the LLM responds.")
        else:
            console.print("[green]✓ All checks passed.[/green]")
    elif not paths.config_valid:
        reason = paths.config_invalid_reason or "invalid JSON"
        console.print(f"[yellow]⚠ Config file is {reason}; the checks above ran on built-in defaults.[/yellow]")
        console.print(f"Fix [cyan]{paths.config_path}[/cyan] (JSON allows no comments or trailing commas).")
    elif routing and routing.provider is None:
        console.print(
            f"[red]✗ Model [bold]{routing.model}[/bold] could not be routed to any configured provider.[/red]"
        )
        console.print(
            "Run [cyan]ddeharness provider list[/cyan] / [cyan]ddeharness provider set[/cyan] to fix routing."
        )


def _render_compute(report: dict, indent: str = "  ") -> None:
    from rich.markup import escape

    if indent == "  ":
        console.print("\n[bold]Protein Design compute[/bold]")
    console.print(
        f"{indent}Placement: {report.get('placement', 'assets')}  OpenDDE: {report.get('fold_mode', 'api')}  "
        f"Device: {report.get('device') or 'not verified'}"
    )
    for check in report.get("checks", []):
        icon = "[green]OK[/green]" if check["ok"] else "[red]FAIL[/red]"
        console.print(
            f"{indent}{icon} {escape(check['name'])}" + (f": {escape(check['error'])}" if check.get("error") else "")
        )
    for service in report.get("external_services") or []:
        from opendde_harness.cli.onboard_compute import external_service_line

        console.print(f"{indent}{escape(external_service_line(service))}")
    if report.get("service") is not None:
        _render_service(report["service"], indent)
    assets = report.get("assets") or {}
    if assets.get("root"):
        console.print(f"{indent}Harness tool weights: {escape(assets['root'])}")
    if assets.get("opendde_root"):
        console.print(f"{indent}OpenDDE data: {escape(assets['opendde_root'])}")
    files = assets.get("files") or []
    for item in files:
        if not item["ok"]:
            console.print(f"{indent}[red]FAIL[/red] {escape(item['path'])}: {escape(item['error'])}")
    if files:
        verification = (
            "SHA256 verified"
            if assets.get("hashes_verified")
            else "presence/size checked; use --verify-hashes for content verification"
        )
        console.print(f"{indent}Assets: {sum(item['ok'] for item in files)}/{len(files)} ({verification})")
    if assets.get("opendde") == "not_required":
        console.print(f"{indent}OpenDDE weights/common data: not required in API mode")
    for worker in report.get("workers", []):
        console.print(f"{indent}Worker: {escape(worker['id'])}")
        _render_compute(worker, indent + "  ")
    if indent == "  " and not report.get("ready"):
        console.print(
            "  Run [cyan]ddeharness onboard[/cyan] to configure or start the service; "
            "[cyan]ddeharness compute prepare[/cyan] downloads missing weights and runtime code."
        )


def _render_service(service: dict, indent: str) -> None:
    """The on-demand container: not running is normal, running shows its queue, idle countdown and GPU leases."""
    from rich.markup import escape

    container = escape(str(service.get("container") or ""))
    if not service.get("running"):
        console.print(f"{indent}Service: [dim]not running  (starts on demand as {container})[/dim]")
        return
    release = "" if service.get("current_release", True) else "  [yellow](previous release; exits when idle)[/yellow]"
    code_id = escape(str(service.get("code_id") or "")[:12])
    console.print(
        f"{indent}Service: [green]running[/green]  {container}  port {service.get('port')}  code {code_id}{release}"
    )
    if not service.get("healthy"):
        return
    idle, limit = service.get("idle_seconds"), service.get("idle_timeout_seconds")
    countdown = (
        f"idle {int(idle)}s of {int(limit)}s" if idle is not None and limit is not None else "idle timeout not reported"
    )
    console.print(
        f"{indent}Jobs: {service.get('jobs_running') or 0} running, {service.get('jobs_queued') or 0} queued  ({countdown})"
    )
    for lease in service.get("gpu_leases") or []:
        detail = ", ".join(f"{key}={value}" for key, value in lease.items()) if isinstance(lease, dict) else str(lease)
        console.print(f"{indent}GPU lease: {escape(detail)}")
    for lease in service.get("task_leases") or []:
        detail = ", ".join(f"{key}={value}" for key, value in lease.items()) if isinstance(lease, dict) else str(lease)
        console.print(f"{indent}Task lease: {escape(detail)}")


def register(app: typer.Typer) -> None:
    @app.command()
    def doctor(
        probe: bool = typer.Option(False, "--probe", help="Send a test message to verify the LLM responds."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON (CI-friendly)."),
        timeout: int = typer.Option(
            15,
            "--timeout",
            help="LLM probe timeout in seconds.",
            min=1,
        ),
        compute_only: bool = typer.Option(
            False, "--compute-only", help="Check Protein Design compute only; skip LLM and memory checks."
        ),
        verify_hashes: bool = typer.Option(
            False, "--verify-hashes", help="Read every required weight and verify its SHA256."
        ),
    ) -> None:
        """Check configuration, and the Protein Design compute setup when one is configured."""
        from opendde_harness.cli.onboard_compute import inspect_compute, load_protein_design_config

        config = load_protein_design_config()
        if compute_only:
            if not config:
                console.print(
                    "[red]✗ Protein Design is not configured.[/red] Run [cyan]ddeharness onboard[/cyan] to set it up."
                )
                raise typer.Exit(1)
            compute = inspect_compute(config, verify_hashes=verify_hashes)
            if json_output:
                console.print_json(json.dumps(compute))
            else:
                _render_compute(compute)
            raise typer.Exit(0 if compute["ready"] else 2)

        report = _gather_static_checks()
        if config:
            report.compute = inspect_compute(config, verify_hashes=verify_hashes)

        if report.config_loaded:
            from opendde_harness.config.opendde_harness import load_opendde_harness_config

            report.memory = _probe_memory(load_opendde_harness_config())

        if probe and report.routing is not None and report.routing.provider is not None:
            report.probe = _run_llm_probe(timeout_s=timeout)

        if json_output:
            console.print_json(json.dumps(asdict(report)))
        else:
            _render_human_output(report)

        raise typer.Exit(report.exit_code())


__all__ = ["register"]
