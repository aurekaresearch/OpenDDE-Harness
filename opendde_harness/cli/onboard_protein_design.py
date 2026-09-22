"""Protein Design deployment defaults for the onboarding wizard."""

import os
import secrets
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import typer

from opendde_harness.cli import _choice
from opendde_harness.cli import _chrome as chrome
from opendde_harness.cli.compute_assets import opendde_root, weights_root
from opendde_harness.plugin.protein_design.core.asset_paths import DEFAULT_CHECKPOINT
from opendde_harness.plugin.protein_design.core.constants import DEFAULT_COMPUTE_URL, DEFAULT_OPENDDE_API_URL


def deployment_paths(saved: dict[str, str], prepared: dict[str, str]) -> tuple[Path, dict[str, str]]:
    """Resolve the Harness weights root and the OpenDDE data paths without locating a source checkout."""
    data = opendde_root(saved)
    checkpoint = Path(saved.get("opendde_checkpoint") or DEFAULT_CHECKPOINT).name
    paths = {
        "opendde_data": str(data),
        "opendde_common": str(data / "common"),
        "opendde_checkpoint": str(data / "checkpoint" / checkpoint),
    }
    paths.update({key: value for key, value in prepared.items() if key in paths})
    return weights_root(saved), paths


def validate_service_url(value: str) -> bool | str:
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password:
            _ = parsed.port
            return True
    except ValueError:
        pass
    return "Use an HTTP(S) URL without embedded credentials."


def device_summary(gpus: str, names: list[str], t: Callable[[str, str], str]) -> str:
    """One line describing what the container will see: ``cpu`` or the GPU inventory."""
    if gpus == "none":
        return "cpu"
    if not names:
        return "cuda " + t(
            "(nvidia-smi unavailable; device list not verified)", "（nvidia-smi 不可用，未验证设备列表）"
        )
    if gpus == "all":
        scope_en, scope_zh = "all visible", "全部可见"
    else:
        wanted = {int(item) for item in gpus.split(",")}
        names = [name for index, name in enumerate(names) if index in wanted]
        scope_en, scope_zh = f"devices {gpus} visible", f"仅可见设备 {gpus}"
    inventory = ", ".join(f"{count} × {name}" for name, count in Counter(names).items())
    return inventory + t(
        f" ({scope_en}; tasks may pin a device per tool, automatic by default)",
        f"（{scope_zh}；任务可按工具指定用卡，默认自动分配）",
    )


def configure_protein_design() -> None:
    from rich.markup import escape

    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.cli import onboard_compute as compute
    from opendde_harness.cli._styles import OPENDDE_HARNESS_STYLE
    from opendde_harness.cli.compute_assets import prepared_defaults, weights_layout
    from opendde_harness.config.update import set_plugin_config_fields
    from opendde_harness.plugin.protein_design.servers import local_service

    q = wizard._require_questionary()
    t = wizard._t
    current = wizard._load_raw_config().get("plugins", {}).get("config", {}).get("protein-design", {})

    def ask(prompt):
        value = prompt.ask()
        if value is None:
            raise typer.Exit(1)
        return value

    @contextmanager
    def wizard_commentary():
        """What the compute layers say, said at the wizard's gutter.

        They are libraries that also run from a script and a container, so they
        write lines; here those lines sit beside everything else the step has
        drawn rather than flush against the edge of the terminal.
        """
        from opendde_harness.cli import _download

        previous = _download.report
        _download.report = lambda text: chrome.caption(wizard.console, escape(text))
        try:
            yield
        finally:
            _download.report = previous

    def ask_row(message, options, *, default=None):
        """A choice between a few things, on one line. Cancelling stops the run,
        as it does for every other prompt here."""
        value = _choice.row(message, options, default=default)
        if value is None:
            raise typer.Exit(1)
        return value

    local = (
        ask_row(
            t("Where should Protein Design run?", "蛋白设计在哪里执行？"),
            [
                (t("Local Linux Docker environment", "本机 Linux Docker 环境"), "local"),
                (t("Existing Linux compute service", "已有的 Linux 计算服务"), "remote"),
            ],
            default="local" if current.get("compute_docker") or not current.get("compute_url") else "remote",
        )
        == "local"
    )
    settings = None
    token = str(current.get("compute_token") or "")
    saved = current.get("compute_docker") or {}
    assets = prepared_defaults()
    try:
        if local:
            idle_seconds = int(saved.get("idle_seconds") or local_service.DEFAULT_IDLE_SECONDS)
            minutes = max(1, idle_seconds // 60)
            chrome.heading(wizard.console, t("Compute service", "计算服务"))
            chrome.caption(
                wizard.console,
                t(
                    "Sequence design, scoring, folding, contact analysis and structure alignment, in a Docker container.",
                    "序列设计、打分、折叠、接触分析与结构比对，运行在一个 Docker 容器里。",
                ),
            )
            chrome.caption(
                wizard.console,
                t(
                    f"Started when a task needs it and removed after {minutes} idle minutes; ddeharness compute stop stops it now.",
                    f"任务需要时启动，空闲 {minutes} 分钟后销毁；ddeharness compute stop 可立即停止。",
                ),
            )
            with chrome.working(wizard.console, t("Reading the Docker environment…", "正在读取 Docker 环境…")):
                info = compute.check_local_docker()
                # One published image for everyone; OPENDDE_HARNESS_COMPUTE_IMAGE
                # is the only override, and it is read inside default_image().
                image = compute.default_image()
                gpus = str(saved.get("gpus") or ("all" if compute.docker_gpu_available(info) else "none"))
                names = compute.gpu_inventory() if gpus != "none" else []
            wizard.console.print()
            facts = [
                (t("Image", "镜像"), escape(image)),
                (t("Device", "计算设备"), escape(device_summary(gpus, names, t))),
            ]
            if saved.get("port"):
                facts.append((t("API port", "API 端口"), str(saved["port"])))
            chrome.fields(wizard.console, facts)
        else:
            compute_url = (
                ask(
                    q.text(
                        t("Harness Docker compute URL:", "Harness Docker 计算服务地址："),
                        default=current.get("compute_url") or DEFAULT_COMPUTE_URL,
                        validate=validate_service_url,
                        style=OPENDDE_HARNESS_STYLE,
                    )
                )
                .strip()
                .rstrip("/")
            )
            token = (
                ask(
                    q.password(
                        t("Compute token (blank keeps existing):", "计算服务 token（留空保留已有值）："),
                        style=OPENDDE_HARNESS_STYLE,
                    )
                ).strip()
                or token
            )
    except compute.ComputeSetupError as exc:
        wizard.console.print(f"[error]{escape(str(exc))}[/error]")
        raise typer.Exit(1) from exc
    defaults = current.get("fold_defaults") or {}
    chrome.heading(wizard.console, t("Folding", "折叠"))
    wizard.console.print()
    mode = ask_row(
        t("OpenDDE fold/refold mode:", "OpenDDE fold/refold 模式："),
        [
            (t("local — in the compute container", "local — 在计算容器里"), "local"),
            (t("api — through the OpenDDE service", "api — 走 OpenDDE 服务"), "api"),
        ],
        # Local is the mode this runs in: the compute container carries the
        # folding weights, so nothing leaves the machine and no service has to
        # answer. A config that already names one keeps it.
        default=defaults.get("execution_mode", "local"),
    )
    fold = {"execution_mode": mode}
    design_modes = []
    if mode == "local" and local:
        selected_mode = ask_row(
            t("Prepare checkpoints for:", "准备哪些设计模式的权重？"),
            [(t("Antibody", "抗体"), "antibody"), ("Minibinder", "minibinder"), (t("Both", "两者"), "both")],
            default="both"
            if len(saved.get("design_modes", [])) == 2
            else (saved.get("design_modes") or ["antibody"])[0],
        )
        design_modes = ["antibody", "minibinder"] if selected_mode == "both" else [selected_mode]
    if mode == "api":
        fold["api_url"] = str(defaults.get("api_url") or DEFAULT_OPENDDE_API_URL).strip().rstrip("/")
        note = (
            t(
                "official default service; override fold_defaults.api_url in config.json to use your own",
                "官方默认服务；如需自建服务，在 config.json 的 fold_defaults.api_url 中覆盖",
            )
            if fold["api_url"] == DEFAULT_OPENDDE_API_URL
            else t("configured in config.json fold_defaults.api_url", "来自 config.json 的 fold_defaults.api_url")
        )
        chrome.fields(wizard.console, [(t("Folding API", "折叠 API"), escape(fold["api_url"]))])
        chrome.caption(wizard.console, note)
    try:
        if local:
            root, assets = deployment_paths(saved, assets)
            package_root = os.environ.get("OPENDDE_HARNESS_COMPUTE_SOURCE_DIR", "").strip()
            if not package_root and saved.get("code_mode") == "checkout":
                package_root = saved.get("package_root", "")
            legacy = {
                "opendde_data": root / "external/opendde/cache/release_data",
                "opendde_common": root / "external/opendde/cache/common",
                "opendde_checkpoint": root / "external/opendde/weights/model.pt",
            }
            for key in assets:
                value = saved.get(key)
                if value and (key not in legacy or Path(value).expanduser().absolute() != legacy[key]):
                    assets[key] = str(Path(value).expanduser().absolute())
            data = Path(assets["opendde_data"]).expanduser().resolve()
            if design_modes:
                from opendde_harness.plugin.protein_design.core.asset_paths import DESIGN_CHECKPOINTS, checkpoint_status

                selected_path = Path(assets["opendde_checkpoint"])
                if selected_path.name in DESIGN_CHECKPOINTS.values():
                    assets["opendde_checkpoint"] = str(data / "checkpoint" / DESIGN_CHECKPOINTS[design_modes[0]])
                for kind, filename in DESIGN_CHECKPOINTS.items():
                    status = checkpoint_status(data / "checkpoint" / filename, kind)
                    chrome.caption(
                        wizard.console,
                        f"{kind}: {filename} — "
                        + (t("ready", "就绪") if status["ready"] else str(status["error"]))
                        + ("" if kind in design_modes else t(" (optional)", "（可选）")),
                    )
            chrome.heading(wizard.console, t("Weights and code", "权重与代码"))
            for line in weights_layout(
                root,
                data,
                with_opendde=mode == "local",
                checkpoint=Path(assets["opendde_checkpoint"]).name,
                translate=t,
            ).splitlines():
                chrome.caption(wizard.console, escape(line))
            output_dir = (
                Path(
                    os.environ.get(
                        "OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT", str(Path.home() / ".opendde_harness/protein_design")
                    )
                )
                .expanduser()
                .resolve()
            )
            settings = compute.DockerSettings(
                image=image,
                mode=mode,
                gpus=gpus,
                package_root=package_root,
                state_dir=str(output_dir.parent),
                output_dir=str(output_dir),
                weights_dir=str(root),
                code_mode="checkout" if package_root else "managed",
                port=int(saved.get("port") or 0),
                idle_seconds=idle_seconds,
                design_modes=design_modes,
            )
            if mode == "local":
                settings.opendde_data = str(data)
                settings.opendde_common = str(data / "common")
                checkpoint = Path(assets["opendde_checkpoint"]).expanduser().resolve()
                settings.opendde_checkpoint = str(
                    checkpoint if checkpoint.is_relative_to(data) else data / "checkpoint" / DEFAULT_CHECKPOINT
                )
            compute.validate_settings(settings, info, require_image=False, require_assets=False)
            chrome.caption(
                wizard.console,
                t(
                    "Versioned tool code is prepared for you; code and models are mounted read-only and a compatible image is reused.",
                    "工具代码按版本自动准备；代码与模型只读挂载，兼容的环境镜像直接复用。",
                ),
            )
            wizard.console.print()
            # The operator's own facts, named. The rest of `settings.saved()` is
            # this program's bookkeeping -- a dump of its keys read as a config
            # file that had leaked onto the screen.
            chrome.fields(
                wizard.console,
                [
                    (t("Weights", "权重目录"), escape(settings.weights_dir)),
                    (t("Output", "输出目录"), escape(settings.output_dir)),
                    (t("Idle timeout", "空闲销毁"), t(f"{minutes} min", f"{minutes} 分钟")),
                ],
            )
        wizard.console.print()
        chrome.caption(
            wizard.console,
            t(
                "New tasks inherit these defaults and explicit YAML settings win. No design is started here.",
                "新任务继承这些默认值，YAML 显式设置优先；此步不会启动设计。",
            ),
        )
        wizard.console.print()
        replace_pool = bool(current.get("compute_workers"))
        if replace_pool and not ask_row(
            t(
                "Replace the existing worker pool for new tasks with this service?",
                "将新任务的已有 worker pool 替换为此计算服务？",
            ),
            [(t("keep it", "保留"), False), (t("replace it", "替换"), True)],
            default=False,
        ):
            wizard.console.print(
                t(
                    "Existing worker pool and connection settings kept. No container was started.",
                    "已保留原有 worker pool 和连接配置，未启动容器。",
                )
            )
            return
        confirmation = (
            t("Start/connect and save Protein Design settings?", "启动/连接并保存 Protein Design 配置？")
            if local
            else t("Save Protein Design settings?", "保存 Protein Design 配置？")
        )
        if not ask_row(confirmation, [(t("yes", "好"), True), (t("no", "不用"), False)], default=True):
            return
        if local:
            token = token or secrets.token_urlsafe(32)
            # No status line around this one: a transfer draws its own progress,
            # and two live regions on one console cannot both draw.
            chrome.caption(
                wizard.console,
                t("Preparing the environment, code and weights…", "正在准备环境、代码与权重…"),
            )
            with wizard_commentary():
                compute.prepare_assets(settings)
            pending = {
                **current,
                "compute_docker": settings.saved(),
                "compute_token": token,
                "fold_defaults": fold,
                "compute_workers": [],
            }
            if not local_service.stop_if_idle(token):
                chrome.caption(
                    wizard.console,
                    t(
                        "The running container is busy and keeps its settings; the new ones apply at its next start.",
                        "运行中的容器正忙，保留其当前设置；新设置将在下次启动时生效。",
                    ),
                )
            with (
                wizard_commentary(),
                chrome.working(
                    wizard.console,
                    t("Starting the compute service…", "正在启动计算服务…"),
                    note=t(
                        "the container loads the image and answers a readiness check; up to 90s",
                        "容器载入镜像并通过就绪检查，最多 90 秒",
                    ),
                ),
            ):
                compute_url = local_service.ensure_compute_service(pending).url
        else:
            with chrome.working(wizard.console, t("Checking the compute service…", "正在检查计算服务…")):
                compute.wait_for_service(compute_url, token, mode, fold.get("api_url", ""))
        fields = {
            "compute_url": compute_url,
            "compute_token": token,
            "fold_defaults": fold,
            "compute_docker": settings.saved() if local else {},
        }
        if replace_pool:
            fields["compute_workers"] = []
        set_plugin_config_fields("protein-design", fields)
        chrome.done(
            wizard.console,
            t("Compute service connected, settings saved.", "已连接计算服务，配置已保存。"),
        )
        with chrome.working(wizard.console, t("Checking the tool services…", "正在检查工具服务…")):
            external = list(compute.probe_external_services(compute_url, token))
        for service in external:
            chrome.caption(wizard.console, escape(compute.external_service_line(service)))
    except compute.ComputeSetupError as exc:
        wizard.console.print(f"[error]{escape(str(exc))}[/error]")
        raise typer.Exit(1) from exc
