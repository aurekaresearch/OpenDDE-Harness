"""Protein Design deployment defaults for the onboarding wizard."""

import os
import secrets
from collections import Counter
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import typer

from opendde_harness.cli.compute_assets import opendde_root, weights_root
from opendde_harness.plugin.protein_design.core.asset_paths import DEFAULT_CHECKPOINT
from opendde_harness.plugin.protein_design.core.constants import DEFAULT_COMPUTE_URL

DEFAULT_OPENDDE_API_URL = "http://115.190.4.167:30080"


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
        return "cuda " + t("(nvidia-smi unavailable; device list not verified)", "（nvidia-smi 不可用，未验证设备列表）")
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

    local = ask(q.select(
        t("Where should Protein Design run?", "蛋白设计在哪里执行？"),
        choices=[
            q.Choice(t("Local Linux Docker environment", "本机 Linux Docker 环境"), value="local"),
            q.Choice(t("Existing Linux compute service", "已有的 Linux 计算服务"), value="remote"),
        ],
        default="local" if current.get("compute_docker") or not current.get("compute_url") else "remote",
        style=OPENDDE_HARNESS_STYLE,
    )) == "local"
    settings = None
    token = str(current.get("compute_token") or "")
    saved = current.get("compute_docker") or {}
    assets = prepared_defaults()
    try:
        if local:
            wizard.console.print(
                t(
                    "Compute service: bioinformatics tools running in a Docker container (SolubleMPNN/ProteinMPNN sequence design, "
                    "ESM2 scoring, OpenDDE folding, PLIP contact analysis, FoldMason structure alignment).",
                    "计算服务：运行在 Docker 容器中的生信工具服务（SolubleMPNN/ProteinMPNN 序列设计、ESM2 打分、OpenDDE 折叠、PLIP 接触分析、FoldMason 结构比对）。",
                )
            )
            idle_seconds = int(saved.get("idle_seconds") or local_service.DEFAULT_IDLE_SECONDS)
            minutes = max(1, idle_seconds // 60)
            wizard.console.print(
                t(
                    f"Compute container: starts on demand when a task needs it and is removed after {minutes} minutes idle; "
                    "ddeharness compute stop stops it now.",
                    f"计算容器：任务需要时按需启动，空闲 {minutes} 分钟后自动销毁；ddeharness compute stop 可立即停止。",
                )
            )
            info = compute.check_local_docker()
            # One published image for everyone; OPENDDE_HARNESS_COMPUTE_IMAGE is the
            # only override, and it is read inside default_image().
            image = compute.default_image()
            wizard.console.print(f"{t('Compute image:', '计算镜像：')} {escape(image)}")
            gpus = str(saved.get("gpus") or ("all" if "nvidia" in (info.get("Runtimes") or {}) else "none"))
            names = compute.gpu_inventory() if gpus != "none" else []
            wizard.console.print(f"{t('Compute device:', '计算设备：')} {escape(device_summary(gpus, names, t))}")
            if saved.get("port"):
                wizard.console.print(
                    f"{t('Compute service API port (compute_docker.port):', '计算服务 API 端口（compute_docker.port）：')} {saved['port']}"
                )
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
        wizard.console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    defaults = current.get("fold_defaults") or {}
    mode = ask(
        q.select(
            t("OpenDDE fold/refold mode:", "OpenDDE fold/refold 模式："),
            choices=["local", "api"],
            default=defaults.get("execution_mode", "api"),
            style=OPENDDE_HARNESS_STYLE,
        )
    )
    fold = {"execution_mode": mode}
    if mode == "api":
        fold["api_url"] = str(defaults.get("api_url") or DEFAULT_OPENDDE_API_URL).strip().rstrip("/")
        note = (
            t("official default service; override fold_defaults.api_url in config.json to use your own", "官方默认服务；如需自建服务，在 config.json 的 fold_defaults.api_url 中覆盖")
            if fold["api_url"] == DEFAULT_OPENDDE_API_URL
            else t("configured in config.json fold_defaults.api_url", "来自 config.json 的 fold_defaults.api_url")
        )
        wizard.console.print(f"{t('OpenDDE folding API:', 'OpenDDE 折叠 API：')} {escape(fold['api_url'])} ({note})")
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
            wizard.console.print(escape(weights_layout(
                root, data, with_opendde=mode == "local", checkpoint=Path(assets["opendde_checkpoint"]).name, translate=t,
            )))
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
            )
            if mode == "local":
                settings.opendde_data = str(data)
                settings.opendde_common = str(data / "common")
                checkpoint = Path(assets["opendde_checkpoint"]).expanduser().resolve()
                settings.opendde_checkpoint = str(
                    checkpoint if checkpoint.is_relative_to(data) else data / "checkpoint" / DEFAULT_CHECKPOINT
                )
            compute.validate_settings(settings, info, require_image=False, require_assets=False)
            wizard.console.print(
                t(
                    "The installed release prepares versioned tool code automatically. Code and models are mounted read-only; compatible environment images are reused.",
                    "已安装的发行包会自动准备带版本的工具代码。代码和模型只读挂载，兼容的环境镜像直接复用。",
                )
            )
            for label, value in settings.saved().items():
                if value:
                    wizard.console.print(f"  {label}: {escape(str(value))}")
        wizard.console.print(
            t(
                "New tasks inherit these defaults; explicit YAML settings take precedence. No design will be started.",
                "新任务继承这些默认值，YAML 显式设置优先。此步骤不会启动设计。",
            )
        )
        replace_pool = bool(current.get("compute_workers"))
        if replace_pool and not ask(
            q.confirm(
                t(
                    "Replace the existing worker pool for new tasks with this service?",
                    "将新任务的已有 worker pool 替换为此计算服务？",
                ),
                default=False,
                style=OPENDDE_HARNESS_STYLE,
            )
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
        if not ask(q.confirm(confirmation, default=True, style=OPENDDE_HARNESS_STYLE)):
            return
        wizard.console.print(
            t(
                "Preparing the selected environment, code and required weights; then checking service readiness and authentication…",
                "正在准备所选环境、代码和必要权重，随后检查计算服务就绪状态和认证…",
            )
        )
        if local:
            token = token or secrets.token_urlsafe(32)
            compute.prepare_assets(settings)
            pending = {
                **current, "compute_docker": settings.saved(), "compute_token": token,
                "fold_defaults": fold, "compute_workers": [],
            }
            if not local_service.stop_if_idle(token):
                wizard.console.print(
                    t(
                        "The running compute container is busy and keeps its current settings; the new settings apply at its next start.",
                        "运行中的计算容器正忙，保留其当前设置；新设置将在下次启动时生效。",
                    )
                )
            compute_url = local_service.ensure_compute_service(pending).url
        else:
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
        wizard.console.print(
            t(
                "Compute service connected. Protein Design settings saved.",
                "已连接计算服务，Protein Design 配置已保存。",
            )
        )
        for service in compute.probe_external_services(compute_url, token):
            wizard.console.print(escape(compute.external_service_line(service)))
    except compute.ComputeSetupError as exc:
        wizard.console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
