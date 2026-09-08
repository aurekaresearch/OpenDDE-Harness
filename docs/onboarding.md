# Onboarding

The first `ddeharness` run starts the onboarding wizard in the terminal before the TUI opens when no provider is configured; without a terminal it exits and asks you to run `ddeharness onboard`. Run `ddeharness onboard` at any time to change settings. There is no wizard inside the TUI: `/onboard` and `/setup` only print a hint to exit and run the terminal command.

The three steps are **LLM**, **long-term memory**, and **Protein Design**. Memory is optional; skip it with `--skip-memory` and skip compute setup with `--skip-protein-design`.

Onboarding does not launch a design task or install Docker/GPU drivers.

The first long-term memory setup menu includes **Skip for now**. Design tasks
still run without it; losing cross-run experience may slightly affect results.
Configure memory when convenient to support long-term learning, or return later
with `ddeharness onboard`.

## Local Docker compute

Unless an external compute service is already configured, Protein Design setup manages a local compute container. The container is ephemeral: it starts on demand when a task needs it and removes itself after an idle timeout (see [container lifecycle](#container-lifecycle)).

1. Choose local Linux Docker or an existing Linux compute service. Local setup checks the platform and Docker daemon first.
2. Nothing about the container is prompted. The wizard first describes the compute service (SolubleMPNN/ProteinMPNN sequence design, ESM2 scoring, OpenDDE folding, PLIP contact analysis, FoldMason structure alignment), then prints the compute image (the saved value, `OPENDDE_HARNESS_COMPUTE_IMAGE`, or the release's pinned environment tag), the compute device (the GPU inventory reported by `nvidia-smi`, all visible to the container by default, or `cpu`), and the idle timeout. The container is named after the installed code release (`opendde-compute-<code-id>`), and the host port is chosen at each start: `compute_docker.port` when you set one in `config.json` (the wizard then prints it), otherwise the first free port from 8080. Neither is saved by the wizard.
3. Select the OpenDDE fold/refold mode: `api` or `local`. For local placement the wizard prints the two data roots (OpenDDE data from `OPENDDE_ROOT_DIR`, default `~/.cache/opendde`; Harness tool weights from `OPENDDE_HARNESS_WEIGHTS_DIR`, default `~/.cache/opendde-harness`) instead of prompting; override them with those variables or `compute_docker.opendde_data` / `compute_docker.weights_dir` in `config.json`. Only CUDA requires the NVIDIA container runtime.
4. After confirmation, reuse/pull the image, prepare code and mode-specific weights, start the container, and check tool readiness and authentication. API mode does not require OpenDDE weights.

The default tool environment is `aurekaresearch/opendde-harness:v1`. You can select another trusted tag or digest with the matching environment ID and contract hash. Missing images are pulled for `linux/amd64`; no source build is attempted. Harness releases do not automatically rebuild this image.

The runtime image includes neither project code nor model weights. Onboarding automatically prepares the installed release's code plus pinned upstream archives in a versioned host directory. Both folding modes use external SolubleMPNN and ESM2 650M; local folding also needs OpenDDE checkpoint/common data. Missing model assets are downloaded and verified after confirmation. No manual Git checkout is needed for a PyPI installation.

In API mode the wizard does not prompt for the upstream OpenDDE API URL: it prints the configured `fold_defaults.api_url`, or the official default `http://115.190.4.167:30080`. To use another service, set `fold_defaults.api_url` under `plugins.config.protein-design` in `config.json` before running the wizard. This endpoint is separate from the Harness compute URL. Explicit task YAML can override folding defaults; see [API configuration](protein-design.md#use-the-hosted-opendde-folding-api).

Containers bind the compute service API to `127.0.0.1`, use all GPUs (or the `compute_docker.gpus` allow-list) and 16 GiB shared memory. Inside the worker a GPU lease table schedules jobs: a fold holds its GPUs exclusively, while ESM2 and SolubleMPNN jobs share a GPU, so folds on disjoint GPUs run at the same time. Task state is mounted read/write; the complete codebase is mounted read-only at `/workspace`, Harness tool weights at `/weights`, and (local folding only) OpenDDE data at `/opendde`. A compute token is generated once, saved in `config.json`, and reused by every container start.

Pulls and downloads precede the 90-second readiness check. Failed pulls or data preparation start no container. A container that fails readiness is not restarted by Docker; it exits by itself when idle, and connection settings are not saved. Image, GPU or mount changes take effect at the next container start.

Onboarding stops an idle container of the installed release so the confirmed settings apply immediately. A busy container (jobs running or queued) keeps running with its current settings; the new settings apply at its next start.

## Container lifecycle

| Situation | Behavior |
| --- | --- |
| A task starts and no container for the installed release is running | Started automatically (image pulled if missing, code and weights verified), readiness checked, then the task proceeds |
| A container for the installed release is running and answers `/health` | Reused without a restart |
| A design task is running, even between compute calls | The worker refreshes a task lease every 60 s (TTL 180 s); the container counts as busy until the task ends and releases it |
| Idle for `compute_docker.idle_seconds` (default 600) with no job and no task lease | Exits and removes itself (`docker run --rm`, no restart policy) |
| A running task's compute request is refused | The worker resolves the endpoint again once (starting the container when needed, following a new port) and retries |
| The TUI exits | Asked to stop if idle; a busy container keeps running |
| `ddeharness compute stop` | Stops an idle container; otherwise prints the running/queued job and task-lease counts and exits 1. `--force` stops it regardless and waits |
| Client upgrade | The new release starts its own container when needed. The old release's container is left alone and exits when idle |

The running instance is recorded in `~/.opendde_harness/compute/local.json` (container, image, code id, port, URL, start time); `ddeharness doctor` reads it and reports the container, port, code id, running/queued jobs, idle countdown and GPU leases. `config.json` keeps only the placement (`compute_docker`), image, folding mode, GPU selection, idle timeout and the two data roots; `compute_url` records the URL of the last start. Old configurations that still carry `container_name` are read and the field ignored.

Set `OPENDDE_HARNESS_COMPUTE_SOURCE_DIR` only when deliberately using a development
checkout; ordinary installations use managed snapshots.

## Connect to existing service

Choose **Existing Linux compute service** at the placement question to use this flow; the wizard then asks only for the service URL and its compute token. It defaults to the existing service when `plugins.config["protein-design"]` already contains a `compute_url` without local `compute_docker` metadata.

For a remote service, merge its URL into `~/.opendde_harness/config.json` while preserving other settings:

```json
{
  "plugins": {
    "config": {
      "protein-design": {
        "compute_url": "https://compute.example.com"
      }
    }
  }
}
```

Replace the example with the supplied URL. If `compute_docker` metadata also exists, it takes precedence; remove that metadata only when intentionally switching away from locally managed compute.

Run `ddeharness onboard`, confirm the URL and enter the compute token (blank keeps the saved one). No local Docker commands run in this flow. `127.0.0.1` refers to the client machine, not a remote server. Use a trusted network and protect credentials.

The Harness compute token differs from the upstream folding API token. For existing services, configure the upstream token on the compute server, not in scientific YAML.

## Saved settings and task defaults

Settings live in `~/.opendde_harness/config.json`, or the directory selected by `OPENDDE_HARNESS_HOME`. No root `.env` is needed.

Protein Design settings include the compute connection, `fold_defaults`, and local deployment metadata when applicable. Explicit task YAML overrides folding defaults. Running tasks retain their frozen configuration.

If a worker registry exists, onboarding asks before replacing it for new tasks with the selected service. Declining preserves the registry and does not start a container. Cancelling before final confirmation preserves the previous Protein Design configuration.

Use `--skip-protein-design` to skip compute setup. Non-interactive onboarding preserves these settings.

## Validate a design

Service readiness checks do not submit GPU inference or prove scientific correctness. Review a mode-compatible [example](examples/) and validate before launch:

```bash
ddeharness protein-design validate --config docs/examples/crlf2_quickstart.yaml
```

This path is relative to the repository root. For API folding, adapt the YAML to the [API requirements](protein-design.md#use-the-hosted-opendde-folding-api). TUI launch requires approval of the resolved design; CLI `start` launches directly.
