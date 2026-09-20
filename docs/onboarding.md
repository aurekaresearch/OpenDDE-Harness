# Onboarding

The first `ddeharness` run starts the onboarding wizard in the terminal before the TUI opens when no provider is configured; without a terminal it exits and asks you to run `ddeharness onboard`. Run `ddeharness onboard` at any time to change settings. There is no wizard inside the TUI: `/setup` prints a reminder to exit and run the terminal command. Providers and models are the exception: `/model` connects a provider, stores its key and switches models from inside the UI (see the [TUI guide](tui.md)).

The four steps are **LLM**, **long-term memory**, **Protein Design** and **web search**. Memory runs on your default model through the model service, so its step asks one question (on or off) and needs no key, endpoint or model of its own; skip it with `--skip-memory` and skip compute setup with `--skip-protein-design`. The web-search step asks for a Brave Search API key (free tier: 2,000 queries a month, from https://api-dashboard.search.brave.com/app/keys); press Enter to skip, and `web_search` uses DuckDuckGo without a key. A provider with a hosted search of its own (the Codex login, OpenAI, Anthropic) searches through that regardless. Pass `--brave-api-key` to set it without prompts.

Onboarding does not launch a design task or install Docker/GPU drivers.

Long-term memory summarises each session into memories with the same model the
conversation runs on (a Codex sign-in included) and recalls them by keyword; there
is no embedding or rerank model. Design tasks still run with memory off, but do not
retain long-term cases and learned skills. Turn it on later with `ddeharness onboard`.

## Step 1: the LLM provider

The first step is the TUI's `/login`, at the terminal: pi's own sign-in, step
for step and word for word. **Select authentication method:** asks which way in
-- **Sign in with an account** or **Sign in with an API key** -- then **Select
provider to configure:** offers the providers that take that way in, and the
sign-in runs right there (pi's login-method menu, then the device code or the
sign-in link), or the key form asks for one masked key. The last option under
the key method, **OpenAI Compatible**, declares an endpoint of your own -- a
provider id, a base URL, an optional key and, only for an endpoint that
publishes no `GET /models`, the model ids -- exactly as the TUI's form does.
Both doors call the same gateway handlers, so what the wizard can connect is
what `/login` can, and a provider is written the same way from either.

The wizard offers the handful almost everybody picks -- OpenAI Codex, OpenAI,
Anthropic, Google, OpenRouter, DeepSeek, Moonshot AI, Z.AI, xAI -- and the
endpoint row; every other provider pi ships is a `/login` away inside the TUI
once the setup is done. Azure OpenAI is a tenant's own resource and is written
with `ddeharness provider set azure-openai-responses --base-url <resource>
--api azure-openai-responses --api-key <key>`; so is any endpoint on a wire
other than Chat Completions.

After the provider is connected the wizard does what `/login` leaves to
`/model`: it checks that the provider answers, asks which model is the
default, and sends one test message (`--skip-test` skips it). Without prompts,
`--provider` with `--api-key` connects one of pi's own and `--provider` with
`--base-url` (plus `--model` for an endpoint that publishes no list) declares
an endpoint; `--provider` alone goes straight to that provider, as `/login
<provider>` does.

## The providers section

The wizard writes `providers` in `~/.opendde_harness/config.json`, and the
section is [pi-ai](https://github.com/earendil-works/pi)'s own `models.json`
shape: a map from pi provider id to a provider declaration. So pi's own
documentation describes this section, and a key is written exactly once, with no
aliases and no second spelling.

```json
"agents": { "defaults": { "model": "openai-codex/gpt-5.5-codex" } },
"providers": {
  "openai-codex": { "login": "oauth" },
  "anthropic":    { "apiKey": "sk-ant-..." },
  "openrouter":   { "apiKey": "sk-or-...", "models": ["anthropic/claude-opus-4-5"] },
  "my-vllm": {
    "baseUrl": "http://gpu-box:8000/v1",
    "api": "openai-completions",
    "models": [
      { "id": "qwen3-32b", "name": "Qwen3 32B", "contextWindow": 131072, "maxTokens": 32768 }
    ]
  }
}
```

Four rules cover the whole section.

**A key is a pi provider id.** `anthropic`, `openai`, `openai-codex`, `google`,
`openrouter`, `deepseek`, `groq`, `xai`, `mistral`, `zai`, `minimax`,
`moonshotai` and the rest; `/login` in the TUI lists them all. Any other key is
a provider your config declares itself, and the name is yours to choose.

**`baseUrl` says which kind an entry is.** No address means one of pi's own: pi
carries the address, the protocol and the model catalogue, so the entry adds the
credential and nothing else. An address means you are declaring the provider,
and an address needs the protocol it serves, so `api` is required beside it --
one of `openai-completions`, `openai-responses`, `anthropic-messages`,
`azure-openai-responses`, `google-generative-ai`, `mistral-conversations`,
`openai-codex-responses`. It is declared and never guessed from the URL. The
same is true the other way round: `api` without `baseUrl` is refused, because
naming a protocol only means something for a provider you are declaring.

**A credential is a key, a sign-in, or the environment.** `apiKey` holds a key.
`"login": "oauth"` says the credential is a sign-in, which lives in the model
service's credential store and never in this file -- run
`ddeharness provider login <id>`. An entry with neither is still a declaration:
pi reads the vendor's own variable itself (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
and the rest), and a self-hosted server usually wants no key at all.

**A model id names its provider.** `agents.defaults.model` is
`"<provider>/<model>"` and the prefix is the only thing that says who serves it;
a bare id is refused rather than routed by its spelling. A provider you declare
must list its models, because pi has no catalogue for one. An entry in `models`
is either the id the endpoint serves or a row describing it -- `name`,
`contextWindow`, `maxTokens`, `reasoning`, `input`, `cost`, plus this project's
own `temperature`, `reasoningEffort` and `catalogModel`. Write a row for what no
catalogue can know: a deployment you sized yourself, a model newer than pi's
catalogue, or a deployment name (`catalogModel` then says which catalogue model
is behind it). `ddeharness provider model set <provider> <model>` writes one.

A config written by an earlier release is refused with one message naming the
change; nothing is migrated, and `ddeharness onboard` writes a fresh file.

## Local Docker compute

Unless an external compute service is already configured, Protein Design setup manages a local compute container. The container is ephemeral: it starts on demand when a task needs it and removes itself after an idle timeout (see [container lifecycle](#container-lifecycle)).

1. Choose local Linux Docker or an existing Linux compute service. Local setup checks the platform and Docker daemon first.
2. Nothing about the container is prompted. The wizard first describes the compute service (SolubleMPNN/ProteinMPNN sequence design, ESM2 scoring, OpenDDE folding, PLIP contact analysis, FoldMason structure alignment), then prints the compute image (`OPENDDE_HARNESS_COMPUTE_IMAGE`, or the release's pinned environment tag), the compute device (the GPU inventory reported by `nvidia-smi`, all visible to the container by default, or `cpu`), and the idle timeout. The container is named after the installed code release (`opendde-compute-<code-id>`), and the host port is chosen at each start: `compute_docker.port` when you set one in `config.json` (the wizard then prints it), otherwise the first free port from 8080. The generated container name and automatically selected port are runtime state; an explicitly configured `compute_docker.port` is preserved.
3. Select the OpenDDE fold/refold mode: `api` or `local`. For local placement the wizard prints the two data roots (OpenDDE data from `OPENDDE_ROOT_DIR`, default `~/.cache/opendde`; Harness tool weights from `OPENDDE_HARNESS_WEIGHTS_DIR`, default `~/.cache/opendde-harness`) instead of prompting; override them with those variables or `compute_docker.opendde_data` / `compute_docker.weights_dir` in `config.json`. Only CUDA requires the NVIDIA container runtime.
4. After confirmation, reuse/pull the image, prepare code and mode-specific weights, start the container, and check tool readiness and authentication. API mode does not require OpenDDE weights.

The default tool environment is `aurekaresearch/opendde-harness:v1`. You can select another trusted tag or digest with the matching environment ID and contract hash. Missing images are pulled for `linux/amd64`; no source build is attempted. Harness releases do not automatically rebuild this image.

The default image does not include PyRosetta. For optional relaxation/interface
scoring, provision a [licensed custom image](../docker/README.md#optional-pyrosetta-runtime)
and set `OPENDDE_HARNESS_COMPUTE_IMAGE` before each onboarding invocation. Confirm
the displayed image; rerunning the wizard without that override selects the
default. Subsequent automatic worker starts use the saved image. A task YAML
enabling analysis does not install dependencies or select a different image.

The runtime image includes neither project code nor model weights. Onboarding automatically prepares the installed release's code plus pinned upstream archives in a versioned host directory. Both folding modes use host-side SolubleMPNN and ESM2 650M weights; local folding also needs OpenDDE checkpoint/common data. Missing model assets are downloaded and verified after confirmation. No manual Git checkout is needed for a PyPI installation.

In API mode the wizard does not prompt for the upstream OpenDDE API URL: it prints the configured `fold_defaults.api_url`, or the official default `https://api.aurekabio.cloud`. To use another service, set `fold_defaults.api_url` under `plugins.config.protein-design` in `config.json` before running the wizard. This endpoint is separate from the Harness compute URL. The official folding gateway, ProTrek, and Target MSA service domains are included in the container's default `NO_PROXY` list so these services connect directly even when the host uses a loopback proxy. Explicit task YAML can override folding defaults; see [API configuration](protein-design.md#use-the-hosted-opendde-folding-api).

Containers bind the compute service API to `127.0.0.1` and use 16 GiB shared memory by default. The wizard preserves `compute_docker.gpus` when configured; otherwise it selects all GPUs when Docker advertises the NVIDIA runtime, or CPU mode when it does not. Inside the worker a GPU lease table schedules jobs: a fold holds its GPUs exclusively, while ESM2 and SolubleMPNN jobs share a GPU, so folds on disjoint GPUs run at the same time. Task state is mounted read/write; runtime code is mounted read-only at `/workspace` and Harness tool weights at `/weights`. In local folding mode, OpenDDE data uses `/opendde` when stored outside the Harness weights root, or the corresponding subdirectory under `/weights` otherwise. A compute token is generated once, saved in `config.json`, and reused by every container start.

Pulls and downloads precede the 90-second readiness check. Failed pulls or data preparation start no container. A container that fails readiness is not restarted by Docker; it exits by itself when idle, and connection settings are not saved. Image, GPU or mount changes take effect at the next container start.

Onboarding stops an idle container of the installed release so the confirmed settings apply immediately. A busy container (jobs running or queued) keeps running with its current settings; the new settings apply at its next start.

## Container lifecycle

| Situation | Behavior |
| --- | --- |
| A task starts and no container for the installed release is running | Started automatically (image pulled if missing, code and weights verified), readiness checked, then the task proceeds |
| A container for the installed release is healthy and its actual image matches the requested image | Reused without a restart |
| A design task is running, even between compute calls | The worker refreshes a task lease every 60 s (TTL 180 s); the container counts as busy until the task ends and releases it |
| Idle for `compute_docker.idle_seconds` (default 600) with no job and no task lease | Exits and removes itself (`docker run --rm`, no restart policy) |
| A running task's compute request is refused | The worker resolves the endpoint again once (starting the container when needed, following a new port) and retries |
| The TUI exits | Asked to stop if idle; a busy container keeps running |
| `ddeharness compute stop` | Stops an idle container; otherwise prints the running/queued job and task-lease counts and exits 1. `--force` stops it regardless and waits |
| Code release or configured image changes while the previous worker is busy | New startup is refused with a retry-after-completion message; the existing task and recorded service state are preserved |
| Code release or configured image changes after the previous worker is idle | The previous service must confirm idle shutdown and exit before a replacement starts |

The running instance is recorded in `~/.opendde_harness/compute/local.json` (container, image, code id, port, URL, start time); `ddeharness doctor` reads it and reports the container, port, code id, running/queued jobs, idle countdown and GPU leases. `config.json` keeps only the placement (`compute_docker`), image, folding mode, GPU selection, idle timeout and the two data roots; `compute_url` records the URL of the last start. Old configurations that still carry `container_name` are read and the field ignored.

Set `OPENDDE_HARNESS_COMPUTE_SOURCE_DIR` only when deliberately using a development
checkout; ordinary installations use managed snapshots.

## Connect to existing service

Choose **Existing Linux compute service** at the placement question to use this flow; the wizard asks for the service URL and its compute token, followed by the OpenDDE folding mode. It defaults to the existing service when `plugins.config["protein-design"]` already contains a `compute_url` without local `compute_docker` metadata.

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

The Harness compute token authenticates the Harness compute service. The current fold integration sends no upstream OpenDDE API token; the selected upstream endpoint must accept its requests without that header.

## Saved settings and task defaults

Settings live in `~/.opendde_harness/config.json`. `OPENDDE_HARNESS_HOME` changes some runtime directories, but does not change the CLI configuration-file location. No root `.env` is needed.

All `compute_docker` and `fold_defaults` fields belong under `plugins.config["protein-design"]`. Protein Design settings include the compute connection and local deployment metadata when applicable. Explicit task YAML overrides folding defaults. Running tasks retain their frozen configuration.

`protein-design validate` and `protein-design start` accept `--opendde-config`
for a non-default application configuration. The detached worker receives that
same resolved path; use the same file for both commands. Its provider/compute
settings are distinct from the design YAML passed with `--config`.

If a worker registry exists, onboarding asks before replacing it for new tasks with the selected service. Declining preserves the registry and does not start a container. Cancelling before final confirmation preserves the previous Protein Design configuration.

Use `--skip-protein-design` to skip compute setup. Non-interactive onboarding preserves these settings. The local wizard rebuilds `compute_docker` and does not retain a manually added `env` mapping. Apply such overrides after onboarding; stop the idle service with `ddeharness compute stop` and let the next task start it with the saved overrides.

## Validate a design

Service readiness checks do not submit GPU inference or prove scientific correctness. Review a mode-compatible [example](examples/) and validate before launch:

```bash
ddeharness protein-design validate --config docs/examples/crlf2_quickstart.yaml
```

This path is relative to the repository root. For API folding, adapt the YAML to the [API requirements](protein-design.md#use-the-hosted-opendde-folding-api). TUI launch requires approval of the resolved design; CLI `start` launches directly.
