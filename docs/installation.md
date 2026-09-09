# Installation

Client installation, compute startup and OpenDDE data preparation are separate steps.

| Command | Action |
| --- | --- |
| `uv tool install --python 3.12 opendde-harness` | Install the client and bundled terminal UI from PyPI |
| `uv tool upgrade opendde-harness` | Update a PyPI installation |
| `ddeharness doctor` | Read-only check of configuration, compute service and required model assets |
| `ddeharness compute prepare --assets-only` | Prepare required weights outside Docker; no service is started |
| `ddeharness compute prepare --code-only` | Prepare versioned runtime code without model downloads |
| `ddeharness onboard` | Configure the LLM provider, long-term memory and Protein Design compute; reuse/pull an image and start compute after confirmation |
| `ddeharness compute stop` | Stop an idle local compute container; refuses while jobs or task leases are active |

## Install the client

Use Linux or macOS for the client. Local compute requires Linux x86-64.

The PyPI package is `opendde-harness`; its command is `ddeharness`. Both methods
use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage an
isolated Python environment. No environment activation is needed.

### Install from PyPI

```bash
uv tool install --python 3.12 opendde-harness
```

The wheel includes the built TUI and the manifests needed to prepare compute
code and model assets. No source checkout or frontend build is required.

To update, close the TUI and run:

```bash
uv tool upgrade opendde-harness
```

### Install from source

Install Git, uv, and Node.js 22 or newer with npm, then run:

```bash
git clone https://github.com/aurekaresearch/OpenDDE-Harness.git
cd OpenDDE-Harness
uv tool install --python 3.12 .
```

The package build installs frontend dependencies and builds the TUI when the
bundle is missing or older than its sources. Source edits do not automatically
update the installed command. To update, close the TUI, preserve local edits,
and run from the checkout:

```bash
git pull --ff-only
uv tool install --python 3.12 --reinstall .
```

To install edits already in the checkout, omit `git pull`.

### Verify the installation

After either installation:

```bash
ddeharness --version
ddeharness --help
ddeharness tui --check
```

If `ddeharness` is not on PATH, run `uv tool update-shell` and open a new terminal.
Package installation prepares the client; run `ddeharness onboard` to configure
the LLM and compute service and prepare model assets.

The release wheel includes the terminal UI. Source builds require Node.js 22 or
newer with npm before installing the package.

Updates preserve configuration, results, and model data. They do not replace
compute images or stop a running container. Existing containers keep their code
until they exit; subsequent starts use the updated release. Run
`ddeharness compute stop` to stop idle compute early, or let it exit when idle.

## Select a compute image

After the publisher has uploaded the image, pull it on the compute host before creating a local deployment:

```bash
docker pull aurekaresearch/opendde-harness:v1
export OPENDDE_HARNESS_COMPUTE_IMAGE=aurekaresearch/opendde-harness:v1
ddeharness onboard
```

Use this reference or another trusted image with the same environment contract. The default image is `aurekaresearch/opendde-harness:v1`. Environment versions are independent of Harness versions; a client release reuses the image when its required environment has not changed.

Onboarding reuses local images and pulls missing ones for `linux/amd64`; it never builds an image or falls back to building after a pull failure. Existing containers retain their image. Private registries may require `docker login`.

The compute host needs a local Docker daemon, compatible NVIDIA drivers and NVIDIA Container Toolkit for GPU use. The image contains only runtime dependencies and tool binaries. Onboarding prepares a code snapshot from the installed release and pinned upstream archives, and prepares or verifies OpenDDE, SolubleMPNN, ESM2 650M and common data. Code and Harness tool weights are mounted read-only at `/workspace` and `/weights`; in local folding mode OpenDDE data is mounted read-only at `/opendde`, or under `/weights` when it is stored inside the Harness weights root. The image contains no model choices or checkpoint paths; the code supplies them when launching a tool.

Code snapshots live under `runtime-code/` in the Harness weights root
(`~/.cache/opendde-harness/runtime-code/`, or under `OPENDDE_HARNESS_WEIGHTS_DIR`) by
version and content hash. Set `OPENDDE_HARNESS_CODE_CACHE` to use another persistent
directory. Snapshots that earlier releases prepared under `~/.opendde_harness/runtime-code/`
are left untouched; remove that directory once no container mounts it. Updates
create a new snapshot; snapshots that no container mounts are then removed except
the two newest ones (all snapshots are kept when Docker is unavailable). A container
started by an earlier release keeps its code until it exits when idle; the updated
release starts its own container when a task needs it (see the
[container lifecycle](onboarding.md#container-lifecycle)). For development,
set `OPENDDE_HARNESS_COMPUTE_SOURCE_DIR` to a prepared checkout.

The pinned upstream sources (OpenDDE, LigandMPNN, PLIP) are downloaded as archives from `codeload.github.com` with a per-source progress display. Each archive is cached by revision under `runtime-code/sources/`, and a new snapshot reuses the cached archive or the verified sources of a previous snapshot, so a client update downloads again only when a pinned revision changes. If GitHub is slow or unreachable, set `https_proxy` (the downloader honours it) or pass `--upstream-dir DIR` to `ddeharness compute prepare`, where `DIR` holds local Git checkouts at the pinned revisions.

For remote services, see [onboarding](onboarding.md#connect-to-existing-service). Ordinary users do not need the `docker/` directory; it contains only [maintainer build inputs](../docker/README.md).

### Registry access and offline transfer

If Docker Hub is unreachable, obtain an owner-approved alternative registry reference and set `OPENDDE_HARNESS_COMPUTE_IMAGE` before onboarding. Onboarding does not silently use third-party mirrors or change Docker daemon settings. Access to a base-image mirror does not guarantee access to this repository. Keep credentials out of image references.

For offline transfer, the publisher exports the tested image:

```bash
docker save --output opendde-harness.tar aurekaresearch/opendde-harness:v1
sha256sum opendde-harness.tar
```

Transfer the archive, verify its checksum against the publisher's trusted value, then run `docker load --input opendde-harness.tar` on the compute host and set `OPENDDE_HARNESS_COMPUTE_IMAGE` to the loaded reference before onboarding. Transfer the matching code snapshot and weights directory separately; the image archive contains neither. Point `OPENDDE_HARNESS_CODE_CACHE` at the transferred cache parent so the installed release can verify and reuse its snapshot offline. A local tag alone is not proof of provenance.

## Download model assets

Only local OpenDDE folding requires the OpenDDE checkpoint and common data; both folding modes require external SolubleMPNN and ESM2 weights. Run on the compute host after client installation:

```bash
ddeharness compute prepare --assets-only
```

Or select persistent storage for either root:

```bash
ddeharness compute prepare --assets-only --root /path/to/harness-weights --opendde-root /path/to/opendde
```

Two roots, following OpenDDE's own convention:

```text
~/.cache/opendde/                      OpenDDE data (OPENDDE_ROOT_DIR); local folding only
  checkpoint/opendde_abag.pt
  common/
    components.cif
    components.cif.rdkit_mol.pkl
    obsolete_to_successor.json
    release_date_cache.json
  SHA256SUMS
~/.cache/opendde-harness/              Harness tool weights (OPENDDE_HARNESS_WEIGHTS_DIR)
  soluble_mpnn/solublempnn_v_48_020.pt
  huggingface/models--facebook--esm2_t33_650M_UR50D/...
  SHA256SUMS
```

Each root is resolved in the same order: the directory saved by onboarding, the one recorded in `~/.opendde_harness/compute-assets.json` by a previous preparation, the environment variable, then the default above. `~/.cache/opendde` is only ever the OpenDDE data root; the Harness root is never resolved to it. SolubleMPNN/ESM2 files already present there are copied into the Harness root once during preparation instead of being downloaded again. `--root` and `--opendde-root` override the roots for one command; the wizard prints both and does not prompt.

Options:

- `--mode api`: prepare ESM2 650M and SolubleMPNN without OpenDDE weights/common data (the default without a configured folding mode).
- `--mode local`: also prepare OpenDDE weights/common data for local CPU/CUDA inference.
- `--checkpoint opendde.pt`: prepare the general checkpoint instead of the antibody-antigen default `opendde_abag.pt`; local mode only.
- `--download-workers 1`: download serially; supported range 1–4, default 2.
- `--code-only` / `--assets-only`: prepare only the runtime code snapshot, or only the weights. Without either, both are prepared.

`ddeharness doctor` reports what is present without downloading, preparing or repairing anything; add `--verify-hashes` to verify every required asset against its published SHA256, `--compute-only` to skip the LLM and memory checks, `--probe` to send one test message to the LLM, and `--json` for machine-readable output. Preparation lives in `ddeharness compute prepare`.

The OpenDDE checkpoint downloads first, followed by common data and shared models. Every file streams over HTTPS with a progress bar (size, speed, ETA), resumes an interrupted transfer where it stopped, and is verified against its SHA256 while it downloads; verified files are reused and mismatched existing files are reported without being overwritten. Proxies come from the usual `https_proxy` variables. Hugging Face URLs fall back to `HF_ENDPOINT` (or `hf-mirror.com`) when `huggingface.co` is unreachable; the `hf` CLI and `curl` are not needed. SolubleMPNN weights are fetched from `ipd.graylab.jhu.edu` first, then the project's Hugging Face repository, then `files.ipd.uw.edu`; every source is verified against the same SHA256. Large local MSA/template databases are not included.

Asset preparation does not reinstall the client or start Docker. Onboarding uses the same preparation functions. Data stays outside the image and is mounted read-only.

## Uninstall

After stopping active tasks and closing the TUI, remove the client with:

```bash
uv tool uninstall opendde-harness
```

This leaves configuration, results, compute services, model data and the source checkout intact. Back up anything needed before removing those separately. Do not remove shared Docker resources, GPU drivers or system runtimes as part of client cleanup.

Application settings live in `~/.opendde_harness/config.json`. `OPENDDE_HARNESS_HOME` does not relocate this file or the protein-design task root; it overrides selected runtime directories such as Node.js, compute state, and tracing. Use `OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT` for task state. Protect the configuration file because it may contain credentials. Default locations include:

```text
~/.opendde_harness/
  config.json                Provider, memory and Protein Design settings
  workspace/                 Agent workspace: TOOLS.md, skills/, agent_memory/,
                             user_memory/, seeded from the bundled templates
  memory/                    Long-term memory store
  compute/local.json         The running local compute instance
  protein_design/<task_id>/  Task state and worker logs
  runtime/                   Private Node.js runtime, when one was installed
  logs/                      Client logs, including memory-server.log
```

Original compute outputs live on the compute host. Captured dashboard structures and metrics are stored in tracing data. Model weights and OpenDDE data live under the two roots above.

The verification commands check CLI/TUI startup, not GPU inference. Onboarding verifies service readiness and authentication before saving compute settings. See [troubleshooting](troubleshooting.md) for installation, image-pull and mount failures.
