# Build the compute runtime image

The image contains Python dependencies, CUDA runtime libraries and tool binaries.
Harness, OpenDDE, LigandMPNN and PLIP source code and all model assets remain on
the host. Mount the codebase at `/workspace`, Harness tool weights at `/weights` and, for local folding, OpenDDE data at `/opendde`, all
read-only. Results use a separate writable mount at `/output`.

## Publish to Docker Hub

The `Protein design tool environment` workflow publishes the image reference in
`environment.json` and an `env-sha-...` tag. It runs manually from `main` or from
an `env-<environment-id>` Git tag, such as `env-cu128-torch271-r2`. Ordinary `v*`
Harness releases do not build or publish environment images. Bump the environment
ID when dependencies or tool binaries change; do not overwrite released environment
tags. Publishing does not update running containers.

Configure a dedicated Linux x86-64 self-hosted runner with the `opendde-build`
label, Docker Buildx, Python 3, GNU `timeout` and sufficient disk space. In the `dockerhub`
GitHub environment, configure:

- Secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`, with push access to the repository.

Protect this environment with required approval and restrict publishing to trusted
branches/tags. Only trusted code should run on a self-hosted publisher. The workflow
does not need model assets. It authenticates and runs `docker/build.sh --push`,
exporting directly to the registry without loading the final image locally.

## Build

Use a Linux x86-64 host with Docker Buildx/BuildKit, GNU `timeout`, network access
and enough disk space for CUDA dependencies. From the repository root:

```bash
bash docker/build.sh
```

No source checkout or weight files are copied into the image. Build stages read
the independent environment contract and compile FoldMason; only pinned dependencies,
the tool binary, license and version records enter the final stage. GCC and
Python headers remain for runtime Triton compilation. The full CUDA development
suite, CMake and Rust remain outside the final image.

```bash
bash docker/build.sh --help
bash docker/build.sh --dry-run
```

The harness pulls `aurekaresearch/opendde-harness:v1`. A local build tags `farmertao/opendde-harness:env-cu128-torch271-r2`, the reference `environment.json` was sealed with: that file is hashed byte for byte into every published image's `org.opendde-harness.environment-sha256` label, so its `image` field cannot be re-pointed without invalidating images that are already published. Building loads the image locally;
it does not start containers or replace existing deployments. The dry run needs
no model assets and performs no network requests.

## Prepare the host codebase and weights

For a PyPI installation, the public package name is `opendde-harness` and its
CLI is `ddeharness`. No Harness Git checkout is needed:

```bash
ddeharness compute prepare --assets-only --root /shared/harness-weights
ddeharness compute prepare --code-only
ddeharness onboard
```

Onboarding normally prepares code automatically. `--code-only` prewarms the cache
without downloading weights or starting Docker. It copies only the installed
Harness code/resources, downloads the pinned upstream source archives, and records
all file hashes in an immutable version directory under `runtime-code/` in the
Harness weights root (`~/.cache/opendde-harness/runtime-code/`, or under
`OPENDDE_HARNESS_WEIGHTS_DIR`). Dependencies and host virtual environments are
not copied. Repeated preparation verifies and reuses the directory; a code change
creates another directory. Upstream archives are cached by revision under
`runtime-code/sources/` and reused (or linked from a previous verified snapshot),
so a new directory downloads nothing unless a pinned revision changed. Existing
snapshots are never overwritten; after a new directory is prepared, snapshots that
no container mounts are deleted except the two newest ones.

Use `--code-cache /shared/runtime-code` to prepare code on persistent storage, and
`--upstream-dir /path/to/verified/external` to reuse local pinned Git checkouts
without downloading upstream archives. Missing or incompatible environment labels
stop onboarding before a compute container is created. Startup also checks the
actual Python dependency versions against the mounted code's environment contract.
Set `OPENDDE_HARNESS_CODE_CACHE=/shared/runtime-code` for onboarding to use the same
custom cache. A one-off `--code-cache` argument does not change saved client settings.

For development, set `OPENDDE_HARNESS_COMPUTE_SOURCE_DIR` to a prepared checkout
before `ddeharness onboard`. This uses that checkout directly instead of a release
snapshot. Prepare its upstream checkouts as follows:

From the Harness repository root, prepare the pinned upstream source checkouts:

```bash
ddeharness compute prepare --sources-only "$PWD"
```

This populates `external/{opendde,ligandmpnn,plip,foldmason}`. Existing checkouts
with another revision or local changes are reported and left unchanged. The
FoldMason source is retained on the host alongside its runtime binary's revision.

Prepare an empty persistent model directory once, independently of image builds:

```bash
mkdir -p /shared/harness-weights
SOLUBLE_MPNN_WEIGHTS=/existing/soluble_mpnn \
ESM_CACHE=/existing/huggingface \
OPENDDE_CHECKPOINT=/existing/opendde.pt \
OPENDDE_COMMON_DIR=/existing/common \
  bash docker/prepare-models.sh /shared/harness-weights
```

The script copies real files, checks the pinned SolubleMPNN and ESM2 650M hashes,
and records checksums for the supplied OpenDDE checkpoint and common data. It
never downloads weights or overwrites a populated destination. The OpenDDE
checkpoint is stored under `checkpoint/` by its own filename; the harness
prepares `opendde_abag.pt` by default and `opendde.pt` on request, and a
supplied custom checkpoint still requires compatibility verification. Hugging Face caches with or without
`hub/` are accepted.

```text
harness-weights/
  checkpoint/opendde_abag.pt
  soluble_mpnn/solublempnn_v_48_020.pt
  huggingface/models--facebook--esm2_t33_650M_UR50D/...
  common/{components.cif,components.cif.rdkit_mol.pkl,obsolete_to_successor.json,release_date_cache.json}
  SHA256SUMS
```

## Container lifecycle

The client never keeps a long-lived container. `ddeharness onboard` and every task
start resolve the compute endpoint the same way: reuse the running container of the
installed code release (`opendde-compute-<code-id>`) when it answers `/health`,
otherwise `docker run --rm -d` a new one with the mounts above, `--gpus all` (or the
configured allow-list), no restart policy and `OPENDDE_HARNESS_COMPUTE_IDLE_SECONDS`
(default 600). The service exits by itself after that many idle seconds with no
job and no task lease (a detached design worker refreshes `PUT /leases/<task_id>`
every 60 s while it runs, so long LLM phases do not count as idle), and
`ddeharness compute stop` asks it to exit now (`POST /shutdown`; `--force` drains
running jobs first). The running instance is recorded in
`~/.opendde_harness/compute/local.json`. After a client upgrade the new release
starts its own container; the previous release's container is left alone and
exits when idle, so the two coexist while old tasks drain. See
[onboarding](../docs/onboarding.md#container-lifecycle).

## Verify and run with mounts

```bash
docker run --rm --read-only --tmpfs /tmp --network none \
  --mount type=bind,source="$PWD",target=/workspace,readonly \
  --mount type=bind,source=/shared/harness-weights,target=/weights,readonly \
  --entrypoint python aurekaresearch/opendde-harness:v1 /workspace/docker/doctor.py

mkdir -p /shared/harness-validation
docker run --rm --read-only --tmpfs /tmp --network none \
  --mount type=bind,source="$PWD",target=/workspace,readonly \
  --mount type=bind,source=/shared/harness-weights,target=/weights,readonly \
  --mount type=bind,source=/shared/harness-validation,target=/validation \
  --entrypoint python aurekaresearch/opendde-harness:v1 /workspace/docker/model-doctor.py \
  --device cpu --mode api --output /validation/cpu-report.json

mkdir -p /shared/harness-output
docker run --rm --gpus all --shm-size 16g -p 127.0.0.1:8080:8080 \
  --mount type=bind,source="$PWD",target=/workspace,readonly \
  --mount type=bind,source=/shared/harness-weights,target=/weights,readonly \
  --mount type=bind,source=/shared/opendde,target=/opendde,readonly \
  --env OPENDDE_ROOT_DIR=/opendde --env STRUCTPRED_OPENDDE_ROOT_DIR=/opendde \
  --mount type=bind,source=/shared/harness-output,target=/output \
  aurekaresearch/opendde-harness:v1 \
  python -m uvicorn opendde_harness.plugin.protein_design.servers.api:create_app \
  --factory --host 0.0.0.0 --port 8080
```

These manual checks use a development checkout. Installed-release onboarding
mounts its prepared code snapshot instead. Image builds verify dependencies without application code;
the mounted checks above verify source imports and offline CPU ESM execution.
Supply `--structure` with a mounted two-chain CIF/PDB to also verify SolubleMPNN,
PLIP, FoldMason and structural analysis. Add `--mode local` for a complete local
OpenDDE prediction and confidence scoring. Repeat with `--device cuda` and a
Docker GPU device request for CUDA verification; each run needs a fresh report path.
Editing mounted
source during a running job is unsupported; restart workers after source updates.

## Build inputs

| File | Purpose |
| --- | --- |
| `Dockerfile` | Runtime-only build with independent dependency and FoldMason stages |
| `Dockerfile.dockerignore` | Allowlisted source build context |
| `build.sh` | Buildx entry point and base-image resolution |
| `environment.json` | Independent environment ID, pinned base images, Python packages and tool binary revision |
| `versions.env` | Upstream source and ESM snapshot revisions |
| `prepare-models.sh` | Prepare a persistent external model directory once |
| `model-checksums.sha256` | Verify staged model assets |
| `doctor.py` | Check dependencies during builds and mounted source imports at runtime |
| `model-doctor.py` | Verify mode-specific assets and run offline CPU/CUDA tool checks with a JSON report |
| `ESM-LICENSE.txt` | License copied with external ESM assets |

Model selections and checkpoint paths are defined by the mounted code, not the
image. The image defaults to a Python interpreter; the client supplies the service
command. Harness tool weights are read from `/weights` (Hugging Face offline mode enabled) and OpenDDE data from `/opendde` (`OPENDDE_ROOT_DIR`); local folding needs the second mount. Missing
mounts fail instead of falling back to an application or model copy in the image.

## Network settings

Base images reuse the local official cache, otherwise pull from the official
registry first and fall back to `m.daocloud.io`. Set `DOCKER_MIRROR_PREFIX` to a
trusted prefix, or to an empty string to disable fallback. A mirrored tag alone
does not prove upstream identity. `IMAGE_PULL_TIMEOUT` defaults to 1800 seconds
per attempt. This does not change Docker daemon settings.

Python packages and Triton use TUNA; CUDA PyTorch wheels use NJU. To use the
official package servers, supply these overrides:

```bash
PYPI_INDEX_URL=https://pypi.org/simple \
PYTORCH_WHEEL_BASE=https://download.pytorch.org/whl \
TRITON_WHEEL_URL=https://download.pytorch.org/whl/triton-3.3.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
bash docker/build.sh
```

The wheel hashes remain verified. APT and upstream source downloads still need
network access. No credentials or entire user caches are included in the image.
