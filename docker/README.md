# Build the compute runtime image

The image contains Python dependencies, CUDA runtime libraries and tool binaries.
Harness, OpenDDE, LigandMPNN and PLIP source code and all model assets remain on
the host. Mount the codebase at `/workspace`, Harness tool weights at `/weights` and, for local folding, OpenDDE data at `/opendde`, all
read-only. Results use a separate writable mount at `/output`.

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

The client pulls `aurekaresearch/opendde-harness:v1`; the build defaults to the
historical reference in `environment.json`. Pass an explicit tag to choose another
build destination. The contract's canonical JSON determines the image compatibility
label, so changing its `image` field would invalidate existing images.
Building loads the image locally without starting containers. The dry run needs
no model assets and performs no network requests.

## Prepare code and model assets

For installed releases, use `ddeharness onboard`; see
[installation](../docs/installation.md) and
[container lifecycle](../docs/onboarding.md#container-lifecycle).
The following checks require a development checkout with pinned upstream sources:

```bash
ddeharness compute prepare --sources-only "$PWD"
```

This prepares `external/{opendde,ligandmpnn,plip,foldmason}` and preserves existing
checkouts with local changes or a different revision. Source revisions live in
`versions.env`; the FoldMason binary revision lives in `environment.json`.

Prepare weights with the current CLI:

```bash
ddeharness compute prepare --assets-only --root /shared/harness-weights
```

Harness tool weights and OpenDDE data use separate roots. Follow the command's
reported paths when mounting them; OpenDDE data defaults to `~/.cache/opendde`.

To copy existing local assets without downloading, use an empty staging directory:

```bash
mkdir -p /shared/staged-models
SOLUBLE_MPNN_WEIGHTS=/existing/soluble_mpnn \
ESM_CACHE=/existing/huggingface \
OPENDDE_CHECKPOINT=/existing/opendde_abag.pt \
OPENDDE_COMMON_DIR=/existing/common \
  bash docker/prepare-models.sh /shared/staged-models
```

The script checks pinned SolubleMPNN/ESM hashes, preserves the checkpoint filename,
and records checksums for the supplied OpenDDE files. Recorded checksums do not
establish checkpoint compatibility. Hugging Face caches with or without `hub/`
are accepted. This staging directory contains both tool weights and OpenDDE data;
mount it at both `/weights` and `/opendde` for local folding.

## Publish manually

There is no CI image publishing workflow. To publish a maintainer-built image,
authenticate with the target registry and supply explicit tags:

```bash
bash docker/build.sh --push YOUR_REGISTRY/opendde-harness:YOUR_TAG
```

Bump the environment ID when dependencies or tool binaries change; keep released
environment tags immutable. Publishing does not update running containers.
`environment.json`, `versions.env`, model checksums and the ESM license are also
packaged with the client and must remain available.

## Verify with mounts

```bash
docker run --rm --read-only --tmpfs /tmp --network none \
  --mount type=bind,source="$PWD",target=/workspace,readonly \
  --entrypoint python aurekaresearch/opendde-harness:v1 /workspace/docker/doctor.py

mkdir -p /shared/harness-validation
docker run --rm --read-only --tmpfs /tmp --network none \
  --mount type=bind,source="$PWD",target=/workspace,readonly \
  --mount type=bind,source=/shared/harness-weights,target=/weights,readonly \
  --mount type=bind,source=/shared/harness-validation,target=/validation \
  --entrypoint python aurekaresearch/opendde-harness:v1 /workspace/docker/model-doctor.py \
  --device cpu --mode api --output /validation/cpu-report.json
```

These manual checks use a development checkout. Installed-release onboarding
mounts its prepared code snapshot instead. Image builds verify dependencies without application code;
the mounted checks above verify source imports and offline CPU ESM execution.
Supply `--structure` with a mounted two-chain CIF/PDB to also verify SolubleMPNN,
PLIP, FoldMason and structural analysis. For local OpenDDE prediction and confidence scoring, add `--mode local`, mount
the OpenDDE data root at `/opendde`, and provide `--structure`. Repeat with `--device cuda` and a
Docker GPU device request for CUDA verification; each run needs an empty output directory. The examples use the published image;
substitute your local tag to validate a new build.
Editing mounted
source during a running job is unsupported; restart workers after source updates.

## Build inputs

| File | Purpose |
| --- | --- |
| `Dockerfile` | Runtime-only build with independent dependency and FoldMason stages |
| `Dockerfile.dockerignore` | Allowlisted build context |
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
