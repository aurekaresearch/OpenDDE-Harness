#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
dry_run=0
push=0
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --help)
            echo "Usage: bash docker/build.sh [--dry-run] [--push] [IMAGE_TAG ...]"
            echo "Default tag comes from docker/environment.json; builds only, never starts containers."
            echo "Builds a runtime-only image. Source code and all model assets must be mounted at runtime."
            echo "--push exports directly to the registry instead of loading the image locally."
            echo "Python packages and Triton use TUNA; PyTorch CUDA wheels use NJU. Override PYPI_INDEX_URL, PYTORCH_WHEEL_BASE and TRITON_WHEEL_URL if needed."
            echo "Base images: use local official cache, otherwise pull official first, then m.daocloud.io on failure."
            echo "DOCKER_MIRROR_PREFIX overrides the fallback registry prefix; empty disables it. IMAGE_PULL_TIMEOUT defaults to 1800 seconds per attempt."
            exit 0
            ;;
        --dry-run) dry_run=1 ;;
        --push) push=1 ;;
        *) echo "Unknown option: $1; use --help" >&2; exit 2 ;;
    esac
    shift
done
metadata=$(python3 -c 'import hashlib,json,sys; p=json.load(open(sys.argv[1])); print(p["id"],p["image"],p["cuda_image"],p["uv_image"],p["foldmason_revision"],hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",", ":")).encode()).hexdigest(),sep="\n")' "$script_dir/environment.json")
environment=()
while IFS= read -r value; do
    environment+=("$value")
done <<< "$metadata"
[[ ${#environment[@]} -eq 6 && "${environment[0]}" =~ ^[a-z0-9][a-z0-9._-]*$ && "${environment[4]}" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid environment.json" >&2; exit 2; }
image_names=("${@:-${environment[1]}}")
mirror_prefix="${DOCKER_MIRROR_PREFIX-m.daocloud.io}"
pull_timeout="${IMAGE_PULL_TIMEOUT:-1800}"
[[ "$pull_timeout" =~ ^[1-9][0-9]*$ ]] || { echo "IMAGE_PULL_TIMEOUT must be a positive number of seconds" >&2; exit 2; }
if [[ -n "$mirror_prefix" && ! "$mirror_prefix" =~ ^[a-zA-Z0-9.-]+(:[0-9]+)?(/[a-zA-Z0-9._-]+)*$ ]]; then
    echo "DOCKER_MIRROR_PREFIX must be a registry prefix without scheme or credentials" >&2
    exit 2
fi
resolve_image() {
    local original="$1" fallback
    if docker image inspect "$original" >/dev/null 2>&1; then
        echo "Using local image: $original" >&2
        printf '%s\n' "$original"
        return
    fi
    echo "Pulling official image: $original" >&2
    if timeout "$pull_timeout" docker pull --platform linux/amd64 "$original" >&2; then
        printf '%s\n' "$original"
        return
    fi
    [[ -n "$mirror_prefix" ]] || { echo "Official pull failed; mirror fallback is disabled" >&2; return 1; }
    fallback="$mirror_prefix/$original"
    echo "Official pull failed; retrying the same image version via $fallback" >&2
    if timeout "$pull_timeout" docker pull --platform linux/amd64 "$fallback" >&2; then
        printf '%s\n' "$fallback"
        return
    fi
    echo "Both official and mirror pulls failed for $original; no build started" >&2
    return 1
}
cuda_image="${environment[2]}"
uv_image="${environment[3]}"
if [[ "$dry_run" -eq 0 ]]; then
    command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 2; }
    cuda_image=$(resolve_image "$cuda_image")
    uv_image=$(resolve_image "$uv_image")
fi
command=(docker buildx build --platform linux/amd64
    --build-arg "PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
    --build-arg "PYTORCH_WHEEL_BASE=${PYTORCH_WHEEL_BASE:-https://mirrors.nju.edu.cn/pytorch/whl}"
    --build-arg "TRITON_WHEEL_URL=${TRITON_WHEEL_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/24/5f/950fb373bf9c01ad4eb5a8cd5eaf32cdf9e238c02f9293557a2129b9c4ac/triton-3.3.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl}"
    --build-arg "CUDA_IMAGE=$cuda_image"
    --build-arg "UV_IMAGE=$uv_image"
    --build-arg "FOLDMASON_REV=${environment[4]}"
    --build-arg "ENVIRONMENT_ID=${environment[0]}"
    --build-arg "ENVIRONMENT_SHA256=${environment[5]}"
    --file "$script_dir/Dockerfile")
if [[ "$push" -eq 1 ]]; then
    command+=(--push)
else
    command+=(--load)
fi
for image_name in "${image_names[@]}"; do
    command+=(--tag "$image_name")
done
command+=("$project_root")
if [[ "$dry_run" -eq 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    "${command[@]}"
fi
