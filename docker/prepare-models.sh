#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
source "$script_dir/versions.env"
destination="${1:?Expected an empty staging directory}"
[[ -d "$destination" && -z "$(ls -A "$destination")" ]] || { echo "Model staging directory must be empty" >&2; exit 2; }
mpnn_dir="${SOLUBLE_MPNN_WEIGHTS:-$project_root/external/ligandmpnn/model_params}"
esm_cache="${ESM_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
opendde_root="${OPENDDE_ROOT_DIR:-$HOME/.cache/opendde}"
checkpoint="${OPENDDE_CHECKPOINT:-$opendde_root/checkpoint/opendde.pt}"
common_dir="${OPENDDE_COMMON_DIR:-$opendde_root/common}"
esm_repo="models--facebook--esm2_t33_650M_UR50D"
esm_source="$esm_cache/$esm_repo/snapshots/$ESM_REV"
if [[ ! -d "$esm_source" ]]; then
    esm_source="$esm_cache/hub/$esm_repo/snapshots/$ESM_REV"
fi
for source in "$mpnn_dir/solublempnn_v_48_020.pt" "$esm_source/config.json" "$esm_source/model.safetensors" "$esm_source/special_tokens_map.json" "$esm_source/tokenizer_config.json" "$esm_source/vocab.txt" "$checkpoint"; do
    [[ -f "$source" && -s "$source" ]] || { echo "Missing local model asset: $source. Set SOLUBLE_MPNN_WEIGHTS, ESM_CACHE and OPENDDE_CHECKPOINT; no weights will be downloaded." >&2; exit 2; }
done
common_names=(components.cif components.cif.rdkit_mol.pkl obsolete_to_successor.json release_date_cache.json)
for name in "${common_names[@]}"; do
    [[ -f "$common_dir/$name" && -s "$common_dir/$name" ]] || { echo "Missing common data: $common_dir/$name. Set OPENDDE_COMMON_DIR." >&2; exit 2; }
done
esm_destination="$destination/huggingface/$esm_repo/snapshots/$ESM_REV"
mkdir -p "$destination/soluble_mpnn" "$esm_destination" "$destination/huggingface/$esm_repo/refs" "$destination/checkpoint" "$destination/common"
cp -L "$mpnn_dir/solublempnn_v_48_020.pt" "$destination/soluble_mpnn/"
for name in config.json model.safetensors special_tokens_map.json tokenizer_config.json vocab.txt; do
    cp -L "$esm_source/$name" "$esm_destination/$name"
done
printf '%s' "$ESM_REV" > "$destination/huggingface/$esm_repo/refs/main"
cp "$script_dir/model-checksums.sha256" "$destination/SHA256SUMS"
(cd "$destination" && sha256sum --check SHA256SUMS)
cp -L "$checkpoint" "$destination/checkpoint/opendde.pt"
for name in "${common_names[@]}"; do
    cp -L "$common_dir/$name" "$destination/common/$name"
done
cp "$script_dir/ESM-LICENSE.txt" "$destination/ESM-LICENSE.txt"
(cd "$destination" && sha256sum checkpoint/opendde.pt common/* >> SHA256SUMS)
chmod -R a+rX "$destination"
echo "Prepared external OpenDDE, SolubleMPNN, ESM-2 650M and common data. Mount this directory read-only at /weights."
