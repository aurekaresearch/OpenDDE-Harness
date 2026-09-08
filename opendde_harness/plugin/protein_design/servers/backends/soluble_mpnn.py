"""SolubleMPNN inference using the official LigandMPNN PyTorch implementation."""

from __future__ import annotations

import hashlib
import math
import os
import secrets
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "solublempnn_v_48_020"
WEIGHTS_URL = f"https://files.ipd.uw.edu/pub/ligandmpnn/{DEFAULT_MODEL}.pt"
WEIGHTS_SHA256 = "7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a"
_SAMPLING_LOCK = threading.Lock()


def resolve_soluble_mpnn_weights_path(
    weights_path: str | None = None,
    *,
    download_if_missing: bool = True,
) -> Path:
    supplied = weights_path or os.environ.get("OPENDDE_HARNESS_PROTEIN_SOLUBLE_MPNN_WEIGHTS_PATH")
    if supplied:
        path = Path(supplied).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SolubleMPNN checkpoint not found: {path}")
        return path.resolve()
    directory = Path(
        os.environ.get(
            "OPENDDE_HARNESS_PROTEIN_SOLUBLE_MPNN_WEIGHTS_DIR",
            str(
                Path(os.environ.get("OPENDDE_HARNESS_PROJECT_ROOT", Path(__file__).resolve().parents[5]))
                / "external/ligandmpnn/model_params"
            ),
        )
    ).expanduser()
    path = directory / f"{DEFAULT_MODEL}.pt"
    if not path.is_file() and download_if_missing:
        _download_weights(path)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError(f"SolubleMPNN checkpoint checksum mismatch: {path}")
    return path


def _download_weights(destination: Path) -> None:
    download_url = os.environ.get("OPENDDE_HARNESS_PROTEIN_SOLUBLE_MPNN_WEIGHTS_URL") or WEIGHTS_URL
    if download_url != WEIGHTS_URL and not download_url.startswith("https://"):
        raise ValueError("The SolubleMPNN weights mirror must use HTTPS")
    budget = float(os.environ.get("OPENDDE_HARNESS_PROTEIN_SOLUBLE_MPNN_DOWNLOAD_TIMEOUT", "1800"))
    if not math.isfinite(budget) or not 0 < budget <= 3600:
        raise ValueError("SolubleMPNN download timeout must be between 0 and 3600 seconds")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        deadline = time.monotonic() + budget
        for attempt in range(1, 4):
            print(f"Downloading SolubleMPNN weights (attempt {attempt}/3): {destination}", file=sys.stderr, flush=True)
            downloaded = 0
            reported = 0
            reported_at = time.monotonic()
            try:
                with (
                    urllib.request.urlopen(download_url, timeout=min(30, budget)) as response,
                    temporary.open("wb") as output,
                ):
                    while chunk := response.read(64 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if downloaded - reported >= 1024 * 1024 or now - reported_at >= 5:
                            print(f"  received {downloaded / 1024 / 1024:.2f} MiB", file=sys.stderr, flush=True)
                            reported = downloaded
                            reported_at = now
                        if now >= deadline:
                            raise TimeoutError("SolubleMPNN download exceeded its time budget")
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500 and exc.code not in (408, 429):
                    raise
                if attempt == 3 or time.monotonic() >= deadline:
                    raise RuntimeError(
                        "SolubleMPNN download failed or stalled. Check access to files.ipd.uw.edu, "
                        "or prepare the checkpoint on a local disk and set "
                        "OPENDDE_HARNESS_PROTEIN_SOLUBLE_MPNN_WEIGHTS_PATH."
                    ) from exc
                print("  connection interrupted; retrying", file=sys.stderr, flush=True)
                time.sleep(attempt)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != WEIGHTS_SHA256:
            raise ValueError("SolubleMPNN checkpoint checksum mismatch")
        import torch

        checkpoint = torch.load(temporary, map_location="cpu", weights_only=True)
        if (
            not isinstance(checkpoint, dict)
            or not checkpoint.get("model_state_dict")
            or not checkpoint.get("num_edges")
        ):
            raise ValueError("Invalid SolubleMPNN checkpoint")
        temporary.chmod(0o644)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class SolubleMPNNClient:
    def __init__(
        self,
        weights_path: str | None = None,
        backbone_noise: float = 0.0,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        import torch

        from external.ligandmpnn.model_utils import ProteinMPNN
        from opendde_harness.cli.compute_environment import resolve_device

        device = resolve_device(device)
        if not math.isfinite(backbone_noise) or backbone_noise < 0:
            raise ValueError("SolubleMPNN backbone_noise must be finite and nonnegative")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("SolubleMPNN seed must be a nonnegative integer")
        self.device = torch.device(device)
        self.seed = seed
        self.weights_path = resolve_soluble_mpnn_weights_path(weights_path)
        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        self.model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=int(checkpoint["num_edges"]),
            augment_eps=backbone_noise,
            device=self.device,
            atom_context_num=1,
            model_type="soluble_mpnn",
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.weights_sha256 = hashlib.sha256(self.weights_path.read_bytes()).hexdigest()

    def design(
        self,
        pdb_path: str,
        chain: str,
        designable_positions: list[int],
        expected_length: int,
        omit_aas: str = "",
        bias_aas: dict[str, float] | None = None,
        bias_by_res: dict[int, dict[str, float]] | None = None,
        forced_residues: dict[int, str] | None = None,
        temperature: float = 0.1,
        num_sequences: int = 10,
    ) -> list[dict[str, Any]]:
        import torch

        from external.ligandmpnn.data_utils import (
            alphabet,
            featurize,
            get_score,
            get_seq_rec,
            parse_PDB,
        )

        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("SolubleMPNN temperature must be finite and positive")
        if isinstance(num_sequences, bool) or not isinstance(num_sequences, int) or not 1 <= num_sequences <= 256:
            raise ValueError("SolubleMPNN num_sequences must be between 1 and 256")
        if not designable_positions or any(
            isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < expected_length for p in designable_positions
        ):
            raise ValueError("SolubleMPNN requires valid zero-based designable_positions")
        if set(omit_aas) - set(alphabet[:-1]) or set(omit_aas) >= set(alphabet[:-1]):
            raise ValueError("SolubleMPNN omit_aas must leave at least one canonical amino acid")

        with tempfile.TemporaryDirectory(prefix="soluble-mpnn-") as temporary:
            source = Path(pdb_path)
            if source.suffix.lower() in {".cif", ".mmcif"}:
                from biotite.structure.io.pdb import PDBFile

                from .structure_contacts import load_structure_model

                pdb = PDBFile()
                pdb.set_structure(load_structure_model(source))
                source = Path(temporary) / "input.pdb"
                pdb.write(source)
            protein, *_ = parse_PDB(str(source), device=self.device)

        chain_indices = np.flatnonzero(protein["chain_letters"] == chain).tolist()
        if len(chain_indices) != expected_length:
            raise ValueError(f"SolubleMPNN chain {chain} length mismatch: {len(chain_indices)} != {expected_length}")
        selected = [chain_indices[p] for p in sorted(set(designable_positions))]
        if not bool(torch.all(protein["mask"][selected])):
            raise ValueError("SolubleMPNN design residues require complete N/CA/C/O backbone atoms")
        protein["chain_mask"] = torch.zeros_like(protein["mask"])
        protein["chain_mask"][selected] = 1
        features = featurize(protein, model_type="soluble_mpnn")
        length = len(protein["S"])
        bias = torch.zeros((1, length, 21), device=self.device)
        for aa, value in (bias_aas or {}).items():
            if aa not in alphabet[:-1] or not math.isfinite(value):
                raise ValueError("Invalid SolubleMPNN amino-acid bias")
            bias[:, :, alphabet.index(aa)] = value
        for position, values in (bias_by_res or {}).items():
            if not 1 <= int(position) <= expected_length:
                raise ValueError("SolubleMPNN residue bias position is out of range")
            for aa, value in values.items():
                if aa not in alphabet[:-1] or not math.isfinite(value):
                    raise ValueError("Invalid SolubleMPNN residue bias")
                bias[0, chain_indices[int(position) - 1], alphabet.index(aa)] += value
        for aa in set(omit_aas + "X"):
            bias[:, :, alphabet.index(aa)] = -1e8
        for position, aa in (forced_residues or {}).items():
            if position not in designable_positions or aa not in alphabet[:-1] or aa in omit_aas:
                raise ValueError("SolubleMPNN forced residue conflicts with design constraints")
            index = chain_indices[position]
            bias[0, index, :] = -1e8
            bias[0, index, alphabet.index(aa)] = 0.0
        features.update(
            batch_size=num_sequences,
            temperature=temperature,
            bias=bias,
            symmetry_residues=[[]],
            symmetry_weights=[[]],
        )
        cuda_devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        # Sampling and backbone noise use upstream global RNGs; isolate concurrent requests.
        with _SAMPLING_LOCK, torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            seed = self.seed if self.seed is not None else secrets.randbits(32)
            torch.random.default_generator.manual_seed(seed)
            for device_index in cuda_devices:
                torch.cuda.default_generators[device_index].manual_seed(seed)
            features["randn"] = torch.randn((num_sequences, length), device=self.device)
            result = self.model.sample(features)
            mask = features["mask"] * features["chain_mask"]
            scores, _ = get_score(result["S"], result["log_probs"], mask)
            recovery = get_seq_rec(features["S"], result["S"], mask)

        fixed = protein["chain_mask"] == 0
        if not bool(torch.all(result["S"][:, fixed] == protein["S"][fixed])):
            raise ValueError("SolubleMPNN changed a fixed residue")
        designs = []
        for index, sampled in enumerate(result["S"]):
            sequence = "".join(alphabet[int(aa)] for aa in sampled[chain_indices])
            if any(sequence[p] != aa for p, aa in (forced_residues or {}).items()):
                raise ValueError("SolubleMPNN failed to preserve a forced anchor")
            score = float(scores[index])
            if not math.isfinite(score) or set(sequence) - set(alphabet[:-1]):
                raise ValueError("SolubleMPNN returned an invalid sequence or score")
            designs.append(
                {
                    "sequence": sequence,
                    "mpnn_score": score,
                    "mpnn_confidence": math.exp(-score),
                    "mpnn_seqid": float(recovery[index]),
                }
            )
        return sorted(designs, key=lambda item: item["mpnn_score"])
