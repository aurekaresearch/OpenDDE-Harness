"""
ESM-based protein language model tools for OpenDDE Harness.

Provides tools for:
- Deep Mutational Scanning (DMS)
- Pseudo Log Likelihood (PLL) calculation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA operations (must be set before torch import)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from opendde_harness.cli.compute_environment import resolve_device

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODEL_NAME: str = os.getenv("ESM_MODEL_NAME") or "facebook/esm2_t12_35M_UR50D"
DEFAULT_CACHE_DIR: str = os.getenv("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_SCORING_MODE: str = "probability"

# Amino acid alphabet
AMINO_ACID_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"

# Hotspot detection thresholds
HOTSPOT_RATIO_THRESHOLD: float = 1.5
BENEFICIAL_RATIO_THRESHOLD: float = 1.05
MIN_MUTATION_SCORE: float = 0.0001


# ============================================================================
# Helper Functions
# ============================================================================


def validate_sequence(sequence: str) -> bool:
    """
    Validate that a sequence contains only valid amino acids.

    Args:
        sequence: Amino acid sequence string

    Returns:
        True if valid, False otherwise
    """
    cleaned = sequence.replace(" ", "").replace("\n", "").upper()
    return all(aa in AMINO_ACID_ALPHABET for aa in cleaned)


def clean_sequence(sequence: str) -> str:
    """
    Clean a sequence by removing whitespace and converting to uppercase.

    Args:
        sequence: Raw sequence string

    Returns:
        Cleaned sequence
    """
    return sequence.replace(" ", "").replace("\n", "").replace("\r", "").upper()


# ============================================================================
# ESMClient Class
# ============================================================================


class ESMClient:
    """
    ESM protein language model client for sequence analysis.

    Provides methods for:
    - Pseudo Log Likelihood (PLL) calculation
    - Deep Mutational Scanning (DMS)

    Usage:
        client = ESMClient(device="cuda:0")
        pll = client.calculate_pseudo_log_likelihood("MKVLWA")
        dms = client.calculate_deep_mutation_scan("MKVLWA")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: str = DEFAULT_CACHE_DIR,
        offline: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize ESM client.

        Args:
            model_name: HuggingFace model name
            cache_dir: Directory containing the downloaded model
            offline: If True, force offline mode (no network requests)
            device: Device to use (e.g., "cuda:0", "cuda:1", "cpu").
                   If None, uses environment variables or auto-detect.
        """
        # Set deterministic CUDA workspace config
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Configure offline mode
        if offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

        # Expand cache directory path
        cache_dir = os.path.expanduser(cache_dir)

        # Determine device
        self.device = resolve_device(device or os.environ.get("ESM_DEVICE"))

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=offline)
        self.model = EsmForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=offline).to(
            self.device
        )
        self.model.eval()

        # Create amino acid to ID mapping
        self.aa_to_id: Dict[str, int] = {aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACID_ALPHABET}

    def _empty_cuda_cache(self) -> None:
        """Release cached CUDA blocks after memory-heavy ESM forwards."""
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def calculate_pseudo_log_likelihood(self, sequence: str, batch_size: Optional[int] = None) -> float:
        """
        Calculate pseudo log likelihood for a protein sequence.

        Higher (less negative) values indicate sequences that the model
        considers more 'natural' or evolutionarily plausible.

        Args:
            sequence: The amino acid sequence to analyze
            batch_size: Batch size for computation (default: all positions)

        Returns:
            The PLL score as a float (negative mean NLL)

        Raises:
            ValueError: If sequence is empty or contains invalid characters
        """
        # Validate and clean sequence
        sequence = clean_sequence(sequence)
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        if not validate_sequence(sequence):
            raise ValueError("Sequence contains invalid amino acids")

        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))[0]

        # Identify residue positions (exclude special tokens and X/unknown)
        special_ids = set(self.tokenizer.all_special_ids)
        x_token_id = self.tokenizer.convert_tokens_to_ids("X")
        unk_token_id = self.tokenizer.unk_token_id
        residue_positions = [
            i
            for i in range(input_ids.shape[0])
            if attention_mask[i].item() == 1
            and int(input_ids[i].item()) not in special_ids
            and int(input_ids[i].item()) not in (x_token_id, unk_token_id)
        ]

        if not residue_positions:
            return 0.0

        # Get mask token ID
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("Tokenizer does not define a mask_token_id")

        # Calculate NLL in chunks
        total_nll_sum = 0.0
        total_count = 0
        chunk_size = batch_size if batch_size is not None else len(residue_positions)

        with torch.no_grad():
            for start in range(0, len(residue_positions), chunk_size):
                chunk_positions = residue_positions[start : start + chunk_size]
                pos = torch.tensor(chunk_positions, device=self.device, dtype=torch.long)
                bsz = pos.shape[0]

                # Create masked input
                masked_ids = input_ids.unsqueeze(0).repeat(bsz, 1).clone()
                masked_ids[torch.arange(bsz, device=self.device), pos] = mask_token_id
                attn = attention_mask.unsqueeze(0).repeat(bsz, 1)

                # Get logits
                logits = self.model(input_ids=masked_ids, attention_mask=attn).logits

                # Mask special tokens
                if self.tokenizer.all_special_ids:
                    logits[:, :, self.tokenizer.all_special_ids] = -float("inf")

                # Extract masked position logits
                masked_logits = logits[torch.arange(bsz, device=self.device), pos, :]
                labels = input_ids[pos]

                # Calculate NLL
                nll_sum = torch.nn.functional.cross_entropy(masked_logits, labels, reduction="sum").item()

                total_nll_sum += float(nll_sum)
                total_count += int(bsz)
                del pos, masked_ids, attn, logits, masked_logits, labels

        mean_nll = total_nll_sum / max(total_count, 1)
        self._empty_cuda_cache()
        return -mean_nll

    def calculate_pseudo_log_likelihood_batch(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """Calculate pseudo log likelihoods for several sequences together.

        Each masked residue is an independent row in a shared forward batch.
        This preserves the single-sequence PLL definition while reducing the
        number of model invocations for a candidate library.
        """
        if not sequences:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        cleaned_sequences = [clean_sequence(sequence) for sequence in sequences]
        for sequence in cleaned_sequences:
            if not sequence:
                raise ValueError("Sequence cannot be empty")
            if not validate_sequence(sequence):
                raise ValueError("Sequence contains invalid amino acids")

        inputs = self.tokenizer(
            cleaned_sequences,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        special_ids = set(self.tokenizer.all_special_ids)
        x_token_id = self.tokenizer.convert_tokens_to_ids("X")
        unk_token_id = self.tokenizer.unk_token_id
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("Tokenizer does not define a mask_token_id")

        owners: List[int] = []
        positions: List[int] = []
        for sequence_index in range(input_ids.shape[0]):
            for position in range(input_ids.shape[1]):
                token_id = int(input_ids[sequence_index, position].item())
                if (
                    attention_mask[sequence_index, position].item() == 1
                    and token_id not in special_ids
                    and token_id not in (x_token_id, unk_token_id)
                ):
                    owners.append(sequence_index)
                    positions.append(position)

        total_nll = [0.0 for _ in cleaned_sequences]
        total_count = [0 for _ in cleaned_sequences]
        with torch.no_grad():
            for start in range(0, len(owners), batch_size):
                owner_batch = torch.tensor(
                    owners[start : start + batch_size],
                    device=self.device,
                    dtype=torch.long,
                )
                position_batch = torch.tensor(
                    positions[start : start + batch_size],
                    device=self.device,
                    dtype=torch.long,
                )
                masked_ids = input_ids.index_select(0, owner_batch).clone()
                masked_ids[
                    torch.arange(masked_ids.shape[0], device=self.device),
                    position_batch,
                ] = mask_token_id
                masked_attention = attention_mask.index_select(0, owner_batch)
                logits = self.model(
                    input_ids=masked_ids,
                    attention_mask=masked_attention,
                ).logits
                if self.tokenizer.all_special_ids:
                    logits[:, :, self.tokenizer.all_special_ids] = -float("inf")
                masked_logits = logits[
                    torch.arange(masked_ids.shape[0], device=self.device),
                    position_batch,
                    :,
                ]
                labels = input_ids[owner_batch, position_batch]
                nll_values = (
                    torch.nn.functional.cross_entropy(
                        masked_logits,
                        labels,
                        reduction="none",
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                for owner, nll in zip(owner_batch.detach().cpu().tolist(), nll_values):
                    total_nll[owner] += float(nll)
                    total_count[owner] += 1
                del (
                    owner_batch,
                    position_batch,
                    masked_ids,
                    masked_attention,
                    logits,
                    masked_logits,
                    labels,
                    nll_values,
                )

        scores = [-nll_sum / max(count, 1) for nll_sum, count in zip(total_nll, total_count)]
        self._empty_cuda_cache()
        return scores

    def calculate_deep_mutation_scan(
        self,
        sequence: str,
        batch_size: Optional[int] = None,
        scoring_mode: str = DEFAULT_SCORING_MODE,
    ) -> List[Dict]:
        """
        Perform deep mutational scan on a protein sequence.

        Predicts the effect of all possible single-point mutations across
        the entire sequence.

        Args:
            sequence: The amino acid sequence to analyze
            batch_size: Batch size for computation
            scoring_mode: "probability" for raw scores, "llr" for log-likelihood ratio

        Returns:
            List of dictionaries, one per position, with mutation scores

        Raises:
            ValueError: If sequence is invalid
        """
        # Validate and clean sequence
        sequence = clean_sequence(sequence)
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        if not validate_sequence(sequence):
            raise ValueError("Sequence contains invalid amino acids")

        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))[0]

        # Identify residue positions
        special_ids = set(self.tokenizer.all_special_ids)
        residue_positions = [
            i
            for i in range(input_ids.shape[0])
            if attention_mask[i].item() == 1 and int(input_ids[i].item()) not in special_ids
        ]

        mask_token_id = self.tokenizer.mask_token_id
        results = []

        # Process in chunks
        chunk_size = batch_size if batch_size is not None else len(residue_positions)

        with torch.no_grad():
            for i in range(0, len(residue_positions), chunk_size):
                chunk_positions = residue_positions[i : i + chunk_size]
                chunk_len = len(chunk_positions)

                # Create batch with masked positions
                batch_input_ids = input_ids.unsqueeze(0).repeat(chunk_len, 1)
                for batch_idx, pos_idx in enumerate(chunk_positions):
                    batch_input_ids[batch_idx, pos_idx] = mask_token_id

                batch_attention_mask = attention_mask.unsqueeze(0).repeat(chunk_len, 1)
                logits = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits

                # Convert to probabilities or log-probabilities
                if scoring_mode == "probability":
                    probs = torch.softmax(logits, dim=-1)
                else:
                    probs = torch.log_softmax(logits, dim=-1)

                # Extract scores for each position
                for batch_idx, pos_idx in enumerate(chunk_positions):
                    seq_idx = i + batch_idx
                    if seq_idx >= len(sequence):
                        break

                    wt_aa = sequence[seq_idx]
                    pos_num = seq_idx + 1

                    current_pos_probs = probs[batch_idx, pos_idx]
                    wt_token_id = input_ids[pos_idx].item()
                    wt_val = current_pos_probs[wt_token_id].item()

                    # Build result row
                    row_data: Dict = {"Pos": pos_num, "WT": wt_aa}
                    if scoring_mode == "probability":
                        row_data["WT_Score"] = round(wt_val, 4)

                    # Add scores for all amino acids
                    for aa in AMINO_ACID_ALPHABET:
                        aa_id = self.aa_to_id[aa]
                        val = current_pos_probs[aa_id].item()
                        score = val if scoring_mode == "probability" else val - wt_val
                        row_data[aa] = round(score, 4)

                    results.append(row_data)

        return results
