import json
from typing import Dict, Optional, Union

import numpy as np

# TM-score calculation constants
TM_SCORE_POWER = 2.0
MIN_CHAIN_LENGTH = 27
NUCLEIC_ACID_MIN_D0 = 2.0
PROTEIN_MIN_D0 = 1.0
D0_SCALE_COEFFICIENT = 1.24
D0_LENGTH_OFFSET = 15
D0_POWER_EXPONENT = 1.0 / 3.0
D0_BASE_OFFSET = 1.8

# -----------------------------------------------------------------------------
# Helper Functions (Extracted from original script)
# -----------------------------------------------------------------------------


def load_first_json_object(json_path):
    """Load a JSON object, tolerating duplicated trailing bytes from rank races."""
    text = open(json_path, "r").read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        data, _ = json.JSONDecoder().raw_decode(text)
        return data


def ptm_func(x: Union[float, np.ndarray], d0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculates the TM-score term: 1 / (1 + (d/d0)^2)"""
    return 1.0 / (1 + (x / d0) ** TM_SCORE_POWER)


def calc_d0(length: int, pair_type: str) -> float:
    """Calculates d0 scaling factor based on sequence length."""
    length = float(length)
    if length < MIN_CHAIN_LENGTH:
        length = MIN_CHAIN_LENGTH
    min_value = PROTEIN_MIN_D0
    if pair_type == "nucleic_acid":
        min_value = NUCLEIC_ACID_MIN_D0
    d0 = D0_SCALE_COEFFICIENT * (length - D0_LENGTH_OFFSET) ** D0_POWER_EXPONENT - D0_BASE_OFFSET
    return max(min_value, d0)


def calc_d0_array(lengths: np.ndarray, pair_type: str) -> np.ndarray:
    """Vectorized version of calc_d0."""
    lengths = np.array(lengths, dtype=float)
    lengths = np.maximum(27, lengths)
    min_value = PROTEIN_MIN_D0
    if pair_type == "nucleic_acid":
        min_value = NUCLEIC_ACID_MIN_D0
    return np.maximum(
        min_value,
        D0_SCALE_COEFFICIENT * (lengths - D0_LENGTH_OFFSET) ** D0_POWER_EXPONENT - D0_BASE_OFFSET,
    )


def classify_chains(chains: np.ndarray, residue_types: np.ndarray) -> Dict[str, str]:
    """Classifies chains as 'protein' or 'nucleic_acid'."""
    nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
    chain_types = {}
    unique_chains = np.unique(chains)
    for chain in unique_chains:
        indices = np.where(chains == chain)[0]
        chain_residues = residue_types[indices]
        nuc_count = sum(residue in nuc_residue_set for residue in chain_residues)
        chain_types[chain] = "nucleic_acid" if nuc_count > 0 else "protein"
    return chain_types


def parse_pdb_atom_line(line: str) -> Optional[Dict]:
    """Parses a PDB ATOM/HETATM line."""
    try:
        atom_num = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21].strip()
        residue_seq_num = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        return {
            "atom_num": atom_num,
            "atom_name": atom_name,
            "residue_name": residue_name,
            "chain_id": chain_id,
            "residue_seq_num": residue_seq_num,
            "x": x,
            "y": y,
            "z": z,
        }
    except ValueError:
        return None


def parse_cif_atom_line(line: str, fielddict: Dict[str, int]) -> Optional[Dict]:
    """Parses an mmCIF ATOM/HETATM line using a field dictionary."""
    linelist = line.split()
    try:
        atom_num = linelist[fielddict["id"]]
        atom_name = linelist[fielddict["label_atom_id"]]
        residue_name = linelist[fielddict["label_comp_id"]]
        chain_id = linelist[fielddict["label_asym_id"]]
        residue_seq_num = linelist[fielddict["label_seq_id"]]
        x = linelist[fielddict["Cartn_x"]]
        y = linelist[fielddict["Cartn_y"]]
        z = linelist[fielddict["Cartn_z"]]

        if residue_seq_num == ".":
            return None  # ligand atom

        return {
            "atom_num": int(atom_num),
            "atom_name": atom_name,
            "residue_name": residue_name,
            "chain_id": chain_id,
            "residue_seq_num": int(residue_seq_num),
            "x": float(x),
            "y": float(y),
            "z": float(z),
        }
    except (ValueError, IndexError, KeyError):
        return None


# -----------------------------------------------------------------------------
# Main Wrapper Function
# -----------------------------------------------------------------------------
# dist_cutoff, pae_cutoff, confidence_json_path, output_cif_path
def _validate_pae_shape(shape, numres: int) -> None:
    """Require one PAE row and column for every parsed structure residue."""
    expected = (numres, numres)
    if tuple(shape) != expected:
        raise ValueError(f"PAE shape {tuple(shape)} does not match residues {expected}")


def _align_pae_to_residues(
    pae_matrix_raw: np.ndarray,
    numres: int,
    representative_atom_indices,
    atom_to_token_idx=None,
    atom_count: Optional[int] = None,
) -> np.ndarray:
    """Align token- or atom-level PAE to the parsed CA/C1 residue order."""
    pae_matrix_raw = np.asarray(pae_matrix_raw)
    if pae_matrix_raw.ndim != 2 or pae_matrix_raw.shape[0] != pae_matrix_raw.shape[1]:
        raise ValueError(f"PAE must be square, got {tuple(pae_matrix_raw.shape)}")
    if pae_matrix_raw.shape == (numres, numres):
        return pae_matrix_raw

    representative_atom_indices = np.asarray(representative_atom_indices, dtype=int)
    if len(representative_atom_indices) != numres:
        raise ValueError(
            f"Representative atom count does not match parsed residues: {len(representative_atom_indices)} != {numres}"
        )

    if atom_to_token_idx is not None:
        atom_to_token_idx = np.asarray(atom_to_token_idx, dtype=int)
        if atom_count is not None and len(atom_to_token_idx) != atom_count:
            raise ValueError(
                f"atom_to_token_idx length does not match structure atoms: {len(atom_to_token_idx)} != {atom_count}"
            )
        if representative_atom_indices.size and representative_atom_indices.max() >= len(atom_to_token_idx):
            raise ValueError("Representative atom index exceeds atom_to_token_idx")
        residue_tokens = atom_to_token_idx[representative_atom_indices]
        if len(np.unique(residue_tokens)) != numres:
            raise ValueError("Multiple parsed residues map to the same PAE token")
        if residue_tokens.size and (residue_tokens.min() < 0 or residue_tokens.max() >= pae_matrix_raw.shape[0]):
            raise ValueError("Residue token index exceeds the PAE matrix")
        return pae_matrix_raw[np.ix_(residue_tokens, residue_tokens)]

    if atom_count is not None and pae_matrix_raw.shape == (atom_count, atom_count):
        return pae_matrix_raw[np.ix_(representative_atom_indices, representative_atom_indices)]

    raise ValueError(
        f"Cannot align PAE shape {tuple(pae_matrix_raw.shape)} to {numres} residues and {atom_count} atoms"
    )


def compute_ipsae(
    dist_cutoff: float,
    pae_cutoff: float,
    confidence_json_path: str,
    output_cif_path: str,
) -> float:
    """
    Computes ipSAE scores with physical distance filtering.

    Updates:
    - Uses dist_cutoff to filter non-interacting pairs.
    - Uses dist_cutoff to count 'num_residues' (interface size).
    """

    confidence_json_path = str(confidence_json_path)
    output_cif_path = str(output_cif_path)

    # 1. Determine Format (AF2/AF3/Boltz1)
    af2, af3, boltz1, cif = False, False, False, False
    if ".pdb" in output_cif_path:
        af2 = True
    elif ".cif" in output_cif_path:
        cif = True
        if confidence_json_path.endswith(".json"):
            af3 = True
        elif confidence_json_path.endswith(".npz"):
            boltz1 = True

    if not (af2 or af3 or boltz1):
        raise ValueError(f"Unknown file combination: {output_cif_path}, {confidence_json_path}")

    # 2. Parse Structure File (PDB/CIF)
    residues = []
    chains = []
    representative_atom_indices = []
    atom_count = 0
    atomsitefield_dict = {}
    atomsitefield_num = 0
    # ... (Helper parsing functions assumed to be imported from previous context) ...
    # For brevity, reusing the parsing logic conceptually

    with open(output_cif_path) as structure_file:
        for line in structure_file:
            if cif and line.startswith("_atom_site."):
                parts = line.strip().split(".")
                if len(parts) > 1:
                    atomsitefield_dict[parts[1]] = atomsitefield_num
                    atomsitefield_num += 1

            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_index = atom_count
                atom_count += 1
                atom = parse_cif_atom_line(line, atomsitefield_dict) if cif else parse_pdb_atom_line(line)
                if atom is None:
                    continue

                # One representative atom per residue, preserving structure order.
                atom_name = atom["atom_name"].strip("\"'")
                is_representative = atom_name == "CA" or (
                    atom["residue_name"] in {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
                    and atom_name in {"C1", "C1*", "C1'"}
                )
                if is_representative:
                    representative_atom_indices.append(atom_index)
                    residues.append(
                        {
                            "coor": np.array([atom["x"], atom["y"], atom["z"]]),
                            "res": atom["residue_name"],
                            "chainid": atom["chain_id"],
                        }
                    )
                    chains.append(atom["chain_id"])

    # Convert to Arrays
    numres = len(residues)
    coordinates = np.array([res["coor"] for res in residues])
    chains = np.array(chains)
    unique_chains = np.unique(chains)
    residue_types = np.array([res["res"] for res in residues])

    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Classify Chains
    chain_types = classify_chains(chains, residue_types)
    chain_pair_type = {c1: {c2: "protein" for c2 in unique_chains if c1 != c2} for c1 in unique_chains}
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            if chain_types[c1] == "nucleic_acid" or chain_types[c2] == "nucleic_acid":
                chain_pair_type[c1][c2] = "nucleic_acid"

    # 3. Load PAE Matrix
    pae_matrix = None
    if af3:
        data = load_first_json_object(confidence_json_path)
        # AlphaFold3 uses 'pae', Protenix uses 'token_pair_pae'
        if "pae" in data:
            # AlphaFold3 official format
            pae_matrix_raw = np.array(data["pae"])
        elif "token_pair_pae" in data:
            # Protenix format
            pae_matrix_raw = np.array(data["token_pair_pae"])
        else:
            raise ValueError(f"PAE field not found in confidence JSON. Available keys: {list(data.keys())}")
        pae_matrix = _align_pae_to_residues(
            pae_matrix_raw,
            numres=numres,
            representative_atom_indices=representative_atom_indices,
            atom_to_token_idx=data.get("atom_to_token_idx"),
            atom_count=atom_count,
        )
    elif boltz1:
        data = np.load(confidence_json_path)
        pae_matrix = data["pae"]  # Assume already masked or check dimensions
    elif af2:
        data = load_first_json_object(confidence_json_path)
        pae_matrix = np.array(data["pae"]) if "pae" in data else np.array(data[0]["predicted_aligned_error"])

    if pae_matrix is None:
        raise ValueError("PAE load failed")
    if not af3:
        pae_matrix = _align_pae_to_residues(
            pae_matrix,
            numres=numres,
            representative_atom_indices=representative_atom_indices,
            atom_count=atom_count,
        )
    _validate_pae_shape(pae_matrix.shape, numres)

    # 4. Compute Scores
    results = {}
    ptm_func_vec = np.vectorize(ptm_func)

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            # --- Masks ---
            mask_c1 = chains == chain1
            mask_c2 = chains == chain2

            # --- 1. Physical Distance Check (Using dist_cutoff) ---
            sub_dist = distances[np.ix_(mask_c1, mask_c2)]

            if sub_dist.size == 0:
                continue

            min_dist = np.min(sub_dist)

            # If the closest atoms are further than cutoff, no physical interaction exists
            # We return a result with 0 scores but correct metadata
            if min_dist > dist_cutoff:
                results[(chain1, chain2)] = {"ipSAE": 0.0, "ipTM": 0.0, "num_residues": 0, "min_dist": float(min_dist)}
                continue

            # Calculate Interface Residues (Chain 1 residues within dist_cutoff of Chain 2)
            # axis=1 min gives the shortest distance from a residue in C1 to any residue in C2
            interface_mask = np.min(sub_dist, axis=1) < dist_cutoff
            num_interface_residues = int(np.sum(interface_mask))

            # --- 2. PAE Calculations ---
            sub_pae = pae_matrix[np.ix_(mask_c1, mask_c2)]

            # Metadata
            n0chn = np.sum(mask_c1) + np.sum(mask_c2)
            d0_chn_val = calc_d0(n0chn, chain_pair_type[chain1][chain2])

            # ipSAE Logic
            valid_interactions = sub_pae < pae_cutoff
            n0res_counts = np.sum(valid_interactions, axis=1)
            d0res_vals = calc_d0_array(n0res_counts, chain_pair_type[chain1][chain2])

            row_scores = []
            for r_idx in range(sub_pae.shape[0]):
                if n0res_counts[r_idx] > 0:
                    valid_indices = valid_interactions[r_idx]
                    pae_vals = sub_pae[r_idx][valid_indices]
                    scores = ptm_func_vec(pae_vals, d0res_vals[r_idx])
                    row_scores.append(scores.mean())
                else:
                    row_scores.append(0.0)

            ipsae_d0res_val = np.max(row_scores) if row_scores else 0.0

            # ipTM Logic
            ptm_matrix_chn = ptm_func_vec(sub_pae, d0_chn_val)
            iptm_row_means = [ptm_matrix_chn[r_idx, :].mean() for r_idx in range(sub_pae.shape[0])]
            iptm_val = np.max(iptm_row_means) if iptm_row_means else 0.0

            # Store Results
            results[(chain1, chain2)] = {
                "ipSAE": float(ipsae_d0res_val),
                "ipTM": float(iptm_val),
                "num_residues": num_interface_residues,  # Now calculated using dist_cutoff
                "min_dist": float(min_dist),  # Useful debugging metric
                "n0chn": int(n0chn),
            }

    # max_ipsae: best binding quality between any chain pair
    # min_ipsae: worst binding quality between any chain pair
    all_ipsae_values = []
    for (chain1, chain2), data in results.items():
        if isinstance(chain1, str):  # Skip non-tuple keys
            all_ipsae_values.append(data["ipSAE"])

    if all_ipsae_values:
        results["max_ipsae"] = float(max(all_ipsae_values))
        results["min_ipsae"] = float(min(all_ipsae_values))
    else:
        results["max_ipsae"] = 0.0
        results["min_ipsae"] = 0.0

    return results["max_ipsae"]
