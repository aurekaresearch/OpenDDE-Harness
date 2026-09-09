---
name: epitope-analysis
description: |
  Fast epitope identification and hotspot coverage analysis using distance-based contact
  detection. Identifies which antigen residues are contacted by the antibody, calculates
  hotspot coverage, and analyzes per-CDR contributions. CDR positions always come from
  the design YAML; the skill never guesses IMGT/PDB residue ranges. No PLIP or Docker dependency.
---

# Epitope Analysis Skill

## Overview

Rapidly identifies epitope residues and analyzes antibody-antigen contacts using simple distance-based calculations. Provides hotspot coverage metrics and CDR contribution analysis.

**What it does:**
- Identifies epitope residues (antigen residues within 4.5Å of antibody)
- Calculates hotspot coverage (if known hotspots provided)
- Analyzes which CDRs contact which epitope regions
- Reports per-residue contact counts

**Key advantage:** Fast (~0.5-2s), no Docker, lightweight dependencies (biotite + scipy)

## Usage

### Single Structure Analysis

```bash
python3 -m opendde_harness.plugin.protein_design.servers.backends.epitope_analysis \
    --config_yaml configs/design.yaml \
    --binder_name seed_name \
    --structure_file outputs/structures/candidate_042_complex.pdb \
    --antibody_chains H,L \
    --antigen_chains A \
    --output epitope_042.json
```

### With Known Hotspots

```bash
python3 -m opendde_harness.plugin.protein_design.servers.backends.epitope_analysis \
    --config_yaml configs/design.yaml \
    --binder_name seed_name \
    --structure_file outputs/structures/candidate_042_complex.pdb \
    --antibody_chains H,L \
    --antigen_chains A \
    --hotspots A:45,A:52,A:67,A:89,A:102 \
    --output epitope_042.json
```

### Batch Processing

```bash
python3 -m opendde_harness.plugin.protein_design.servers.backends.epitope_batch \
    --config_yaml configs/design.yaml \
    --binder_name seed_name \
    --structure_dir outputs/structures/ \
    --antibody_chains H,L \
    --antigen_chains A \
    --output_csv epitope_results.csv
```

## Output Format

### JSON Output

```json
{
  "structure": "candidate_042_complex.pdb",
  "antibody_chains": ["H", "L"],
  "antigen_chains": ["A"],
  "epitope_residues": [
    {"chain": "A", "residue_id": 45, "residue_name": "TYR", "contacts": 8},
    {"chain": "A", "residue_id": 52, "residue_name": "ASP", "contacts": 5},
    {"chain": "A", "residue_id": 67, "residue_name": "GLU", "contacts": 12}
  ],
  "epitope_size": 18,
  "total_contacts": 127,
  "hotspot_analysis": {
    "provided_hotspots": 5,
    "contacted_hotspots": 4,
    "hotspot_coverage": 0.80,
    "missed_hotspots": ["A:102"]
  },
  "cdr_contributions": {
    "CDR1_H": {"residues_contacted": 5, "total_contacts": 23},
    "CDR2_H": {"residues_contacted": 7, "total_contacts": 38},
    "CDR3_H": {"residues_contacted": 12, "total_contacts": 51},
    "CDR1_L": {"residues_contacted": 2, "total_contacts": 8},
    "CDR2_L": {"residues_contacted": 1, "total_contacts": 3},
    "CDR3_L": {"residues_contacted": 3, "total_contacts": 4}
  }
}
```

### Text Output

```
=== Epitope Analysis ===

Structure: candidate_042_complex.pdb
Antibody chains: H, L
Antigen chains: A

Epitope Summary:
  Total epitope residues: 18
  Total contacts: 127
  Average contacts per residue: 7.1

Hotspot Coverage:
  Provided hotspots: 5
  Contacted hotspots: 4
  Coverage: 80.0%
  Missed hotspots: A:102

Top Contacted Residues:
  A:67  GLU  12 contacts
  A:75  ARG  11 contacts
  A:45  TYR   8 contacts
  A:52  ASP   5 contacts

CDR Contributions:
  CDR3_H: 12 residues, 51 contacts (40.2%)
  CDR2_H:  7 residues, 38 contacts (29.9%)
  CDR1_H:  5 residues, 23 contacts (18.1%)
  CDR3_L:  3 residues,  4 contacts (3.1%)
  CDR1_L:  2 residues,  8 contacts (6.3%)
  CDR2_L:  1 residue,   3 contacts (2.4%)
```

## Parameters

### Distance Cutoff

Default: 4.5Å (standard for antibody-antigen contacts)

```bash
# Custom cutoff
python3 -m opendde_harness.plugin.protein_design.servers.backends.epitope_analysis \
    --config_yaml configs/design.yaml \
    --binder_name seed_name \
    --structure_file structure.pdb \
    --antibody_chains H,L \
    --antigen_chains A \
    --cutoff 5.0  # More permissive
```

### CDR Definition

CDR positions are required from `initial_binders[].chains` in the design YAML.
Use `cdr_regions` or `designable_residues` directly, or define framework
`fixed_residues`; the complement is treated as the CDR/designable positions.
All YAML positions are 0-based sequence indices and are mapped to structure
residues by chain-local residue order, not by PDB/mmCIF residue number.

```bash
python3 -m opendde_harness.plugin.protein_design.servers.backends.epitope_analysis \
    --config_yaml configs/design.yaml \
    --binder_name seed_name \
    --structure_file structure.pdb \
    --antibody_chains H,L \
    --antigen_chains A
```

`--binder_name` may be omitted when the YAML has one initial binder, or when all
initial binders have identical CDR configurations. Otherwise it is required.

### Atom Selection

Default: Heavy atoms only (C, N, O, S)

```bash
# Include all atoms
python3 -m opendde_harness.plugin.protein_design.servers.backends.epitope_analysis \
    --config_yaml configs/design.yaml \
    --binder_name seed_name \
    --structure_file structure.pdb \
    --antibody_chains H,L \
    --antigen_chains A \
    --include_hydrogen
```

## Integration with Workflow

### After Structure Prediction

```python
from opendde_harness.plugin.protein_design.servers.backends.epitope_analysis import (
    analyze_epitope,
    load_cdr_regions_from_yaml,
)

# After folding
structure_file = f"outputs/{run_id}/structures/{candidate.name}_complex.pdb"

epitope_result = analyze_epitope(
    structure_file=structure_file,
    antibody_chains=["H", "L"],
    antigen_chains=["A"],
    cdr_regions=load_cdr_regions_from_yaml("configs/design.yaml", "seed_name"),
    hotspots=["A:45", "A:52", "A:67", "A:89", "A:102"],
)

# Check hotspot coverage
if epitope_result["hotspot_analysis"]["hotspot_coverage"] < 0.6:
    print(f"⚠️  Low hotspot coverage: {epitope_result['hotspot_analysis']['hotspot_coverage']:.1%}")
```

### With ReflectAgent

Agent can use this skill to:
- Verify CDR3_H is contributing to binding (should be dominant)
- Check if known hotspots are being contacted
- Compare epitope coverage across variants

```python
# Agent calls this skill
epitope_result = run_epitope_analysis(structure_file)

# Agent interprets results
if epitope_result["cdr_contributions"]["CDR3_H"]["total_contacts"] < 30:
    # Flag: CDR3_H should be the major contributor
    warning = "CDR3_H shows weak binding contribution"
```

## Algorithm Details

### Contact Detection

```python
# Build one antigen cKDTree, then query neighboring antigen atoms for each
# antibody atom within the configured cutoff. Each close atom pair is counted.
antigen_tree = cKDTree(antigen_heavy_atoms.coord)
neighbors = antigen_tree.query_ball_point(antibody_heavy_atoms.coord, cutoff)
```

### Hotspot Coverage

```python
contacted_hotspots = 0
for hotspot in provided_hotspots:
    if hotspot in epitope_residues:
        contacted_hotspots += 1

coverage = contacted_hotspots / len(provided_hotspots)
```

### CDR Assignment

Uses only CDR/designable positions from the YAML binder chain configuration.
Configured 0-based sequence positions are mapped onto each structure chain in
residue order, so non-contiguous or non-1-based structure numbering is supported.

## Performance

- **Speed**: 0.5-2 seconds per structure
- **Memory**: O(antigen atoms + reported contact pairs); no dense atom-pair matrix
- **Dependencies**: biotite, numpy, scipy
- **No Docker**: Pure Python implementation

## Comparison with structure-analysis (PLIP)

| Feature | epitope-analysis | structure-analysis (PLIP) |
|---------|-----------------|---------------------------|
| **Speed** | 0.5-2s | 5-10s |
| **Dependencies** | biotite + scipy | PLIP in configured |
| **Interaction types** | Distance-based | H-bonds, hydrophobic, pi-stacking, salt bridges |
| **Epitope ID** | ✅ Fast | ✅ Detailed |
| **Hotspot coverage** | ✅ | ❌ (need to parse PLIP output) |
| **CDR contributions** | ✅ | ⚠️ (manual calculation) |
| **Use case** | Quick screening | Detailed analysis |

**When to use epitope-analysis:**
- Need fast epitope identification
- Analyzing many structures (batch)
- Only need contact-based epitope definition
- Want CDR contribution breakdown

**When to use structure-analysis (PLIP):**
- Need detailed interaction types
- Analyzing specific binding mechanisms
- Want salt bridge / H-bond information

## Advanced Usage

### Custom Hotspot Definitions

```python
# Define hotspots per antigen chain
hotspots = {
    "A": [45, 52, 67, 89, 102],
    "B": [23, 34, 56],
}

result = analyze_epitope(
    structure_file="complex.pdb",
    antibody_chains=["H", "L"],
    antigen_chains=["A", "B"],
    cdr_regions=load_cdr_regions_from_yaml("configs/design.yaml", "seed_name"),
    hotspots=hotspots,
)
```

## Troubleshooting

**Issue**: "No epitope residues found"
- Check antibody/antigen chain IDs are correct
- Verify structure contains both antibody and antigen
- Try increasing cutoff (default 4.5Å)

**Issue**: "CDR configuration unavailable"
- Ensure the selected YAML binder chain defines `cdr_regions`,
  `designable_residues`, or framework `fixed_residues`
- Use `--binder_name` when initial binders have different CDR configurations

**Issue**: "Hotspot not found in structure"
- Check the target structure residue numbering used by the hotspot definition
- Verify hotspot notation: "A:45" means chain A, residue 45

**Issue**: "Different results from PLIP"
- Expected: Distance-based vs interaction-type-based
- PLIP is more selective (only specific interactions)
- This skill counts all close contacts

## Example Workflow

```bash
# 1. Predict structure (already done in your workflow)
# outputs/structures/candidate_042_complex.pdb

# 2. Run epitope analysis
python3 -m opendde_harness.plugin.protein_design.servers.backends.epitope_analysis \
    --config_yaml configs/design.yaml \
    --binder_name seed_name \
    --structure_file outputs/structures/candidate_042_complex.pdb \
    --antibody_chains H,L \
    --antigen_chains A \
    --hotspots A:45,A:52,A:67 \
    --output epitope_042.json

# 3. Check results
cat epitope_042.json | jq '.hotspot_analysis.hotspot_coverage'
# Output: 1.0 (100% coverage)

```

## Related Skills

- **structure-analysis**: Detailed PLIP-based interaction profiling
- **evo-reflect-agent**: Uses epitope analysis for mutation strategy
- **quality-agent**: Can check if hotspots are maintained

## References

- Kunik et al. (2012). "Paratome: an online tool for systematic identification of antigen-binding regions"
- Thornton group epitope standards: 4.5Å distance cutoff
