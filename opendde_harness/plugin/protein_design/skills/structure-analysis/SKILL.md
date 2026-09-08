---
name: structure-analysis
description: |
  Analyze protein-ligand or protein-protein interactions from a single structure
  file with PLIP in the OpenDDE Harness compute service.
  Use this skill when you need to profile interaction types such as hydrogen
  bonds, hydrophobic contacts, pi-stacking, salt bridges, or water bridges.

  PLIP (Protein-Ligand Interaction Profiler) runs inline in the persistent
  configured compute service. It does not launch or pull a separate
  PLIP container at runtime.
---

# Structure Analysis Skill

## EvoReflect Direct Tool

Within ReflectAgent, call `analyze_current_cycle_structure`. It accepts trusted
candidate IDs paired with the exact structure paths from Reflection's allowed
catalog. The compute service validates that every path belongs to the active
task output, invokes the packaged backend with validated arguments, parses the reports,
and returns a bounded interaction summary. Do not invent paths or load raw PLIP
reports into model context.

Call the tool at most once per Reflection and analyze no more than three
informative complexes: the current parent, the best new candidate, and one
contrasting candidate. Copy candidate IDs, structure paths, binder chains,
target chains, and cycle number exactly from the trusted catalog. PLIP failure
is non-fatal; continue from fold metrics and supplied contact-gate evidence and
mark detailed interaction evidence unavailable.

## Standalone Usage

```bash
python -m opendde_harness.plugin.protein_design.servers.backends.plip_runner \
    outputs/.../some_complex_structure.cif \
    --cyc-num 1 \
    --name cycle_1_mut_001 \
    --binder-chain <binder_chain_ids> \
    -o outputs/.../some_structure_interactions.txt
```

Pass a SINGLE predicted complex structure file. The file should already contain both target and binder chains. For protein-protein complexes, provide the binder chain IDs explicitly with `--binder-chain`.

Converts `.cif`/`.mmcif` to `.pdb` automatically when needed, runs PLIP on the given structure, and writes the interaction report to a txt file.

Do not split the complex into separate binder/target files. This wrapper expects one complex file and optional binder chain IDs.

## Interpreting Results

Focus on the interaction summary PLIP reports:

- **H-bonds / salt bridges** — often strongest interaction signals
- **Hydrophobic contacts** — useful for interface packing
- **Pi-stacking** — notable for aromatic residues

Output is a text report with interaction type counts, residue identities, and distances per binding site.

Use interaction claims only for the exact analyzed complex. Do not infer
hydrogen bonds, salt bridges, pi-stacking, or residue contacts from confidence
scores alone.

## Dependencies

The persistent configured compute service provides both runtime
dependencies:

- **PLIP**: invoked inline through `PLIP_COMMAND`
- **biotite**: converts `.cif`/`.mmcif` structures to standard `.pdb`

No separate PLIP Docker image is required or launched by this skill.
