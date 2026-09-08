---
name: evolutionary-analysis
description: |
  Analyze objective-aware search lineages and a bounded, cached FoldMason
  structural-similarity tree during Reflection.
---

# Evolutionary Analysis

## ReflectAgent Tool

ReflectAgent receives `analyze_evolution_tree` as a direct tool. OpenDDE Harness runs this
tool at most once per Reflection over the task's append-only search history.
Lineage trajectories are an internal component of this Skill; DesignAgent
receives only the compact evidence and recommendations produced by Reflection.

The tool always computes the cheap, exact parent-child lineage. Python applies one
fixed automatic FoldMason policy: it uses a deterministic representative set,
reuses the latest cached tree when too few new structures exist, and degrades to
lineage-only evidence on timeout or tool failure. This policy is not configurable
by YAML, the user, or ReflectAgent. ReflectAgent must not interpret a FoldMason
guide tree as natural phylogeny or evolutionary time.

Call the tool exactly once during each Reflection. Copy the candidate database,
current parent, objective direction, cycle, binder chains, and CDR regions from
the current trusted task context. Do not inspect directories, substitute another
database, or invent missing paths. Tool failure is non-fatal: retain the supplied
trajectory evidence and mark only the unavailable evolutionary evidence missing.

## Standalone Usage

Population-only analysis:

```bash
python -m opendde_harness.plugin.protein_design.servers.backends.evolution_analysis \
    --candidates_json_path ~/.opendde_harness/protein_design/TASK_ID/search_history.json
```

The standalone script can still summarize already generated Newick and alignment
files. The design loop itself owns structure extraction, FoldMason execution,
candidate-to-leaf mapping, caching, and timeouts.

## Input

The candidate path may be `search_history.json`, `population.json`, or a task
directory containing one of those files. Reflection always uses
`search_history.json` when it is available.

OpenDDE Harness's canonical fields are `candidate_id`, `parent_id`, `cycle`, `objective`,
`objective_key`, `minimize`, `metrics`, `mutations`, `skill_id`, gate evidence,
and the population action. Unknown or obsolete external field names are not
accepted; producers must normalize data at OpenDDE Harness's task boundary.

## Report Sections

- known parent-child search lineages
- FoldMason structural-similarity summary for each binder chain when available
- objective-aware root-to-leaf trajectories
- mutation recurrence across candidates and independent parents
- sequence conservation for each supplied alignment

Do not interpret candidate count alone as convergence. Require recurrence
across independent parents and positive objective-aware improvement evidence.
Keep the known parent-child search lineage separate from the FoldMason
structural-similarity guide tree. Report exact candidate IDs, mutations, and
objective values. Describe structural clusters or conservation only when the
FoldMason result provides them.

## Component CLIs

For a smaller report, use:

```bash
python -m opendde_harness.plugin.protein_design.servers.backends.lineage_analysis \
    --candidates_json_path ~/.opendde_harness/protein_design/TASK_ID/search_history.json
```
