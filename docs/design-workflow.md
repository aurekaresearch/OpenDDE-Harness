# Design workflow

User-facing operational overview. For field definitions, see the [YAML reference](../opendde_harness/plugin/protein_design/skills/protein-design/references/yaml-configuration.md).

## Design loop

```
                    ┌─────────────────────────────────┐
                    │                                 ▼
Target + binder + constraints                Router limits legal skills
         │                                         │
         ▼                                         ▼
       Analyze                        Design Agent selects a skill
                                               │
                    ┌──────────────────────────┘
                    ▼
          Generate candidate sequences
                    │
                    ▼
               OpenDDE fold
                    │
                    ▼
      Loss, structural metrics and gates
                    │
                    ▼
          Population admission ──────────────────────────────┐
                    │                                        │
          ┌─────────┴──────────┐                            │
          ▼                    ▼                            ▼
   Parent selection   Reflection + memory        Optional terminal refold
          │                    │                    + PostFilter
          └─────────┬──────────┘
                    │
                    └────────────────────────────────────────┘
                                  (next cycle)
```

Each cycle follows the same simple contract:

1. Select a parent from the retained population.
2. Let the Router expose only legal design skills.
3. Let the Design Agent choose one skill using the current metrics and search history.
4. Generate sequences while keeping framework and explicitly fixed residues immutable.
5. Fold and score every valid candidate.
6. Apply hard gates, deduplication, diversity constraints, and population admission.
7. Reflect periodically, store structured design cases, and reuse learned skills in later cycles.

## Built-in design capabilities

| Capability | Purpose |
| --- | --- |
| CDR point mutation | Local refinement of one or more CDR positions |
| CDR full redesign | Replace an entire mutable CDR sequence basin |
| Antibody inverse folding | Structure-conditioned sequence proposals with SolubleMPNN |
| ESM2-guided mutation | Sequence-likelihood-guided substitutions |
| Epitope and structure analysis | Explain contacts, hotspots, geometry, and failure modes |
| Evolution and lineage analysis | Summarize progress, stagnation, and successful lineages during Reflection |
| Developability filter | Optional terminal assessment of experimental liabilities |

All design skills use a shared proposal contract. The Router controls legality and scope; the Agent makes the semantic choice; the executor validates and runs the selected skill.

## Runtime architecture

OpenDDE Harness has two layers:

- **Local client environment:** TUI and CLI, agents, configuration validation, gates, population logic, long-term memory, tracing, and the dashboard.
- **On-demand Docker compute service:** OpenDDE folding and GPU-heavy protein tools. The container starts when a task needs it, runs every backend directly inside that one container without starting nested model containers, and removes itself when idle.

A task binds to one compute URL. Different tasks may therefore use Docker services on different GPU servers while remaining visible in one TUI and tracing dashboard.

CDR regions define the mutable scope; explicitly fixed residues take precedence. Local and hosted API folding use the same configurable weighted loss. API requests enable atom confidence by default to obtain the required tensors. See [compute modes](protein-design.md#use-the-hosted-opendde-folding-api).

The search tree displays recorded parent–child lineage, not a phylogenetic tree. Reflection may also use structural evolution analysis; these are distinct sources of evidence.
