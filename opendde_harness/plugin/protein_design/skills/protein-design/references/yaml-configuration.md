# Protein-design YAML configuration

Read this reference when creating or editing a task configuration. Generate the nested YAML form described here, then run `ddeharness protein-design validate --config <path>`. Do not add guessed fields.

## Preparation and review checklist

Read `protein_design_context` first (CLI fallback: `ddeharness protein-design context --json`). Resolve configured compute/folding and MSA policy before authoring YAML. The following sections are a checklist, not three mandatory question rounds. Reuse settings from an explicitly selected example, ask only unresolved choices, combine related questions, and preserve confirmed answers. Once complete, validate and ask one explicit final approval-and-launch question.

### 1. Target / antigen

- target name, organism/species, isoform or domain boundaries when relevant;
- target chain IDs and complete sequences, including source/provenance;
- intended epitope or hotspot residues and whether numbering is zero-based YAML indexing or an external biological numbering scheme;
- whether target MSA features are used, plus absolute unpaired/paired A3M paths and provenance;
- initial target-binder complex structure path and chain mapping, when supplied.

When the user supplies only a target name, use Web search to resolve the missing identity and sequence context before asking for confirmation. Prefer UniProt/NCBI sequence records and use RCSB PDB plus primary literature for epitope evidence. Record accessions or URLs, sequence version, construct/domain boundaries, and the source numbering scheme. Do not silently choose among species, isoforms, extracellular-domain constructs, or competing reported epitopes. Map source residue numbering to zero-based YAML hotspot indices only after confirming the exact sequence; otherwise leave the mapping unresolved. Web search cannot substitute for a readable local MSA or structure path on the compute worker.

API mode uses service-managed MSA: do not offer disabling MSA or search for local A3M files. For local folding only, if target MSA policy is unresolved, ask with other missing target information whether the user wants to:

1. set `fold.use_msa: false`;
2. provide existing paired/unpaired A3M paths; or
3. search online using `protein_design_search_target_msa`.

Do not call the search tool until the user explicitly chooses option 3. The tool accepts one target chain at a time and must never receive a binder/antibody sequence. Do not ask the user to supply compute placement before search. Omit the tool's optional `compute_url` to use the configured worker pool or default endpoint; pass it only when the reviewed design already has an explicit `compute.url`. Worker ID and profile are internal placement concerns and are not MSA-tool arguments. The tool uses Protenix's public MMseqs2 service by default on the selected compute worker and returns `unpaired_msa_path`, `paired_msa_path`, `alignment_depth`, `cached`, `compute_url`, and `compute_worker_id`. Put the two paths under `target.chains.<chain_id>` and bind `compute.url` or `compute.worker_id` to the returned worker. This keeps the generated files visible to Fold. For a multichain antigen, repeat only for target chains for which the user requested MSA search. The public service is shared and rate-limited; set `MMSEQS_SERVICE_HOST_URL`, `OPENDDE_HARNESS_MSA_SERVER_MODE`, and `OPENDDE_HARNESS_MSA_SEARCH_TIMEOUT` on the compute service for a compatible private endpoint and bounded wait.

### 2. Binder / antibody

- binder format (single `VHH`, single-chain `scFv`, or paired `VH/VL`) and binder chain IDs;
- complete initial binder sequences and seed provenance;
- CDR1/CDR2/CDR3 regions for each binder chain;
- framework/fixed residues and any narrower `designable_residues`;
- intended masked `X` bootstrap positions, if any.

If neither the user nor an explicitly selected example supplies a scaffold, read [antibody-frameworks.md](antibody-frameworks.md). Offer only compatible entries, explain that they are starting variable-region scaffolds rather than target-specific binders, and ask the user to choose. Do not restart scaffold selection for a complete example.

Framework and CDR regions belong to the binder/antibody section. Fixed residues always override every design permission.

This schema is antibody-only. Every binder chain must declare `chain_type` and provide either explicit `cdr_regions` or an explicit immutable framework in `fixed_residues`. A runnable topology must be one VHH chain, one scFv chain, or exactly one VH plus one VL chain. General protein, peptide, enzyme, receptor, and other non-antibody binder configurations are rejected during validation.

### 3. Design, scoring, and compute

- cycle count, proposals per cycle, mutation-count range, and reflection interval;
- optimization metric, minimize/maximize direction, and component weights;
- geometry/contact/quality gates;
- constrained-elite population and parent-selection settings;
- enabled proposal Skills and their weights;
- primary OpenDDE fold options, MSA behavior, GPU/compute placement;
- terminal post-refold/PostFilter enabled state, top K, and refold backend.
- transient cycle failure policy: `cycle_retry_limit` (default `2`, so up to three attempts total) and `skip_failed_cycles` (default `true`);

Once choices are resolved, prepare and validate the YAML using `--json`, then present its absolute path and final summary including inherited folding/MSA policy and loss weights. Ask explicitly whether the user approves this exact configuration and authorizes launch. Validation is not service readiness or launch approval. Revalidate only after changed input/configuration or an error; renew approval if the plan changes.

## Detailed field reference and staged design

The complete defaults, constraints, cycle scheduling rules, and redundancy guidance
are in [`docs/protein-design-yaml.md`](../../../../../../docs/protein-design-yaml.md).
Use `design.cycle_schedule` for ordered, inclusive, zero-based intervals overriding
`num_sequences`, `population_size`, `router_skill_probabilities`,
`router_selection_strategy`, and `parent_fitness_temperature`. Omitted fields inherit
the original baseline. Use `weighted` routing for probability-based cycle selection;
the default `agent` mode treats weights as priors. A stage map replaces all skill weights.

## Schema template (not directly executable)

The following placeholders illustrate the schema; replace them with verified sequences and real paths before validation. Runnable examples are `docs/examples/crlf2_quickstart.yaml` and `docs/examples/cacng1_quickstart.yaml`; obtain their absolute paths from the context tool without requiring an example README. To inherit configured placement and folding mode, omit `compute` and `fold.execution_mode`/`fold.api_url` overrides. Explicit YAML settings override defaults. Omit `design.loss_weights` to use the built-in weights; never copy redacted display URLs into YAML.

```yaml
seed: 42

design:
  n_cycles: 100
  num_sequences: 8
  reflection_interval: 20
  num_mutations: 1-4
  optimization_metric: loss
  # Always list every coefficient. Omitted keys inherit built-in defaults;
  # use 0.0 for terms that must not contribute to the objective.
  loss_weights:
    plddt: 0.0
    i_plddt: 0.0
    pae: 0.0
    i_pae: 0.0
    i_ptm: 1.0
    con: 0.0
    i_con: 0.1
    rg: 0.0
    dgram_cce: 0.0
    esm2: 0.1
  cdr_contact_fraction_threshold: 0.5
  hotspot_contact_cutoff_a: 5.0
  quality_check_threshold: 0.7
  parent_selection_strategy: fitness
  router_skill_probabilities:
    cdr-point-mutation: 0.65
    cdr-full-redesign: 0.10
    antibody-inverse-folding: 0.15
    esm2-guided-mutation: 0.10
  post_refold_filter:
    enabled: false
    top_k: 20

fold:
  model: opendde
  gpus: "0,1,2,3,4,5,6,7"
  enable_batch_inference: true
  seeds: [973520]
  use_msa: true
  enable_msa_search: false

target:
  name: TARGET_NAME
  chains:
    A:
      sequence: TARGET_SEQUENCE
      unpairedMsaPath: /absolute/path/to/unpaired.a3m
      pairedMsaPath: /absolute/path/to/paired.a3m
      hotspots: []

initial_binders:
  - name: initial_vhh
    chains:
      D:
        sequence: BINDER_SEQUENCE
        cdr_regions: "25:34,49:58,98:114"
        fixed_residues: "0:24,35:48,59:97,115:124"
        chain_type: VHH
```

Replace every placeholder. Do not emit placeholder text in a runnable file.

## Root fields

| Field | Meaning | Rules |
|---|---|---|
| `seed` | Python-side deterministic search seed. | Integer; default `42`. |
| `compute` | Optional task-level compute placement. | Omit to use the plugin's configured pool. |
| `llm` | Optional per-task Agent model settings. | Omit to inherit OpenDDE Harness defaults. |
| `design` | Search, objective, gate, population, and Skill settings. | Required for the nested form. |
| `fold` | Primary structure-prediction backend and backend options. | Use `model: opendde` in generated configurations. |
| `target` | Target identity and target-chain inputs. | A real name and at least one chain sequence are required. |
| `initial_binders` | One or more starting binder candidates. | Each entry must preserve a consistent chain topology. |

## Compute placement

Use at most one selector unless an explicit URL and matching worker ID are intentionally paired.

| Field | Meaning |
|---|---|
| `compute.url` | Bind the whole task to one compute service URL. |
| `compute.worker_id` | Select one worker from `plugins.config.protein-design.compute_workers`. |
| `compute.profile` | Let OpenDDE Harness choose the least-loaded healthy worker with this profile. |
| `compute.placement` | GPU placement inside the selected worker: `auto` (default) or a mapping. |

Placement is immutable after task start. All cycles and population reads use the selected worker.

### GPU placement inside one worker

Leave `compute.placement` unset unless the user asks for specific GPUs. The
service then leases the least-loaded free GPUs per job. Use the `compute.gpus`
inventory from `protein_design_context` for the available indices.

```yaml
compute:
  placement:
    fold: [1, 2, 3]
    esm: 0
    mpnn: 0
    cp_degree: 3
```

| Field | Meaning |
|---|---|
| `placement.fold` | GPU indices held exclusively by one OpenDDE fold. |
| `placement.esm` | Single GPU index for ESM-2 scoring and ESM-guided proposals. |
| `placement.mpnn` | Single GPU index for SolubleMPNN generation. |
| `placement.cp_degree` | Fold-CP context-parallel degree; defaults to `len(fold)`, otherwise `1`. |

`fold` must list exactly `cp_degree` distinct indices. ESM and SolubleMPNN may
share one GPU with each other but never with a running fold. An explicit request
for a busy GPU waits in the queue instead of failing. Validate the file with
`ddeharness protein-design validate --config <path>` before launch.

## LLM settings

| Field | Meaning |
|---|---|
| `llm.model_name` | Model used by the task's Agents. |
| `llm.temperature` | Sampling temperature. |
| `llm.max_tokens` | Maximum response tokens per Agent call. |

## Design fields

Loss weights are literal coefficients and are not normalized. A partial
`loss_weights` mapping overrides only the named built-in defaults, so omitting
a key does **not** disable that term. Runnable configurations should list all
ten supported keys and set every unused term explicitly to `0.0`.

| Field | Meaning | Default or constraint |
|---|---|---|
| `n_cycles` | Number of design cycles. | `3`; must be positive. |
| `num_sequences` | Requested proposals per cycle. | `8`; must be positive. |
| `reflection_interval` | Cycles between Reflection calls. | `20`; must be positive. |
| `num_mutations` | Instruction passed to mutation Skills. | May be an integer or bounded text such as `1-4`. |
| `initial_structure_path` | Optional verified target-binder complex used as initial structural context. | Absolute readable path; chain IDs must match the YAML. |
| `optimization_metric` | Primary search objective. | Default `loss`, which is minimized; `iptm` is maximized. No other objective is supported. |
| `loss_weights.plddt` | Binder confidence-loss weight. | Default `1.0`. |
| `loss_weights.i_plddt` | CDR confidence-loss weight. | Default `1.0`. |
| `loss_weights.pae` | Binder/global normalized PAE-loss weight. | Default `0.1`. |
| `loss_weights.i_pae` | CDR-target interface PAE-loss weight. | Default `0.5`. |
| `loss_weights.i_ptm` | Interface-pTM loss (`1 - ipTM`) weight. | Default `1.0`. |
| `loss_weights.con` | Binder intrachain contact-loss weight. | Default `0.1`. |
| `loss_weights.i_con` | CDR-versus-framework paratope contact-loss weight. | Default `0.1`. |
| `loss_weights.rg` | Binder radius-of-gyration regularizer weight. | Default `0.1`. |
| `loss_weights.dgram_cce` | Framework distogram consistency weight. | Default `0.01`. |
| `loss_weights.esm2` | ESM-2 sequence-prior weight. | Default `0.1`. |
| `cdr_contact_fraction_threshold` | Hard population gate for CDR/antibody contacts. | Default `0.5`. |
| `hotspot_contact_cutoff_a` | Contact cutoff in angstroms for hotspot evidence. | Default `5.0`. |
| `enable_quality_check` | Enable the Quality Agent. | Default `true`. |
| `quality_check_threshold` | ipTM threshold that triggers a Quality Agent check for eligible non-full-redesign candidates. | Default `0.7`; range `[0,1]`. |
| `population_size` | Maximum retained constrained-elite population. | Default `20`. |
| `constrained_min_cdr_distance` | Minimum normalized CDR edit distance from every retained candidate. | Default `0.05`; range `[0,1]`. |
| `constrained_max_position_reuse_fraction` | Maximum retained-candidate reuse fraction for a mutation position. | Default `0.75`. |
| `constrained_max_mutation_reuse_fraction` | Maximum reuse fraction for an exact amino-acid replacement. | Default `0.30`. |
| `parent_selection_strategy` | Parent sampler. | One of `uniform`, `greedy`, `fitness`, `llm`; default `fitness`. |
| `parent_fitness_temperature_start` | Initial fitness-sampling temperature. | Default `1.0`; positive. |
| `parent_fitness_temperature_end` | Final fitness-sampling temperature. | Default `0.2`; positive. |
| `parent_fitness_uniform_fraction` | Uniform exploration mixed into fitness sampling. | Default `0.10`; range `[0,1]`. |
| `router_skill_probabilities` | Availability weights for built-in proposal Skills. | Non-negative mapping; zero disables a Skill. |
| `esm_device` | ESM-2 device hint. | Example: `cuda:0`. |
| `post_refold_filter.enabled` | Generate SolubleMPNN variants from trajectory structures, refold selected sequences, and let the PostFilter Agent rank usable results. | Default `false`. |
| `post_refold_filter.top_k` | Maximum number selected from the Agent's final order. | Default `20`; positive. |

When enabled, terminal refolding uses a bounded set of unique, gate-passing,
finitely scored trajectory parents (at most `4 * top_k`), ordered by the objective.
SolubleMPNN produces variants for refolding; inspect final selection errors and
mode because invalid agent ranking can fall back to objective ordering. See the
complete reference for the current selection policy. Only `enabled` and `top_k`
are accepted in this block.

Supported router keys are exactly:

- `cdr-point-mutation`
- `cdr-full-redesign`
- `antibody-inverse-folding`
- `esm2-guided-mutation`

## Primary fold fields

| Field | Meaning |
|---|---|
| `fold.model` | Primary design-loop fold backend. Generated configs must use `opendde`. |
| `fold.gpus` | GPU IDs visible to the backend, usually a comma-separated string. |
| `fold.enable_batch_inference` | Allow backend batch folding. |
| `fold.seeds` | Structure-prediction seeds. |
| `fold.use_msa` | Local mode uses supplied MSA features. API mode requires this to be true because the remote service owns MSA/template preparation. |
| `fold.enable_msa_search` | Backend-side fallback search during Fold. Keep `false` for reviewed configurations; use `protein_design_search_target_msa` before launch so paths and provenance are explicit. |
| `fold.execution_mode` | `local` (default), `docker`, or `api`. API mode submits an asynchronous remote OpenDDE job and downloads its result archive. |
| `fold.image` | Optional Docker-mode image override; managed compute normally provides the image. Unused for local/API prediction. |
| `fold.api_url` | External gateway origin; defaults to `https://api.aurekabio.cloud`. Explicit YAML overrides `OPENDDE_HARNESS_OPENDDE_API_URL`, which overrides the default. |
| `fold.api_poll_interval_seconds` | Remote job polling interval; default `5`. |
| `fold.api_timeout_seconds` | Overall remote job timeout including queue and warm-up; default `7200`. |
| `fold.api_request_timeout_seconds` | Timeout for each submit, status, or download HTTP request; default `60`. |
| `fold.api_stalled_poll_limit` | Abort and let the cycle retry after this many consecutive `possibly_stalled` polls; default `3`. |
| `fold.diffusion_samples` | Maps to remote `parameters.n_samples`; default `1`. |
| `fold.diffusion_steps` | Maps to remote `parameters.n_step`; default `200`. |

Remote API mode sends raw chains, preserves chain IDs and model seeds, polls the
LRO-style job until completion, and safely extracts the downloaded ZIP into the
task fold directory. Remote NFS paths in the job response are never treated as
local paths. The service performs its own MSA/template preparation, so local
`pairedMsaPath` and `unpairedMsaPath` values are rejected in API mode.

API requests default to `fold.need_atom_confidence: true`, producing matched
`full_data_sample_*.json` artifacts with atom pLDDT, token PAE, and contact
probabilities. Use `design.optimization_metric: loss` in both API and local mode
for the same 10-component weighted objective. Disabling atom confidence with
API loss optimization is rejected before launch; missing tensors fail scoring
without falling back to ipTM.

Minimal remote fold block:

```yaml
design:
  optimization_metric: loss

fold:
  model: opendde
  execution_mode: api
  api_url: https://api.aurekabio.cloud
  seeds: [973520]
  diffusion_samples: 1
  diffusion_steps: 200
  use_msa: true
```

Unknown keys under `fold` are rejected before the task starts. Use only fields
documented by this schema; backend-specific settings must first be added to the
typed fold contract and its validation.

## Target fields

| Field | Meaning |
|---|---|
| `target.name` | Exact long-term-memory scope key for this biological target. Do not silently normalize it. |
| `target.chains.<chain_id>.sequence` | Target-chain amino-acid sequence. |
| `target.chains.<chain_id>.unpairedMsaPath` | Absolute path to an unpaired A3M file. |
| `target.chains.<chain_id>.pairedMsaPath` | Absolute path to a paired A3M file when available. |
| `target.chains.<chain_id>.hotspots` | Zero-based target residue positions defining an intended epitope. An empty list means no specified hotspots. |

If only a target name is provided, search the Web to resolve species and isoform before choosing a sequence. If multiple biologically reasonable sequences, domain boundaries, or epitopes remain, show sourced alternatives and ask the user instead of selecting silently. Do not invent hotspots when no reliable epitope is available; propose `hotspots: []` only as an explicit user-confirmed unconstrained design choice.

## Initial binder fields

| Field | Meaning |
|---|---|
| `initial_binders[].name` | Stable identifier for the starting candidate. |
| `initial_binders[].chains.<chain_id>.sequence` | Full binder-chain sequence. `X` is allowed only at intended mutable bootstrap positions. |
| `cdr_regions` | CDR permission regions. Quoted string ranges are zero-based and inclusive, for example `"25:34,49:58,98:114"`. |
| `designable_residues` | Optional narrower explicit mutation permission. When present, it takes precedence over `cdr_regions`. |
| `fixed_residues` | Immutable residues. This always overrides both `cdr_regions` and `designable_residues`; quote range strings. |
| `chain_type` | Required antibody-chain type: `VHH`, `scFv`, `VH`, or `VL`. `VK`, `VL-kappa`, and `VL-lambda` normalize to `VL`. Other values are rejected. |

For VH/VL designs, provide exactly one chain of each type and keep their IDs consistent with every supplied structure. Single-chain designs must be explicitly typed as VHH or scFv. Never concatenate separate VH/VL chains into one sequence for configuration purposes.

## Validation checklist

Before final launch review, verify:

1. Target identity, species, isoform, chain sequence, and source are known.
2. Every MSA and structure path is absolute and readable on the compute worker.
3. Target, binder, and structure chain IDs agree.
4. The main `fold.model` is `opendde`.
5. Every designed binder is explicitly identified as a supported antibody topology, and each chain defines CDR regions or an immutable framework.
6. CDR and fixed ranges are zero-based, inclusive, in bounds, and non-overlapping in effect; fixed always wins.
7. Framework positions are not mutable.
8. `X` occurs only in mutable bootstrap residues; generated children containing `X` cannot enter the population.
9. Objective direction is correct: minimize `loss`, maximize `iptm`.
10. Compute placement resolves to a worker advertising OpenDDE.
11. `ddeharness protein-design validate --config <path>` succeeds.
12. Target/antigen, binder/antibody, and design/compute sections were explicitly confirmed within at most three rounds.
13. The user explicitly confirmed launching the final validated YAML after seeing its summary in the third round.

Always quote residue-range strings. YAML 1.1 parsers can interpret an unquoted
single range such as `2:4` as the sexagesimal integer `124`.
