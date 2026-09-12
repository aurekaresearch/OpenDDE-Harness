# Protein-design YAML parameter reference

This reference describes task YAML accepted by `ddeharness protein-design validate`
and `ddeharness protein-design start`. Start from [CRLF2](examples/crlf2_quickstart.yaml)
or [CACNG1](examples/cacng1_quickstart.yaml). YAML contains scientific settings;
LLM credentials, compute tokens, and service configuration belong in the application
configuration, not this file. Unknown keys are rejected before launch.

## Configuration precedence and indexing

- Explicit task `fold` fields override application `fold_defaults`. Remaining
  fields use the documented backend defaults and environment resolution below.
- `design` fields define the baseline policy. A matching `design.cycle_schedule`
  entry overrides only its listed fields. Gaps and omitted fields use the original
  baseline, **not the previous stage**.
- Runtime `num_sequences` adjustments take priority over the schedule and remain
  active for later cycles until adjusted again. They are not written back to YAML.
- Cycles start at **0**; `start_cycle` and `end_cycle` are **inclusive**. A ten-cycle
  task runs cycles 0 through 9. Initial scoring is displayed as cycle -1 and is
  outside the schedule. Optional terminal refolding is also outside the schedule.
- Residue positions are zero-based and inclusive. Quote ranges such as
  `"25:34,49:58,98:114"`; unquoted `2:4` can be parsed as a YAML 1.1 integer.

## Exploration followed by refinement

Add this `design` block to a complete target/binder YAML. The numbers illustrate
configuration, not experimentally validated optimal settings.

```yaml
design:
  n_cycles: 10
  num_sequences: 8
  population_size: 20
  router_selection_strategy: weighted
  cycle_schedule:
    - start_cycle: 0
      end_cycle: 3
      num_sequences: 24
      population_size: 40
      parent_fitness_temperature: 1.5
      router_skill_probabilities:
        cdr-full-redesign: 0.8
        cdr-point-mutation: 0.2
    - start_cycle: 4
      end_cycle: 6
      num_sequences: 12
      population_size: 20
      parent_fitness_temperature: 0.5
      router_skill_probabilities:
        cdr-full-redesign: 0.4
        cdr-point-mutation: 0.6
    - start_cycle: 7
      end_cycle: 9
      num_sequences: 4
      population_size: 8
      parent_fitness_temperature: 0.15
      router_skill_probabilities:
        cdr-full-redesign: 0.1
        cdr-point-mutation: 0.9
```

`agent` is the backward-compatible router default: the LLM chooses among legal
skills and sees the weights as priors. `weighted` draws one legal skill per cycle
using the weights and the task seed, then restricts the LLM to that skill. It
controls **probabilities across cycles**, not an exact quota or a within-batch
mixture. A retry of the same cycle uses the same draw. Ten cycles need not realize
an exact 80/20 split. Changing batch sizes also means a fraction of cycles is not
the same as a fraction of generated sequences.

The legal skill menu is evaluated first. Masked CDRs, an empty population, or a
poly-alanine mutable region can require bootstrap redesign. Explicit bootstrap
or stagnation redesign takes precedence over weights. Inverse folding requires
parent structure context; ESM2 proposals require the corresponding compute model.
Weights cannot make an unavailable or inappropriate operation available.

A stage's `router_skill_probabilities` replaces the whole mapping: omitted skills
have zero weight. The same rule applies to an explicit baseline mapping in
`weighted` mode. For compatibility, baseline `agent` mode retains the historical
implicit point-mutation weight of 1 if that key is omitted. List all desired weights
explicitly when comparing policies. Positive weights are normalized; their sum
need not be 1. The mandatory bootstrap/legal fallback may still override zeros.

At a stage boundary the policy is applied before parent selection and design.
Shrinking `population_size` reselects survivors under the new capacity and diversity
limits before choosing a parent; increasing it permits future growth and does not
restore discarded candidates automatically. The best historical candidate remains
part of run history. Speculative next-cycle design is skipped across stage
boundaries so it cannot use stale parameters; within a stage it remains available.
This can reduce overlap at a boundary but does not add extra design cycles.

The validation summary includes the schedule. Each actual cycle emits a
`cycle_config` progress event and a `protein_design.cycle_config` trace artifact,
including the effective batch size, capacity, router policy, and parent temperature.

## Stagnation and exploration

`design.stagnation_full_redesign_threshold` defaults to `0` (disabled), including
when omitted from the quickstarts. For example, setting it to `5` requests a full
CDR redesign after five evaluated cycles without an improvement in the best
objective. An improvement resets the streak. The intended repeat interval between
forced redesigns is at least five cycles. Fixed/framework residues remain protected.
This counter measures completed search evaluations, not elapsed time: a slow or
stuck API request does not trigger redesign. Skipped failed cycles do not necessarily
advance this counter.

Forced bootstrap/stagnation redesign takes precedence over ordinary router weights.
Other exploration controls include nonzero full-redesign weights, higher fitness
parent temperatures, uniform parent sampling mixed into fitness selection, and
population diversity constraints. Reflection can provide revised design guidance.
None guarantees escape from a local optimum. There is currently no automatic
stagnation-triggered increase in temperature, population capacity, or proposal count;
`cycle_schedule` follows fixed cycle intervals rather than reacting to progress.

**Current limitation:** speculative next-cycle design uses a copy of the current
metadata before the current fold updates the stagnation streak. Its reuse check
does not yet validate that streak, and a speculative forced-redesign timestamp is
not synchronized back to live metadata. Therefore the trigger round and cooldown
are not yet guaranteed to follow the intended rule exactly when speculation is
reused. This known issue remains to be fixed and tested before claiming strict
stagnation-trigger timing. Stage-boundary speculation is disabled separately.

## Root sections

| Field | Default / accepted values | Purpose |
|---|---|---|
| `seed` | `42`, integer | Reproducible parent draws and weighted skill draws; separate from structure-prediction `fold.seeds`. |
| `design` | Optional mapping | Search policy and gates; omitted settings use defaults below. |
| `fold` | Optional mapping | Folding backend settings, merged with onboarding defaults. |
| `target` | Required mapping | Target name and one or more antigen chains. |
| `initial_binders` | Required non-empty list | Antibody scaffold and residue permissions. |
| `compute` | Optional mapping | Pin a service or select a configured worker/profile. |
| `llm` | Optional mapping | Task-specific model settings; credentials remain in application configuration. |
| `benchmark_metadata` | Empty mapping | Provenance/benchmark annotations. Does not tune search behavior. |

## Design fields

Names in this table are relative to `design`.

| Field | Default / constraint | Effect |
|---|---|---|
| `n_cycles` | `3`; positive integer | Number of design cycles, excluding initial scoring and terminal refolding. |
| `num_sequences` | `8`; positive integer | Requested proposals per cycle; invalid or duplicate proposals can reduce actual usable output. Stage-configurable. |
| `num_mutations` | Instruction text; e.g. `"1-4"` | Mutation-count instruction supplied to the design prompt. It is not a universal hard numeric constraint for every skill. |
| `reflection_interval` | `20`; positive integer | Request reflection after each multiple of this many completed cycle positions. |
| `cycle_retry_limit` | `2`; integer 0–5 | Retries after the first attempt; 2 means up to 3 attempts. Retries stay in the same schedule interval. |
| `skip_failed_cycles` | `true` | Continue past skippable failed cycles after retries. A fold batch with no scored results remains fatal. |
| `cycle_schedule` | Empty list | Ordered, non-overlapping cycle intervals. See the stage schema below. |
| `population_size` | `20`; positive integer | Maximum retained population, not a guarantee that this many pass gates/diversity checks. Stage-configurable. |
| `constrained_min_cdr_distance` | `0.05`; [0,1] | Minimum normalized CDR edit distance from every retained candidate. |
| `constrained_max_position_reuse_fraction` | `0.75`; (0,1] | Limits how many retained candidates reuse a mutated position; cap scales with population capacity. |
| `constrained_max_mutation_reuse_fraction` | `0.30`; (0,1] | Limits reuse of an exact residue replacement within the retained population. |
| `parent_selection_strategy` | `fitness`; `uniform`, `greedy`, `fitness`, `llm` | Uniform sampling, best objective, fitness-weighted sampling, or agent selection. Temperature affects only `fitness`. |
| `parent_fitness_temperature` | Unset; finite positive number | Optional constant fitness-sampling temperature. Stage-configurable. Lower values favor better objectives more strongly. |
| `parent_fitness_temperature_start` | `1.0`; positive | Starting temperature of geometric annealing over all cycles when a constant temperature is unset. |
| `parent_fitness_temperature_end` | `0.2`; positive | Ending temperature of geometric annealing. Do not specify baseline start/end together with a baseline constant temperature. A stage constant temporarily overrides annealing. |
| `parent_fitness_uniform_fraction` | `0.10`; [0,1] | Uniform exploration mixed into fitness weights. 1 makes selection uniform regardless of temperature. |
| `router_selection_strategy` | `agent`; `agent` or `weighted` | Agent choice with priors, or reproducible weighted selection of a single legal skill per cycle. Stage-configurable. |
| `router_skill_probabilities` | Point mutation only | Weights for the four skills below. Non-negative finite values with a positive total; zero normally disables a skill. Stage-configurable. |
| `bootstrap_full_redesign_cycles` | `0`; non-negative integer | Force full redesign for cycles below this number. Masked/unusable parents can still require bootstrap when this is 0. |
| `stagnation_full_redesign_threshold` | `0` (off); non-negative integer | Request full redesign after the configured no-improvement streak, subject to its repeat interval. |
| `optimization_metric` | `loss`; `loss` or `iptm` | Minimize the weighted loss or maximize ipTM. No separate direction field is needed. |
| `loss_weights` | Built-in coefficients below | Partial mapping overrides named coefficients; omitted coefficients retain defaults. Set a term to 0 to disable it. |
| `cdr_contact_fraction_threshold` | `0.5` | Hard CDR-contact gate threshold; intended fractional values are in [0,1]. |
| `hotspot_contact_cutoff_a` | `5.0` Å | Distance used to evaluate hotspot contact evidence. Use a positive distance. |
| `enable_quality_check` | `true` | Enable eligible Quality Agent checks after geometry gates. |
| `quality_check_threshold` | `0.7`; [0,1] | ipTM threshold that **triggers** quality checking for non-full-redesign candidates. It is not a minimum agent acceptance score. |
| `initial_structure_path` | Unset | Optional compute-readable initial complex used for structural design context. It does not skip initial folding. |
| `esm_device` | Unset | ESM2 device hint such as `cuda:0`; compute placement/runtime may determine the actual device. API folding still uses local ESM2 for relevant scoring/proposals. |
| `post_refold_filter` | Optional mapping | Optional terminal SolubleMPNN/refold/filter stage. |
| `post_refold_filter.enabled` | `false` | Enable terminal refolding and agent ranking. Extra computation follows the search. |
| `post_refold_filter.top_k` | `20`; positive integer | Maximum final selections; also determines the current bounded refold parent pool. Has no effect when disabled. |

Supported skill keys under `router_skill_probabilities`:

| Key | Operation |
|---|---|
| `cdr-point-mutation` | LLM-guided bounded edits to mutable CDR residues. |
| `cdr-full-redesign` | Broad CDR sequence redesign while respecting configured permissions. |
| `antibody-inverse-folding` | SolubleMPNN proposals using an available parent structure. |
| `esm2-guided-mutation` | ESM2-guided sequence edits; requires ESM2 on compute. |

### Cycle schedule entry

| Field | Requirement | Meaning |
|---|---|---|
| `start_cycle` | Required integer ≥0 | First included cycle. |
| `end_cycle` | Required integer ≥`start_cycle`, less than `n_cycles` | Last included cycle. |
| `num_sequences` | Optional positive integer | Override batch size in this interval. |
| `population_size` | Optional positive integer | Override retained capacity in this interval. |
| `router_selection_strategy` | Optional `agent` or `weighted` | Override selection mode in this interval. |
| `router_skill_probabilities` | Optional non-negative finite mapping, positive total | Replace the entire skill-weight map for this interval. |
| `parent_fitness_temperature` | Optional finite positive number | Override parent-sampling temperature with a constant for this interval. |

Every entry must override at least one parameter. Entries must be listed in order;
overlap, reversed/out-of-range intervals, misspelled keys, and empty policies are
errors. Other design settings remain task-wide; do not put folding, LLM, gate,
residue, or reflection settings inside a schedule entry.

### Loss coefficients

Each row is a field under `design.loss_weights`. Values must be finite and
non-negative. Omitting the mapping uses all built-in defaults. A fully explicit
mapping is useful for reproducibility; a short partial mapping is sufficient to
change selected terms only.

| Field | Default | Weighted component |
|---|---|---|
| `plddt` | 1.0 | Binder confidence loss. |
| `i_plddt` | 1.0 | CDR confidence loss. |
| `pae` | 0.1 | Binder PAE loss. |
| `i_pae` | 0.5 | CDR–target interface PAE loss. |
| `i_ptm` | 1.0 | Interface pTM loss (`1 - ipTM`). |
| `con` | 0.1 | Binder intrachain contact loss. |
| `i_con` | 0.1 | Paratope contact loss. |
| `rg` | 0.1 | Binder radius-of-gyration regularization. |
| `dgram_cce` | 0.01 | Framework distogram consistency. |
| `esm2` | 0.1 | ESM2 sequence-prior loss. |

Terminal refolding currently uses unique, gate-passing, finitely scored trajectory
parents ordered by the objective, capped at `4 * top_k`. SolubleMPNN samples 40
variants per parent and selects up to the required four distinct valid variants
for refolding. Failed or insufficient groups remain failures. Agent rankings must
cover eligible candidates with contiguous ranks. If the post-filter agent fails,
the implementation can fall back to deterministic objective ordering; inspect
`final_selection` errors and mode rather than assuming agent ranking succeeded.

## Fold fields

Names below are relative to `fold`. Defaults apply after onboarding defaults are
merged. API mode owns its remote MSA/template pipeline and rejects `use_msa: false`
and local A3M paths. Local GPU/image settings do not allocate remote API GPUs.

| Field | Default | Meaning / applicability |
|---|---|---|
| `model` | `opendde` | Only supported fold/refold backend. Usually safe to omit. |
| `execution_mode` | Environment override, otherwise `local` | `local`, `docker`, or `api`; mode of prediction on compute. |
| `api_url` | Environment override, otherwise `https://api.aurekabio.cloud` | API gateway; explicit task URL has highest precedence. |
| `api_poll_interval_seconds` | 5 | API status-poll interval. |
| `api_timeout_seconds` | 7200 | Remote job wait budget, including queue/warm-up. The client compute-job wait has a separate timeout. |
| `api_request_timeout_seconds` | 60 | Per-request HTTP timeout for submit/status/download operations. |
| `api_stalled_poll_limit` | 3 | Consecutive possibly-stalled polls before aborting. |
| `seeds` | `[42]` | Structure-prediction model seeds; independent from root `seed`. |
| `diffusion_samples` | 1 | Samples per prediction seed. API parameter `n_samples`. |
| `diffusion_steps` | 200 | Diffusion steps. API parameter `n_step`. |
| `need_atom_confidence` | `true` | Request atom/tensor confidence used for loss scoring. Required for API loss optimization. |
| `use_msa` | `true` | Local MSA features. API service always manages MSA. |
| `enable_msa_search` | `false` | Local fallback MSA/template search; prefer preparing target A3M files explicitly before launch. |
| `use_templates` | `false` | Local template use. Does not disable the API service pipeline. |
| `recycling_cycles` | 10 | Local model recycling iterations. |
| `deterministic` | `true` | Local deterministic inference setting. |
| `gpus` | `"all"` | Local GPU selection; use worker-visible IDs. API mode does not use this to select remote GPUs. |
| `persistent_worker` | `true` | Reuse the local inference worker instead of one-shot execution where supported. |
| `persistent_worker_timeout_seconds` | 900 | Persistent worker wait/readiness timeout. |
| `subprocess_timeout_seconds` | 7200 | One-shot Docker batch timeout. |
| `ipsae_dist_cutoff` | 10.0 Å | Distance threshold for ipSAE reporting. |
| `ipsae_pae_cutoff` | 10.0 | PAE threshold for ipSAE reporting. |
| `enable_batch_inference` | `true` | Accepted compatibility field; the current predictor does not branch on this flag. Omit rather than rely on `false` to force serial inference. |
| `target_msa` | Unset | Accepted compatibility mapping with no current predictor read. Use per-chain A3M fields instead. |
| `image` | `STRUCTPRED_IMAGE_OPENDDE`, otherwise `opendde:latest` | Docker-mode image override; unused for local/API prediction. Managed compute normally supplies the runtime image. |

API download retry count/delay are currently implementation defaults (five retries,
three seconds), not accepted YAML fields. Do not add guessed retry keys. Model and
asset paths are compute-service settings, not task YAML keys.

## Target and antibody fields

| Field | Requirement / meaning |
|---|---|
| `target.name` | Required non-empty name; also used as the target's memory scope. |
| `target.chains` | Required non-empty mapping of distinct chain IDs. A chain may be a sequence string or the mapping below. |
| `target.chains.<id>.sequence` | Required antigen amino-acid sequence. Verify construct and source independently. |
| `target.chains.<id>.hotspots` | Optional intended epitope positions, list or quoted ranges. Omitted/empty means no explicit hotspots. |
| `target.chains.<id>.unpairedMsaPath` | Optional compute-readable unpaired A3M path; local folding only. |
| `target.chains.<id>.pairedMsaPath` | Optional compute-readable paired A3M path; local folding only. |
| `initial_binders[].name` | Optional candidate ID, otherwise `initial_0`, `initial_1`, etc. IDs must be unique. |
| `initial_binders[].chains` | Required antibody chain mapping. IDs must differ from target IDs. |
| `initial_binders[].chains.<id>.sequence` | Required initial sequence. `X` may represent intended mutable bootstrap positions. |
| `initial_binders[].chains.<id>.chain_type` | Required: one `VHH`, one `scFv`, or paired `VH` and `VL`. `VK`, `VLK`, `VL-kappa`, `VL-lambda` are accepted VL aliases. |
| `initial_binders[].chains.<id>.cdr_regions` | CDR annotation and default mutable positions. At least this or `fixed_residues` is required. |
| `initial_binders[].chains.<id>.designable_residues` | Optional explicit mutable subset, taking precedence over CDR-derived permissions. Fixed residues still win. |
| `initial_binders[].chains.<id>.fixed_residues` | Immutable positions; overrides all design permissions. If exactly the complement of CDRs and no broader explicit designable set exists, it is redundant for effective permissions. |
| `initial_binders[].chains.<id>.unpairedMsaPath` | Optional local binder unpaired A3M path. |
| `initial_binders[].chains.<id>.pairedMsaPath` | Optional local binder paired A3M path. |

When only CDRs are configured, all positions outside them are fixed automatically.
When only fixed residues are configured, the complement becomes the mutable/CDR
region. Multiple initial binders currently must agree on chain IDs, initial chain
sequences, residue permissions, MSA inputs, and chain types; use separate tasks for
different scaffold topologies. Binder-side `hotspots` is not a supported key.

## Compute and LLM fields

| Field | Default / meaning |
|---|---|
| `compute.url` | Optional service URL; pins the complete task to that endpoint. |
| `compute.worker_id` | Optional registered worker ID. Prefer one placement selector rather than conflicting selectors. |
| `compute.profile` | Optional configured worker profile. |
| `compute.placement` | `auto` or a mapping specifying resource placement below. |
| `compute.placement.fold` | Optional list of GPU indices for folding. |
| `compute.placement.esm` | Optional ESM2 GPU index. |
| `compute.placement.mpnn` | Optional SolubleMPNN GPU index. |
| `compute.placement.cp_degree` | Defaults to the length of an explicit `fold` GPU list, otherwise `1`; must equal that list length when supplied. |
| `llm.model_name` | Inherit configured default model; task model override. |
| `llm.temperature` | Inherit the phase profile; task generation temperature override. |
| `llm.max_tokens` | Inherit the phase profile; positive output token limit. |

## Removing redundant settings safely

The bundled quickstarts omit explicit default design gates/objective, the disabled
post-filter block, unused batch flag, and framework positions already implied by
CDRs. They retain non-default mutation instructions, skill weights, folding seeds,
local MSA policy, and GPU/device hints because deleting those could change the run.

- Default-valued `design` fields can usually be omitted to inherit defaults. Keep
  them if pinning current behavior across software versions is intentional.
- Do not blindly delete a default-valued `fold` field: onboarding `fold_defaults`
  may differ. For example, explicit `enable_msa_search: false` can override a saved
  `true`; removing it changes behavior.
- `fixed_residues` inside a CDR, or combined with a broader `designable_residues`, is
  not redundant. Do not delete it based only on the presence of `cdr_regions`.
- Temperature has no sampling effect under `uniform`, `greedy`, or `llm` parent
  selection. `top_k` has no effect while post-filtering is disabled.
- API-local-only fields and compatibility fields above may be omitted, but do not
  replace a deliberate service configuration by guessing another mode.
- Legacy top-level `post_refold_filter`, unsupported post-filter ranking keys,
  and unknown fields are errors; validation reports them instead of ignoring them.

Validate the edited YAML with the same application configuration used to launch:

```bash
ddeharness protein-design validate --config /absolute/path/design.yaml --json
```

Then inspect the reported schedule, resolved folding mode, compute service and loss
weights. Validation checks configuration; it does not establish that LLM credentials,
models, GPU memory, or external services are available.
