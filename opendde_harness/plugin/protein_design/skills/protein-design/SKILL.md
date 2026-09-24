---
name: protein-design
description: Prepare, validate, run and interpret antibody design or mini binder sequence design and optimization tasks from YAML or user intent.
metadata:
  opendde:
    always: false
---

# Protein design

## Design scope

Choose between antibody design (the default: VHH, scFv or paired VH/VL) and fixed-length single-chain mini binder design and optimization (`design.type: minibinder`). De novo non-antibody backbone generation, enzymes and arbitrary protein design are not supported. Never relabel a mini binder as an antibody.

For mini binders, read [references/minibinder.md](references/minibinder.md) instead of antibody-framework guidance. Accept a complete or X-masked starting sequence of a confirmed length and explicit mutable positions; every X must be mutable. Full redesign resolves X before folding; do not invent CDR annotations. Apply the shared target preparation and launch approval steps below, substituting mini binder design/fixed regions for antibody format/CDR/framework questions. If identity is ambiguous, resolve the designed binder's type before preparing the configuration.

## Read preparation context first

Call `protein_design_context` once at the start of configuration preparation. If unavailable, run `ddeharness protein-design context --json`. Use its verified example paths, configured compute placement, folding/MSA policy, and default loss weights before searching files or writing YAML. Do not dump config.json, inspect tokens, or keep retrying the context call without a configuration change. Its readiness is `not_checked` and worker placement is not yet bound; do not claim the service has been tested. Display URLs are redacted and must not be copied back into YAML.

Choose the workflow from the user's input:

- **Run an example or supplied YAML:** read the selected file first. Preserve its scientific settings, reuse its target/scaffold/CDR annotations, and do not restart scaffold selection when these are already specified. Research only missing/contradictory evidence or verification explicitly requested by the user. Copy bundled files to the task configuration directory before requested adaptations. Review and validate the final copy, then ask one combined launch-confirmation question if nothing is unresolved.
- **Create a new design:** a target-only request is not automatically a request to run an example. Read [references/yaml-configuration.md](references/yaml-configuration.md), resolve missing target/epitope evidence, and ask only unresolved scientific choices. For an antibody missing its scaffold, read [references/antibody-frameworks.md](references/antibody-frameworks.md). For a mini binder, read [references/minibinder.md](references/minibinder.md) and help find public non-antibody starting sequences/structures with provenance. If none is suitable, explain the missing input rather than refusing research or switching to an antibody. Offer compatible choices for confirmation; do not silently substitute example material.

## Resolve example paths

The only bundled YAMLs are `docs/examples/crlf2_quickstart.yaml` (CRLF2, complete starting VHH) and `docs/examples/cacng1_quickstart.yaml` (CACNG1, masked CDRs). Read the exact absolute path returned by the context tool; no README is required. The TUI workspace may differ from the checkout. If unavailable, ask for the real checkout or YAML path and pass that checkout to the context tool once. Do not perform repeated global wildcard searches, invent a path, or fabricate an example as a fallback.

If the intended example is ambiguous, ask which target the user wants. Resolve relative resource paths according to the loader, not the TUI working directory. Verify required inputs and output placement without searching for optional PDB/A3M files that the chosen mode does not need. Locating an example is not launch authorization.

## Inherit configured compute and scoring

When the user requests their configured compute service and folding mode, omit `compute` placement and `fold.execution_mode`/`fold.api_url` overrides so the normal configuration loader inherits them. Do not hard-code a URL found in a prior example or copy redacted display URLs. Omit `design.loss_weights` to use defaults unless different weights are explicitly requested. For an existing YAML with user-specified overrides, surface conflicts rather than silently discarding them.

Set `compute.placement` only when the user names GPUs, for example "run ESM on GPU 0 and OpenDDE candidates on 1-3". Otherwise omit it and let the compute service distribute fold candidates over available GPUs inside one container. Map the request onto the `compute.gpus` inventory from `protein_design_context`: `fold` is a list of GPU indices for independent single-GPU candidate workers; `esm` and `mpnn` are one index each. `cp_degree` defaults to 1 (Fold-CP off) and must only be raised when the user explicitly requests context parallelism; then `fold` must list exactly `cp_degree` GPUs. Confirm the mapping with the user and validate the YAML with `ddeharness protein-design validate --config <absolute-path> --json`.

Resolve MSA policy before file searches or MSA questions. In API mode, MSA is service-managed: do not search for local A3M files or offer disabling MSA. When adapting a local quickstart to the user's configured API mode, remove its local-only MSA switches from the task copy and use service-managed MSA; disclose this adaptation in the final plan. Do not silently discard user-supplied MSA files. If the mode is `service_default` and mode-specific choices matter, resolve that ambiguity rather than assuming local or API mode.

## Target discovery by Web search

When the user gives only a target name, such as “design a VHH against PD-L1,” actively search the Web for the missing target information instead of immediately asking the user to provide it. Prefer authoritative records: UniProt or NCBI for identity, species, isoform, domain boundaries, and sequence; RCSB PDB and primary literature for experimentally supported epitopes or interaction residues. Search results are evidence for configuration preparation, not launch authorization.

For every resolved value, retain the source URL or accession, sequence version or residue numbering scheme when available, and any domain truncation applied. Reconcile biological numbering with the one-based YAML positions explicitly. Never convert a reported epitope into `hotspots` until the isoform and numbering mapping are unambiguous. If sources disagree or several isoforms, extracellular constructs, or epitopes are plausible, present the alternatives rather than choosing silently. If no supported epitope is found, say so; use `hotspots: []` only after the user accepts an unconstrained target surface. Cache evidence within the conversation and do not repeat identical fetches without a specific missing fact; an example is not proof of an experimentally validated epitope.

Web search does not create local MSA or structure files. Verify required file references are accessible to the selected compute worker. For local folding only, if MSA policy is unresolved, combine these choices with other missing target information: disable MSA, provide existing paired/unpaired A3M paths, or explicitly authorize online search with `protein_design_search_target_msa`. Do not re-ask when the reviewed YAML or user has already settled the choice. Call the MSA tool only after authorization, pass target/antigen sequences only, and omit `compute_url` unless the reviewed design specifies one. Copy returned A3M paths into the target chain and bind the final YAML to the returned worker because those files live on that worker.

## Mandatory review gate

Starting a design consumes compute and fixes scientific choices. A request to prepare or launch an unseen configuration is not final launch approval. Review these categories, but do not turn them into three mandatory conversational rounds. Ask only unresolved choices, combine related questions, and retain confirmed answers. A complete example can proceed directly to one final plan and launch-confirmation question.

1. **Target / antigen** — target identity, sequence/provenance, domain boundaries, epitope/hotspot numbering, mode-compatible MSA policy, and any supplied initial complex.
2. **Binder** — design type, starting sequences, chain types, fixed/designable positions, and seed provenance. For antibodies, also review format (`VHH`, `scFv`, or `VH/VL`), CDRs and framework; for mini binders, require a complete single chain without CDR annotations. Reuse confirmed inputs instead of asking the user to pick a scaffold again.
3. **Final plan and launch** — after unresolved choices are settled, write the task YAML and run `ddeharness protein-design validate --config <absolute-path> --json`. Show a compact confirmation card: target/epitope, scaffold and fixed/designable regions, objective/default weights or overrides, cycles/candidates, inherited folding mode and MSA policy, configured compute placement (not a claimed bound worker), output location, PostFilter, and YAML path. State what was validated versus still untested. Ask explicitly: "Do you approve this configuration and authorize starting this design?" Do not end with a vague invitation to ask to start later.

Use `ask_user` for related unresolved questions or the final approval when available; otherwise ask in the normal reply and stop. Mark inferred values and sources. Do not repeat validation without changed input/configuration or a validation error. An absent reply, a generic earlier "run it," and validation success are not final approval. If the user changes the plan, validate the changed configuration and renew approval of that exact plan; do not reuse stale consent.

Call `protein_design_start` only after an explicit affirmative reply to the final launch review, and pass `user_confirmed: true`. If any required field is unresolved, contradictory, or unconfirmed, do not call the tool.

## Prepare a configuration

Use one-based sequence positions in YAML and agent-facing residue fields:
the first residue is 1, the last is the chain length, and ranges include both
endpoints. PDB/mmCIF author residue IDs need an explicit sequence mapping.

Resolve target identity, species, sequence, domain boundaries, epitope, binder format, and seed provenance from the user's input and available authoritative search tools. Use Web search whenever required target facts are absent, and cite the evidence in the staged review. Never fabricate a sequence, MSA path, structure path, hotspot, or chain mapping. Ask only when an unresolved ambiguity would materially change the experiment.

Write the generated configuration under the resolved user-home directory at `$HOME/.opendde_harness/protein_design/configs/` unless the user chooses another location, and pass its absolute path to validation and launch. Use OpenDDE for the primary `fold.model`. Preserve framework residues: `fixed_residues` overrides every design permission, while `cdr_regions` limits design to the named CDR residues when no narrower `designable_residues` is supplied.

Validate the file before launch using `ddeharness protein-design validate --config` followed by the actual, shell-quoted absolute YAML path. Never pass a documentation placeholder as a filename.

During review, report resolved values, provenance, assumptions, and unresolved choices. The final validated plan is not authorized for execution until the combined launch review is explicitly approved. Validation does not replace user confirmation.

## Run and control a task

Call `protein_design_start` only with the validated YAML path and `user_confirmed: true` after the mandatory review gate. The orchestration loop is deterministic: analyze, design, and reflection sessions make bounded decisions, while folding, scoring, population updates, and selection remain Python-controlled.

After successful launch, report the returned task ID, target, total cycles, compute backend, actual bound compute URL, and actual output directory. Do not present a default or guessed URL/path as an observed binding; if the response omits it, check task status and report any remaining unknown value honestly.

Use `protein_design_status` before changing a running task. Report both the optimization objective and its diagnostic components. A lower loss is better when `minimize` is true; a higher ipTM is better when ipTM is the direct objective. Do not interpret pLDDT as binding confidence.

Use `protein_design_adjust` only for the cycle-boundary fields
`num_sequences` and `reflection_interval`. Do not change the target, objective
direction, fold backend, framework sequence, or residue permissions during a
running task.

Use `protein_design_candidates` to inspect ranked candidates. Before recommending experimental sequences, require canonical amino acids, preserved framework residues, successful structure prediction, gate compliance, and metric provenance. Prefer a diverse shortlist over near-duplicate sequences with indistinguishable scores.

Use `protein_design_stop` when the user asks to stop or continuing would violate an explicit resource or safety constraint. The stop is cooperative and takes effect at a safe cycle boundary.
