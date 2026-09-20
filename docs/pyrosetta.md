# PyRosetta relaxation and interface scoring

PyRosetta analysis is opt-in and runs in the **compute service's Python
environment**, including when folding uses the remote OpenDDE API. With managed
Docker compute, that environment is inside the worker container. The client
does not import it.

## Installation

PyRosetta is **not Apache-2.0**: its downloads have a separate
[non-commercial license; commercial use requires a separate license](https://www.pyrosetta.org/downloads).
Verify your usage rights before installation or redistribution.

### Docker-managed compute

For the **Local Linux Docker environment** selected during onboarding, follow the
[optional runtime image build and verification](../docker/README.md#optional-pyrosetta-runtime).
That procedure builds `private/opendde-harness:pyrosetta`, checks its label and
PyRosetta import, and selects it with `OPENDDE_HARNESS_COMPUTE_IMAGE` during
onboarding. The default shared image does not contain PyRosetta. Installing the
extra only in the client's or host's `.venv` does not provision the container.

For work on a fork, use an [editable client](installation.md#develop-from-an-editable-checkout)
and [mount the development checkout](installation.md#use-a-development-checkout-for-local-compute).
Let active tasks finish before switching worker code or images. Neither building
the image nor enabling analysis in a new YAML changes an already running task.

### Verify the selected runtime

First inspect the accepted task's `workflow.json`, `snapshot.json` and `worker.log`
under its task directory. Check task `compute` overrides, the saved worker
registry/profile, and the application configuration passed with
`--opendde-config`. The local Docker image is irrelevant if the task selected a
different registered/explicit service. Ports are runtime details, not identities.

For managed Docker, `ddeharness doctor --compute-only` reports the current local
container and code ID; `~/.opendde_harness/compute/local.json` records that instance.
Match it to the task's resolved endpoint before inspecting it. On the compute host,
replace `CONTAINER_NAME` with that verified running container:

```bash
docker inspect --format '{{.Id}} {{.Image}} {{.Config.Image}}' CONTAINER_NAME
docker inspect --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' CONTAINER_NAME
docker top CONTAINER_NAME -eo pid,args
docker exec CONTAINER_NAME /opt/runtime/bin/python -c \
  'import sys, subprocess; print(sys.executable); subprocess.run([sys.executable, "-c", "import pyrosetta; print(pyrosetta.version())"], check=True)'
```

The standard image uses `/opt/runtime/bin/python`; for a custom service, substitute
the interpreter actually launching its analysis children. Confirm it against the
recorded `metadata.pyrosetta.provenance.python_executable`, worker-module path and
runtime environment from a scored candidate. Retain the immutable image ID and
mounted code identity, not just its mutable tag. An import in a newly created
container or host shell is not proof that the selected worker uses that environment.

If the wrong image/code is selected, preserve the failed task and fix routing or
saved settings before submitting a new trial. Reuse a working compatible image.
Do not install into the host to repair a container, replace a busy worker, or call
`compute stop --force` as a readiness check. An occupied worker is a reason to wait
for availability while performing independent checks, not to interrupt its run.

### Host-native compute service

If the compute service runs directly on the host, install into its Python
environment. From its source checkout, using a recent uv with flat-index support:

```bash
uv sync --extra pyrosetta
uv run --extra pyrosetta python -c 'import pyrosetta; print(pyrosetta.version())'
```

This installs pinned quarterly release `2026.29+releasequarterly.80a0635615` and
Biotite for mmCIF conversion. `pyproject.toml` binds only PyRosetta to the official
quarterly flat index; other dependencies continue to use PyPI. The lockfile keeps
platform-specific wheel URLs (the upstream index does not supply wheel hashes).
The wheel download is about 1.6 GiB on Linux x86-64. Supported wheels cover
CPython 3.12–3.14 on Linux x86-64 and macOS x86-64/ARM64; unsupported hosts need
a compatible separately provisioned build. Keep `--extra pyrosetta` on subsequent
`uv run`/`uv sync` commands so uv does not remove the optional backend. Add
`--extra protein-design` when also provisioning the existing prediction dependencies.

For a development service that also needs prediction dependencies, preserve both
extras while syncing or running commands:

```bash
uv sync --locked --extra dev --extra protein-design --extra pyrosetta --dev
uv run --extra protein-design --extra pyrosetta python -c 'import pyrosetta; print(pyrosetta.version())'
```

Plain `make install` does not retain the PyRosetta extra; use the explicit sync
above for this environment. A Docker-managed client can continue using
`make install` because its scientific dependencies live in the image.

## Workflow configuration

Add these sections to a complete [design YAML](protein-design-yaml.md):

```yaml
fold:
  pyrosetta:
    enabled: true
    max_workers: 4
    timeout_seconds: 900.0
    relax_repeats: 5
    max_iter: 200
    constrain_to_start: true
    pack_separated: true
    contact_distance: 5.0
    seed: 42
    on_failure: fail

design:
  optimization_metric: loss
  metric_loss_terms:
    rosetta_interface_dg:
      direction: minimize
      weight: 0.5
      scale: 10.0
      reference: 0.0
    rosetta_interface_sc:
      direction: maximize
      weight: 0.2
      scale: 1.0
      reference: 0.0
```

The loss weights above are illustrative, not calibrated biological thresholds.
Existing `design.loss_weights` and their partial-override semantics are unchanged.
Omitting both new settings preserves the previous objective and requires no
PyRosetta installation. To collect metrics without changing ranking, enable
analysis and omit `metric_loss_terms`; this also works with `optimization_metric: iptm`.

## Configuration and execution

All options below are under `fold.pyrosetta`. Unknown keys, string booleans,
invalid ranges, and non-finite values are rejected before prediction.

| Option | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Run analysis after successful folds and terminal refolds. |
| `max_workers` | `4` | Concurrent child processes per fold batch, capped by CPU affinity and batch size; integer 1–128. Reduce for memory-limited hosts. |
| `timeout_seconds` | `900.0` | Positive wall-clock limit for each child, including imports and conversion; expiry kills and reaps the child. |
| `relax_repeats` | `5` | FastRelax repeats, integer 1–100. Relaxation cannot be bypassed. |
| `max_iter` | `200` | Positive maximum iterations per minimization stage, not a total protocol iteration count. |
| `constrain_to_start` | `true` | Restrain coordinates to the input throughout relaxation, with coordinate-constraint weight 1.0. |
| `pack_separated` | `true` | Repack separated partners during interface-energy calculation. |
| `contact_distance` | `5.0` | Maximum minimum heavy-atom distance between partners for contact-residue reporting, in Å; positive and at most 20. This does not change the original-fold contact gate or InterfaceAnalyzer's neighbor-based interface selection. |
| `seed` | `42` | Per-process Rosetta random seed, integer 1–2147483647; independent of fold/root seeds. |
| `on_failure` | `fail` | `fail` rejects a candidate after analysis failure; `continue` retains confidence scoring unless the loss requires missing interface metrics. |

The existing compute fold operation orchestrates the batch. A bounded thread pool
supervises independent Python subprocesses; each child initializes PyRosetta once
and owns one pose. This avoids sharing Rosetta state across threads or forking a
loaded GPU runtime. Rosetta and BLAS/OpenMP workers use one thread per process.
Results retain candidate order even when completion order differs. Concurrent
compute jobs each have their own limit; provision their aggregate CPU and memory
budget accordingly. The existing fold GPU lease remains held until scoring finishes.

Each child runs torsional **FastRelax with ref2015**, moving backbone and side-chain
torsions with rigid-body jumps fixed. Sequence identity is checked before and after
relaxation. Coordinate restraints are removed before unconstrained ref2015 scoring
and **InterfaceAnalyzer**, which uses the relaxed pose. InterfaceAnalyzer does not
repack the bound input again; separated-state repacking follows the option above.

The analyzed structure is the first selected structure from the fold result, the
same one used for confidence-loss geometry. Additional diffusion samples are not
independently analyzed. All configured binder chains form one partner and all
target chains form the other (`HL_A`, for example). Chain IDs must be distinct,
single ASCII letters or digits. PDB and mmCIF inputs are supported; mmCIF is
converted in the child without modifying the source. No ambiguous chain remapping
is performed. Missing/mismatched sequences or absent buried surface fail explicitly.

The original fold remains `candidate.structure_path`, so confidence scores, contact
gates, and the normal structure viewer continue to describe that fold. The relaxed
PDB is saved separately under the task's fold/refold analysis directory and linked
in `candidate.metadata.pyrosetta.relaxed_structure_path`. Each invocation uses a
unique directory; retries and refolds do not reuse stale analysis. Logs and worker
results are saved next to the relaxed structure.

## Metrics and loss

All eight scores are ordinary finite entries in `Candidate.metrics`, returned by
the existing compute API and persisted through population/history/tracing. They are
not placeholders when analysis is disabled or unavailable.

| Metric | Definition / units |
|---|---|
| `rosetta_total_score` | Unrestrained ref2015 score of the relaxed complex, Rosetta energy units (REU). |
| `rosetta_interface_dg` | InterfaceAnalyzer separated-state binding score, REU. Lower is typically favored. |
| `rosetta_interface_sasa` | Buried interface surface area (dSASA), Å². |
| `rosetta_interface_dg_per_sasa` | `100 * interface_dg / interface_sasa`, 100×REU/Å². |
| `rosetta_interface_sc` | Shape complementarity, dimensionless. Higher is typically favored. |
| `rosetta_interface_hbonds` | Number of cross-interface hydrogen bonds. |
| `rosetta_interface_unsat_hbonds` | InterfaceAnalyzer change in buried unsatisfied hydrogen bonds. |
| `rosetta_interface_residues` | Number of interface residues across both partners. |

These are modeling scores, not measured binding free energies. Packstat is not
computed. Metric definitions follow [Rosetta InterfaceAnalyzer](https://docs.rosettacommons.org/docs/latest/application_documentation/analysis/interface-analyzer)
and relaxation follows [FastRelax](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/FastRelaxMover).
REU is an arbitrary, protocol-dependent energy scale, not kcal/mol or kJ/mol;
experimental conversion requires a suitable benchmark rather than a universal
factor. See [Rosetta's units documentation](https://docs.rosettacommons.org/docs/latest/rosetta_basics/Units-in-Rosetta).
The factor of 100 in `rosetta_interface_dg_per_sasa` is part of that metric's
definition, not a conversion to physical energy or a normalized confidence score.

### Per-contact-residue REU evidence

`candidate.metadata.pyrosetta.contact_residues` contains both binder and target
residues whose relaxed heavy atoms contact the other partner within `contact_distance`.
Each record includes `chain_id`, zero-based within-chain `residue_index` (the same
indexing used for mutations), `amino_acid`, `partner`, PDB residue number and
insertion code, minimum partner distance in Å, and these finite REU scores:

- `bound_score_reu`: InterfaceAnalyzer's full weighted ref2015 residue score in
  the relaxed bound complex, including intra-partner interactions; this is not a binding energy.
- `separated_score_reu` and `interface_dg_reu`: InterfaceAnalyzer's per-residue
  separated score and bound-minus-separated difference. All three energies use
  the same InterfaceAnalyzer decomposition, with `bound - separated == interface_dg`.

The selected parent's evidence is included explicitly in the sequence-design
prompt, and every analyzed candidate's status/scores reach next-cycle feedback.
Fixed positions remain immutable regardless of energy. These diagnostics are not
mutation-effect predictions and are not directly selectable scalar loss terms;
use the registered aggregate metrics below for composite loss. The full list is
persisted with the candidate and in the worker result; dashboard trace metadata
retains its existing 64-item list limit and shows the displayed/total count.

`design.metric_loss_terms` accepts the registered metric names above plus
`min_ipae` and `ipsae` described below. Each
term requires `direction: minimize` or `maximize`; defaults are `weight: 1.0`,
`scale: 1.0`, and `reference: 0.0`. Weight must be finite and nonnegative, scale
finite and positive, and reference finite. Every nonzero term contributes:

```text
sign * weight * (raw_metric - reference) / scale
sign = +1 for minimize, -1 for maximize
base_loss = structure_loss + esm2_contribution
metric_loss = sum(metric_contributions)
loss = base_loss + metric_loss
```

Raw metrics are never overwritten with normalized values. The existing loss
breakdown records raw value, normalized value, direction, scale, reference, weight,
and contribution for each enabled term, plus the original `base_loss`, `metric_loss`
and composite `loss`. Selection minimizes `loss`, not `base_loss`. Older artifacts
without `base_loss` can reconstruct it as `structure_loss + esm2_contribution`.
The formula version is unchanged because exposing the subtotal does not change
the arithmetic. Positive PyRosetta terms require enabled analysis; all positive
metric terms require loss optimization. At least
one legacy loss coefficient must remain positive; standalone interface-only ranking
is not supported. A zero-weight metric need not exist. Missing, null, nonnumeric,
non-finite, or overflowing enabled terms fail scoring; no candidate-specific
renormalization or zero substitution occurs.

Analysis metadata records `success`, `failed`, or `skipped`, a diagnostic error,
elapsed time, source/relaxed paths, configuration, units, and PyRosetta version.
Disabled analysis adds no metadata or scores. Import errors, native crashes,
timeouts, malformed worker results, and invalid metrics publish no partial scores.
With `on_failure: fail`, failed candidates have no objective and cannot enter the
population. With `continue`, confidence-only ranking can proceed and the analysis
error remains visible. Missing loss-required metrics still reject the candidate.
If every candidate fails, the existing batch failure/retry policy applies.

### Raw confidence metric loss terms

Two optional confidence terms use the same composite objective, bounded groups,
persisted contribution breakdown, search selection, and terminal-refolding path:

| Key | Required direction | Raw value |
| --- | --- | --- |
| `min_ipae` | `minimize` | Minimum valid raw PAE in either directed binder–target block, in Å. This is **not** the normalized mean structural component `i_pae`. |
| `ipsae` | `maximize` | Existing fold-backend ipSAE, dimensionless in [0, 1], taking the maximum over directed chain pairs. |

Neither term is enabled by default, and neither supplies universal good/bad
anchors. In `bounded_grouped` mode, every positive-weight term must have explicit
`loss_combination.anchors` and exactly one group assignment. Anchor order must
match the required direction. Choose weights/group budgets with the overlapping
PAE and interface-confidence information in mind; adding terms should not
implicitly increase a group's budget.

These terms do not require PyRosetta, but do require `fold.need_atom_confidence`
to remain enabled. ipSAE uses the configured `ipsae_pae_cutoff` and
`ipsae_dist_cutoff` (both default to 10); changing these changes its meaning.
For multichain complexes the reported maximum can involve a chain pair other
than the intended binder–target interface; it is not a binder-specific aggregate.

Missing Min ipAE rejects a candidate when the term has positive weight. As an
explicit exception, uncomputable ipSAE uses **0** for reporting and scoring.
`metadata.ipsae` records `status: unavailable`, `value: null`, and
`fallback_value: 0.0`; a genuinely computed zero records `status: success`.
The fold log records calculation errors. Refolding recomputes both metrics and
replaces stale scores and availability metadata. Nonfinite or out-of-domain
metric inputs are not accepted by the objective.

### Interpreting linear-objective magnitude and choosing scales

The objective is a signed ranking score, not a probability or nonnegative error.
A favorable negative interface-energy contribution can cancel positive confidence
and unsatisfied-H-bond contributions. Small or negative totals are therefore valid;
do not take the absolute value, clip the total to zero, or rescale accepted runs
merely to make the displayed number larger. Compare raw metrics, the original loss,
each contribution, and the hard-gate evidence together.

For example, a real three-candidate workflow with unchanged five-repeat relaxation
produced the following decomposition. Values are rounded for this table; persisted
artifacts retain full precision. Coefficients were `0.5 / 10` for interface dG,
`-1 / 1` for shape complementarity relative to `0.5`, and `0.25 / 5` for unsatisfied
H-bonds. These observations exercise the formula; they are not calibrated defaults.

| Candidate | Original loss | dG contribution | Shape contribution | Unsatisfied-H-bond contribution | Composite loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Framework | 2.37834928 | -3.41108258 | -0.04806668 | 1.10000000 | 0.01920002 |
| Mutation 001 | 2.02137372 | -2.49241568 | -0.11578742 | 0.65000000 | 0.06317062 |
| Mutation 002 | 2.04921697 | -2.94838087 | -0.10244870 | 0.95000000 | -0.05161259 |

Mutation 001 has the lower original loss, but mutation 002 has the lower composite
and was ranked first. Across these three candidates, contribution ranges were
`0.35698` for the original loss, `0.91867` for dG, `0.06772` for shape complementarity
and `0.45` for unsatisfied H-bonds. Thus dG had the largest observed range after
scaling. This small, related sample does not establish whether that influence is
scientifically desirable.

The current linear objective has sensitivity `sign * weight / scale` to each raw
metric. A shared reference changes every candidate by the same constant, so it
changes the displayed zero point but not ranking. Increasing scale reduces a term's
influence; multiplying weight and scale by the same factor changes nothing.
Weight normalization alone cannot correct unequal raw units, and an unbounded
linear term can still dominate outside the range used to choose its scale.

### Opt-in bounded objective with fixed group budgets

`design.loss_combination` opts into `bounded-fixed-grouped-v1`; omitting it preserves
the existing linear formula and version. No universal ranges or group budgets are
provided. Supply a nonempty `calibration_id`, explicit fixed anchors for **every**
enabled structural, ESM2 and metric term, and a complete, nonoverlapping grouping.
This separates numerical range from deliberate scientific priorities:

For a reusable, explicitly selected 40/40/20 policy, see
[default bounded loss v1](examples/loss_presets/README.md). It is a loss-policy
fragment with documented weights and anchors, not an implicit application default
or a standalone workflow.

1. Transform each enabled loss component into a dimensionless penalty using fixed,
   documented good/bad anchors: `p = clip((raw - good) / (bad - good), 0, 1)`.
   For a quantity to minimize, `good < bad`; for one to maximize, `good > bad`.
   Anchors are scoring preferences, not replacements for hard eligibility gates.
2. Group related evidence, for example confidence/geometry, ESM2 sequence plausibility,
   interface energy, and interface geometry/H-bonds. Normalize positive weights within
   each group, and assign explicit positive group budgets whose sum is one.
3. Minimize `sum(group_budget * sum(normalized_term_weight * penalty))`.
   The total lies in `[0, 1]`; a group's maximum possible contribution is its budget.
   Adding correlated energy descriptors cannot implicitly expand that group's budget.
   This bounds influence caused by numerical range, while making remaining tradeoffs explicit.

Mapping individual responses to a common `[0, 1]` preference scale has precedent in
the [NIST desirability approach](https://www.itl.nist.gov/div898/handbook/pri/section5/pri5322.htm).
The weighted group-budget sum above is this application's design, not NIST's
geometric-mean aggregation or an experimentally validated binding model.

The following is a complete illustrative loss-policy section, **not calibrated
scientific defaults**. Retain the complete target/binder/fold configuration and
enable `fold.pyrosetta.enabled`. All ten default confidence/ESM2 coefficients remain
enabled here; their positive values become relative weights within their groups.

```yaml
design:
  optimization_metric: loss
  metric_loss_terms:
    rosetta_interface_dg: {direction: minimize, weight: 1.0}
    rosetta_interface_sc: {direction: maximize, weight: 1.0}
    rosetta_interface_unsat_hbonds: {direction: minimize, weight: 1.0}
  loss_combination:
    mode: bounded_grouped
    calibration_id: example-only-not-scientifically-calibrated
    groups:
      confidence:
        budget: 0.25
        terms: [plddt, i_plddt, pae, i_pae, i_ptm, con, i_con, rg, dgram_cce]
      sequence:
        budget: 0.25
        terms: [esm2]
      interface_energy:
        budget: 0.25
        terms: [rosetta_interface_dg]
      interface_geometry:
        budget: 0.25
        terms: [rosetta_interface_sc, rosetta_interface_unsat_hbonds]
    anchors:
      plddt: {good: 0.0, bad: 1.0}
      i_plddt: {good: 0.0, bad: 1.0}
      pae: {good: 0.0, bad: 1.0}
      i_pae: {good: 0.0, bad: 1.0}
      i_ptm: {good: 0.0, bad: 1.0}
      con: {good: 0.0, bad: 10.0}
      i_con: {good: 0.0, bad: 10.0}
      rg: {good: -1.0, bad: 1.0}
      dgram_cce: {good: 0.0, bad: 2.0}
      esm2: {good: 0.0, bad: -5.0}
      rosetta_interface_dg: {good: -100.0, bad: 0.0}
      rosetta_interface_sc: {good: 1.0, bad: 0.0}
      rosetta_interface_unsat_hbonds: {good: 0.0, bad: 30.0}
```

Structural anchors apply to the scorer's **loss components**, not raw pLDDT or PAE
measurements: all structural components are minimized. `esm2` anchors apply to raw
mean PLL and are maximized. Metric directions come from `metric_loss_terms`.
Good/bad orientation must match the direction; finite anchors must have distinct,
finite spans. Disabled zero-weight terms must not have anchors or group assignments.
Every enabled term must appear exactly once. Unknown fields, incomplete coverage,
invalid budgets and nonfinite values fail validation before folding.

For component `i` in group `g`, the recorded contribution is
`budget_g * weight_i / sum(group_weights) * penalty_i`. In bounded mode metric
`reference` and `scale` no longer normalize the selection score: fixed anchors
explicitly supersede them. Their original meaning is retained only in the linear
audit breakdown. Changing a raw measurement's units together with its anchors
leaves its bounded contribution unchanged.

The persisted top-level `structure_loss`, `esm2_contribution`, `base_loss`,
`metric_loss` and `loss` are now **dimensionless bounded contributions**. Selection
uses this top-level `loss`. `original_base_loss`, `legacy_loss`, and the complete
`legacy_loss_breakdown` retain the original signed linear calculation for audit,
never for selection. Each bounded component records raw value, good/bad anchors,
penalty, group, original and normalized weights, budget, effective weight and
contribution; ESM2 has the same fields under `esm2_component`. Group totals and the
resolved calibration configuration are persisted too. Missing required metrics
still fail; weights are never renormalized around absent measurements.

Both the bounded score and its preserved linear audit must remain finite. An
astronomically large but finite input whose legacy arithmetic overflows is rejected
with a `legacy ... must be finite` error, not published with an incomplete audit.
Hard scientific gates and quality checks remain separate from the objective and
cannot be bypassed by favorable bounded scores. Workload stages and runtime
adjustments do not permit changing anchors or budgets during a run.

| Approach | Benefit | Limitation |
| --- | --- | --- |
| Existing fixed reference/scale, linear sum | Transparent raw sensitivity; no saturation | Unbounded contributions; requires justified scales |
| Fixed bounded penalties and group budgets | Explicit maximum influence; interpretable good/bad anchors | Clipping loses distinctions beyond anchors and may create ties |
| Smooth bounded sigmoid with fixed anchors | Retains graded changes beyond anchors | Saturates in the tails; less direct threshold interpretation |
| Geometric desirability or worst-component objective | Penalizes an especially weak component strongly | Introduces a veto/bottleneck preference, potentially duplicating hard gates |

Do not estimate scales or good/bad anchors from each current batch, cycle or
surviving population: the same candidate would acquire different scores as its
neighbors changed. Freeze and version calibration metadata before a run, together
with the fold backend, Rosetta protocol, model versions and target/scaffold scope.
Keep every enabled component required; missing or nonfinite values must fail
scoring rather than trigger candidate-specific weight renormalization.

Before proposing defaults, use an independent representative calibration set with
multiple targets, scaffolds and repeated seeds, then evaluate held-out rankings,
gate pass rates, correlations between metrics, clipping frequency and sensitivity
to weights/anchors. Where biological claims are intended, validate against relevant
experimental outcomes. Apply this procedure to the existing structural and ESM2
components as well as the added interface metrics. Three related candidates can
demonstrate cancellation, sensitivity and ranking behavior, but cannot justify
universal anchors, group budgets, or conversion of REU into experimental affinity.

## Dashboard and validation

The candidate table shows recorded interface metrics with readable labels and
units; they are available for sorting, filtering, and comparison axes through the
existing metric selector. The analysis status beside each candidate has a tooltip
with errors, duration, and interface-loss contributions. Key interface scores are
accompanied by a contact-residue REU preview in the status tooltip. They are
included in the metric trend view. Post-filter results use fresh refold
metrics and status, never an earlier design's analysis. The structure viewer still
shows the original fold; the relaxed artifact resides on compute.

Run the offline unit tests and process smoke tests from a development checkout.
These checks do **not** require the licensed PyRosetta extra:

```bash
uv run --extra dev pytest tests/test_pyrosetta_analysis.py tests/test_loss_objective.py tests/test_bounded_loss.py tests/test_loss_presets.py tests/test_confidence_metric_loss.py tests/test_loss_confidence_scorer.py tests/test_protein_design_pyrosetta_metrics.py -q
uv run --extra dev pytest tests/integration/test_pyrosetta_process_smoke.py -m integration -q
make test-dashboard
```

To validate scientific execution on a provisioned compute host, set
`OPENDDE_TEST_PYROSETTA_PDB` to a representative complex and
`OPENDDE_TEST_PYROSETTA_CHAINS` to JSON of the form
`{"sequences":{"A":"TARGET_SEQUENCE","B":"BINDER_SEQUENCE"},"binder":["B"],"target":["A"]}`,
then run:

```bash
uv run --extra dev --extra protein-design --extra pyrosetta pytest tests/integration/test_pyrosetta_real_backend.py -m integration -q
```

This test deliberately skips when the licensed backend or the supplied complex is
absent. It runs two copies in separate workers with one relaxation repeat each,
checks deterministic scores and residue-energy arithmetic, and preserves the
source. Production defaults use five repeats. A passing process test alone does
not validate scientific scores.

If running tests from a host-native service environment that already uses optional
extras, retain them on every `uv run`/`uv sync` invocation, or use a separate test
environment. Do not synchronize the Python environment of an active service.

### End-to-end verification checklist

An image build, successful import, valid YAML, accepted submission or skipped
integration test does not verify the complete workflow. Use a separately named
copy of an accepted target/scaffold configuration; preserve the original and its
logs. Do not weaken gates to make a trial pass.

1. **Review the reduced workload.** For example, set `design.n_cycles: 1`,
   `design.num_sequences: 2` and `design.population_size: 2`. Review/remove
   workload-only `cycle_schedule` overrides in the copy, since they can override
   those counts or extend beyond the new cycle limit. Initial binder scoring,
   bootstrap/redesign behavior and retries also consume work. For bounded terminal
   verification, use `design.post_refold_filter` with `enabled: true`,
   `max_parents: 1`, `samples_per_parent: 8`, `survivors_per_parent: 2`, and
   `top_k: 1`. These are workload examples, not scientific defaults or a runtime
   guarantee; hard gates may leave fewer than two eligible competitors.
2. **Keep the intended science.** Preserve target, scaffold, fixed/CDR masks,
   hotspots/no-hotspots, MSA settings, contact/quality gates and production
   relaxation settings. Keep PyRosetta enabled, `on_failure: fail`, loss
   optimization and at least one nonzero Rosetta loss term. For bounded scoring,
   review every active anchor and group budget, or explicitly apply the reusable
   preset. Do not add dummy measurements for missing required terms.
3. **Validate, then actually launch.** Use the same application config for both
   commands. The following `start` authorizes compute and provider usage, without
   another interactive confirmation:

   ```bash
   ddeharness protein-design validate --config reviewed-bounded-trial.yaml --json
   ddeharness protein-design start --config reviewed-bounded-trial.yaml --json
   ```

4. **Record the task ID and follow it to terminal status.** Use the task-status
   tool/TUI and the tracing dashboard, plus `snapshot.json` and `worker.log` in
   the task directory. Confirm that the newly submitted task reached the intended
   worker. On failure, inspect the actual traceback and artifacts, correct the
   cause, and submit a new uniquely named trial when necessary. Never overwrite
   the failed run to disguise the failure.
5. **Audit real scores and selection.** Retain candidate IDs, raw Rosetta metrics,
   original loss, every configured metric contribution and composite loss. For
   bounded scoring, independently recompute penalties and effective weights from
   raw values/anchors; include the separately stored `esm2_component`. Check the
   sum against `metrics.loss` and `candidate.objective`. Check population and
   terminal ordering use that objective after hard gates, not an LLM advisory
   rank, fold ranking score, or original loss. A competing-candidate example is
   stronger evidence than selecting the only eligible candidate.
6. **Check display and artifacts.** Compare dashboard raw values, loss details and
   selected candidates with persisted results. Retain the trial/resolved workflow,
   worker identity, log, confidence files, original folds, relaxed PDBs, analysis
   JSON and terminal decisions. Report a failed enabled terminal stage as failure,
   not success based only on completed search cycles.

These checks establish execution and arithmetic correctness, not experimental
binding affinity or biological suitability. Calibrate the policy on representative
data before making biological claims; a small passing trial is not calibration.
