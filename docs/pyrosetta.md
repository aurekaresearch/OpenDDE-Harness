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

`design.metric_loss_terms` accepts only the registered metric names above. Each
term requires `direction: minimize` or `maximize`; defaults are `weight: 1.0`,
`scale: 1.0`, and `reference: 0.0`. Weight must be finite and nonnegative, scale
finite and positive, and reference finite. Every nonzero term contributes:

```text
sign * weight * (raw_metric - reference) / scale
sign = +1 for minimize, -1 for maximize
total_loss = existing_confidence_and_ESM2_loss + sum(metric_contributions)
```

Raw metrics are never overwritten with normalized values. The existing loss
breakdown records raw value, normalized value, direction, scale, reference, weight,
and contribution for each enabled term, plus `metric_loss` and the composite formula
version. Positive terms require enabled analysis and loss optimization. At least
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

## Dashboard and validation

The candidate table shows recorded interface metrics with readable labels and
units; they are available for sorting, filtering, and comparison axes through the
existing metric selector. The analysis status beside each candidate has a tooltip
with errors, duration, and interface-loss contributions. Key interface scores are
accompanied by a contact-residue REU preview in the status tooltip. They are
prioritized in the eight-chart trend view. Post-filter results use fresh refold
metrics and status, never an earlier design's analysis. The structure viewer still
shows the original fold; the relaxed artifact resides on compute.

Run the offline unit tests and the real-process smoke tests:

```bash
uv run --extra pyrosetta pytest tests/test_pyrosetta_analysis.py tests/test_loss_objective.py tests/test_loss_confidence_scorer.py tests/test_protein_design_pyrosetta_metrics.py -q
uv run --extra pyrosetta pytest tests/integration/test_pyrosetta_process_smoke.py -m integration -q
node --test opendde_harness/tracing/viewer/test/*.test.js
```

To validate scientific execution on a provisioned compute host, set
`OPENDDE_TEST_PYROSETTA_PDB` to a representative complex and
`OPENDDE_TEST_PYROSETTA_CHAINS` to JSON of the form
`{"sequences":{"A":"TARGET_SEQUENCE","B":"BINDER_SEQUENCE"},"binder":["B"],"target":["A"]}`,
then run:

```bash
uv run --extra pyrosetta pytest tests/integration/test_pyrosetta_real_backend.py -m integration -q
```

This test deliberately skips when the licensed backend or the supplied complex is
absent. It runs two copies in separate workers with one relaxation repeat each,
checks deterministic scores and residue-energy arithmetic, and preserves the
source. Production defaults use five repeats. A passing process test alone does
not validate scientific scores.
