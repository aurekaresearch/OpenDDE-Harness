# Reusable loss policies

## Default bounded loss v1

[`default_bounded_v1.yaml`](default_bounded_v1.yaml) records a user-selected
40% confidence / 40% energy / 20% geometry-sequence policy. It is an opt-in,
target-independent configuration fragment, not a standalone workflow or an
automatic change to the application's built-in defaults. Its anchors and weights
are preferences, not a universally calibrated model of binding affinity.

For each term, form a penalty `p = clip((value - good) / (bad - good), 0, 1)`.
All `p` values below are these penalties, not raw metric values. Minimize:

```text
L = 0.4 * Confidence + 0.4 * Energy + 0.2 * GeometrySequence

Confidence = 0.50 * p(ipSAE) + 0.25 * p(ipTM)
           + 0.125 * p(binder pLDDT) + 0.125 * p(CDR pLDDT)

Energy = 0.25 * p(interface deltaG) + 0.25 * p(shape complementarity)
       + 0.50 * p(100 * interface deltaG / buried SASA)

GeometrySequence = 0.50 * p(ESM2) + 0.25 * p(i_con)
                 + 0.125 * p(rg) + 0.125 * p(con)
```

| Group | Term / exact key | Final weight | Good -> bad on the human-readable scale |
| --- | --- | ---: | --- |
| Confidence | ipSAE / `ipsae` | 20% | 1 -> 0 |
| Confidence | ipTM / `i_ptm` | 10% | 1 -> 0 |
| Confidence | Binder pLDDT / `plddt` | 5% | 100 -> 0 |
| Confidence | Designed-CDR pLDDT / `i_plddt` | 5% | 100 -> 0 |
| Energy | Interface deltaG / `rosetta_interface_dg` | 10% | -200 -> 0 REU |
| Energy | Shape complementarity / `rosetta_interface_sc` | 10% | 1 -> 0 |
| Energy | Energy per area / `rosetta_interface_dg_per_sasa` | 20% | -10 -> 0 (100 * REU / square angstrom) |
| GeometrySequence | ESM2 / `esm2` | 10% | 0 -> -5 mean log-probability |
| GeometrySequence | Paratope contact loss / `i_con` | 5% | 0 -> 10 |
| GeometrySequence | Expansion penalty / `rg` | 2.5% | 0 -> 1 |
| GeometrySequence | Internal binder contact loss / `con` | 2.5% | 0 -> 10 |

The YAML coefficients sum to 0.4, 0.4, and 0.2 within the respective groups. The
scorer divides each coefficient by its group's sum, then multiplies by the group
budget. Consequently the final weights are exactly those in the table, summing
to 1, rather than being multiplied by the budgets a second time. The loss lies
in [0, 1], with lower being better. Group budgets cap contributions; they do not
eliminate correlations or guarantee equal sensitivity over observed data ranges.

All other currently supported terms have explicit zero coefficients: `pae`,
`i_pae`, `dgram_cce`, `min_ipae`, `rosetta_total_score`, `rosetta_interface_sasa`,
`rosetta_interface_hbonds`, `rosetta_interface_unsat_hbonds`, and
`rosetta_interface_residues`. Zero-weight terms do not participate in the
objective; they are not necessarily excluded from upstream measurement. Min ipAE
is available but not selected for this default. It has no assumed scoring bounds.

### Applying the policy

In a source checkout, the policy is at
`docs/examples/loss_presets/default_bounded_v1.yaml`. Releases containing this
feature also ship the YAML and this README under
`opendde_harness/plugin/protein_design/examples/loss_presets/`. To read the installed
copy using the Python interpreter from the environment containing the package:

```python
from importlib.resources import files

preset = files("opendde_harness").joinpath(
    "plugin/protein_design/examples/loss_presets/default_bounded_v1.yaml"
)
print(preset.read_text(encoding="utf-8"))
```

Older releases do not contain this resource. Check your installed revision rather
than substituting a similarly named file from a different release.

Start with a complete workflow containing the intended target, binder/scaffold,
scientific gates, workload, provider and compute settings. Replace these four
fields under its existing `design` section with the preset's full values:

- `optimization_metric`
- `loss_weights`
- `metric_loss_terms`
- `loss_combination`

Merge the preset's `fold` fields into the existing `fold` section, including
merging its `pyrosetta` fields without dropping other analysis settings. Preserve
the other design/fold fields and all other workflow sections. Replace the loss
mappings as complete mappings; do not recursively merge old anchors/groups or
leave old positive coefficients behind. There is no YAML `include`, `extends`,
or named-preset loader syntax: the resulting complete workflow must explicitly
contain these settings before normal configuration validation and submission.

The policy enables PyRosetta and requires confidence output. It does not choose
a compute endpoint/image, change scientific gates, or submit a run. The selected
worker must already provide PyRosetta and the other configured model assets.

Save the merged workflow to a new filename, then validate it:

```bash
ddeharness protein-design validate --config reviewed-bounded-trial.yaml --json
```

Use `--opendde-config /path/to/config.json` as well if your application uses a
non-default provider/compute configuration. Validation checks configuration, not
worker imports or scientific execution. Complete the
[real-run checklist](../../pyrosetta.md#end-to-end-verification-checklist) before
scaling up. In a wheel, consult the corresponding `opendde_harness/docs/pyrosetta.md`
resource or the repository documentation for that same release.

### Metric semantics and missing values

- The two pLDDT components and `i_ptm` are stored as `1 - confidence`, with raw
  confidence on [0, 1]. Their stored anchors are therefore good=0, bad=1.
- `rg` retains the approved expansion-only bounds. Values at or below the
  reference radius receive zero penalty, not an extra reward for shrinking.
- `rosetta_interface_dg_per_sasa` includes the factor of 100; its denominator is
  buried SASA summed across both partners, not half that area.
- ipSAE uses explicit PAE and distance cutoffs of 10 angstrom. The current
  implementation takes the maximum across directed chain pairs. Distance is a
  whole-chain-pair rejection check, not an individual PAE-entry mask. Review
  this aggregation before reusing it for a multichain complex.
- By user choice, unavailable ipSAE may use raw 0, giving full ipSAE penalty and
  a 0.2 contribution. Retain `status: unavailable` and fallback metadata so it
  remains distinct from a genuinely measured zero. Other required metrics are
  not allowed to disappear silently. PyRosetta failure remains an error.

To change a weight to zero, also remove that term's active anchor and group
assignment. Empty groups are invalid. If disabling a whole group, explicitly
choose new positive budgets summing to one; do not silently redistribute them.
Use a new calibration identifier when changing anchors or weighting policy so
persisted results remain distinguishable. Old task configurations are untouched.
