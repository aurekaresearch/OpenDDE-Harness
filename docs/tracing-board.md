# Tracing Dashboard

Run `ddeharness tracing --port 4318`, open the printed URL and choose
**Protein design**. If another process occupies the requested port, the command
can choose the next free one. When the client and viewer run on a remote host,
forward the actual viewer port with SSH; for port 4318, use
`ssh -L 4318:127.0.0.1:4318 USER@HOST` (replace USER and HOST).

## Candidate workspace

- Choose a task by its target, timestamp and unique task ID.
- Browse candidates by cycle, sort/filter the table and select structures to compare.
  Scroll horizontally to see all recorded numeric metrics; the header stays aligned
  with the rows. Copy retrieves the complete sequence, not its shortened preview.
- Above the properties chart, choose **Color by → Age / cycle** (the default)
  to colour candidates by their creation cycle: older generations are orange and
  newer generations purple. This means generation, not elapsed wall-clock age;
  candidates from the same cycle have the same colour. **Last visible metric**
  restores metric-based colouring. The choice is independent of visible axes and
  is remembered in this browser across refreshes. It changes visualization only,
  never the running task, candidate scores, or selection.
- Sequence summaries show CDR labels directly. Hover for the complete sequence;
  the full sequence and structure viewer's sequence strip use matching purple CDR
  highlighting. CDR positions come from the
  task's saved, parsed YAML configuration; positions are zero-based and inclusive.
- The Mol* viewer displays the target and binder with distinct colours and
  supports design, chain, element, pLDDT and residue-order colouring. These
  structure colours are independent of the candidate-line colour selector.
- **Min ipAE** is the raw minimum predicted aligned error in Å across all
  configured binder–target residue pairs with valid alignment frames, considering
  both matrix directions. The frame mask follows the model's
  [chain-pair PAE calculation](https://github.com/bytedance/Protenix/blob/main/protenix/model/sample_confidence.py).
  Target–target, binder–binder and same-chain pairs are excluded. It is not the
  normalized mean `i_pae` loss, whole-complex PAE, or the distinct ipSAE score.
  The metric uses confidence-matrix indices mapped to the actual folded structure
  chains; missing or inconsistent confidence/mapping data remains unavailable
  with an explanation in `metadata.min_ipae`, never a fabricated zero. Historical
  runs without this recorded metric are not retroactively rewritten. A low
  minimum describes the best individual pair, not the entire interface, and does
  not replace contact/quality gates or alter the configured composite objective.
- Switch **View** to **Post-filter** for terminal ranking, per-candidate metrics and
  structures. If that stage has not produced data, the panel remains empty.
- Optional [PyRosetta analysis](pyrosetta.md) adds relaxed interface scores to the
  same numeric columns and comparison controls. The candidate's PyRosetta status
  tooltip explains failures and loss contributions. Structures in the viewer remain
  the original folds; the relaxed PDB is retained separately on compute.

## Design loop and search tree

The left panel shows Design → Fold → Reflection → Parent selection. Click a step
to inspect its recorded Input/Output and filter by cycle. Expand an event to read
its payload; Markdown text is rendered. A step can have no events yet.

The search tree sits to the right on wide screens. Small nodes and curved edges
show recorded parent–child relationships; names are hidden to avoid clutter.
Hover a candidate node for a structure, sequence and score preview; click to keep
the card open and drag its structure to rotate it. The sequence is expanded by
default, with highlighted CDRs. Cards choose a position within available space
and remain attached to the tree, rather than following the screen during scrolling.
Close the card with × or click another candidate to switch. Keyboard focus shows
lineage details; Enter selects a node.
Use +/−, Reset and dragging to navigate; ordinary scrolling moves through cycles,
while Ctrl/Command + wheel zooms the tree. This is **search lineage**, not a
sequence- or structure-derived phylogenetic tree. The view is capped at 2,000 nodes.

## Metric trends

Wide desktop layouts show up to eight charts in two rows of four; narrow screens
use two or one column. pTM is excluded from this section; iPTM remains available.
Metrics with no recorded series are omitted rather than filled with invented data.
Hover or focus a data point to synchronize the cycle across every metric:

- **Current cycle best**: metrics of that cycle's best-scored candidate.
- **Global best**: metrics of the global objective-best candidate recorded at that cycle.

The candidate is chosen by the configured objective, not by independently maximizing
each plotted metric. Therefore a component's global-best line need not be monotonic.
Missing values appear as **—**, never zero or a neighboring cycle's value.

## Refresh and troubleshooting

The page polls approximately every five seconds. Task selection, candidate selection,
open controls, loop step/cycle and scroll state are preserved for the same task.
Waiting events may show pending output; missing historical payloads cannot be
reconstructed by the viewer. For absent structures, inspect the task's artifacts and
compute access rather than interpreting an empty viewer as a folding failure.

Task state and worker logs default to
`~/.opendde_harness/protein_design/<task_id>/`, overridden by
`OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT`. Dashboard events and captured structures
live under `~/.opendde_harness/traces/logs/`, overridden by
`OPENDDE_HARNESS_TRACING_DIR`. Original structure files are on the compute host.
See [setup and monitoring](protein-design.md#monitor-a-design-task) and
[example configurations](examples/).
