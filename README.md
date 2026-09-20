# OpenDDE Harness


[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
![Status](https://img.shields.io/badge/status-preview-orange)
[![PyPI](https://img.shields.io/pypi/v/opendde-harness.svg?cacheSeconds=300)](https://pypi.org/project/opendde-harness/)


![tui](docs/assets/tui.png)

Harness for agentic antibody design: prepare targets, optimize CDR sequences, predict structures with OpenDDE, and inspect results. Supports VHH, scFv and paired VH/VL binders while preserving configured fixed residues.

- Guided setup and reviewed design plans in natural language.
- Generate antibody sequences through LLM reasoning and structural verification.
- Agentic evolutional discovery for antibody design.
- Optional [PyRosetta relaxation and interface scoring](docs/pyrosetta.md), with
  auditable per-metric loss contributions and a reusable
  [bounded 40/40/20 loss policy](docs/examples/loss_presets/README.md).

> [!NOTE]
> OpenDDE Harness is an early preview, and you may encounter bugs. If something
> goes wrong, please [open an issue](https://github.com/aurekaresearch/OpenDDE-Harness/issues)
> with steps to reproduce it and your `ddeharness doctor --json` output.
> Your feedback helps us fix problems and improve the tool together.


## News

- **2026-09-09: Introducing OpenDDE Harness (preview) for agentic antibody design! Read the [technical report](docs/assets/OpenDDE_harness_tech_report.pdf).**
    - Design VHH, scFv, and paired VH/VL binders with LLM-guided sequence optimization and OpenDDE structure prediction.
    - Get started with the [installation guide](docs/installation.md), [onboarding](docs/onboarding.md), and [design examples](docs/examples/).
    - Inspect candidate sequences, structures, and design progress in the [tracing dashboard](docs/tracing-board.md).

See the [changelog](CHANGELOG.md) for release details.


https://github.com/user-attachments/assets/1530c7e6-9a7e-4069-b861-0e6cf58e6752


## Installation

Use a Linux or macOS client with Python 3.12 or newer. Native Windows is not
currently supported. Use
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) to install the
`opendde-harness` package and its `ddeharness` command in an isolated environment.
No environment activation is needed.

### Install from PyPI

```bash
uv tool install --python 3.12 opendde-harness
```

The release wheel includes the built terminal UI.

Features listed under **Unreleased** in the changelog require a source revision
containing them until a release is published. Follow the
[source/development installation guide](docs/installation.md#develop-from-an-editable-checkout)
to evaluate this branch; installing the current PyPI release does not select it.
PyRosetta is separately licensed and must be provisioned in the **compute worker**,
not merely in the client environment. See the
[setup checklist](docs/installation.md#enable-pyrosetta-and-bounded-scoring).

To update a PyPI installation, close the TUI and run:

```bash
uv tool upgrade opendde-harness
```

OpenDDE Harness checks PyPI once a day and says when a newer release is out:
in the TUI's status bar and once in its transcript at launch, and in
`ddeharness doctor`. `ddeharness upgrade` runs the update for your kind of
install (`uv tool upgrade` or `pip install --upgrade`). Set
`OPENDDE_HARNESS_NO_UPDATE_CHECK=1` to turn the check off.

### After installation

After installation, verify the client:

```bash
ddeharness --version
ddeharness --help
ddeharness tui --check
```

If the command is not found, run `uv tool update-shell` and open a new terminal.
Updates preserve configuration, results, and model data. Running compute
containers keep their current code until they exit; later starts use the updated
code. See the [installation guide](docs/installation.md) for runtime overrides,
compute environments, and model download options.

### Uninstall

Close the TUI and remove the client with:

```bash
uv tool uninstall opendde-harness
```

Configuration, results, model data, and compute services are retained.

## Quick Start

### 1. Configure the client and compute

```bash
ddeharness onboard
```

Follow the wizard to configure your LLM provider, optional memory, and a local
Docker environment or existing Linux compute service. Choose API or local OpenDDE
folding; both use the Harness compute service.

Local compute requires Linux x86-64 and Docker; CUDA also requires NVIDIA drivers
and NVIDIA Container Toolkit. See [onboarding](docs/onboarding.md) for setup and
model preparation details.

After setup, check your configuration, memory service, and compute readiness:

```bash
ddeharness doctor
```

Use `ddeharness doctor --probe` to send a test message to your LLM, or
`ddeharness doctor --compute-only` to check just the compute service. When
reporting a problem, include `ddeharness doctor --json` output and your command
or design request. See [troubleshooting](docs/troubleshooting.md) for common issues.

### 2. Start a design

Launch the terminal UI:

```bash
ddeharness
```

Then describe your task:

> Design a VHH against human CRLF2. Verify the target and epitope, keep the framework fixed, and design CDRs. Show the plan and start only after I confirm.

See the [terminal UI guide](docs/tui.md) for the screen, commands and keys.
CLI-based design workflows are also supported. See the
[CLI usage guide](docs/cli.md).
For YAML-based runs, see the [complete parameter reference](docs/protein-design-yaml.md)
and the [CRLF2 staged example](docs/examples/crlf2_scheduled.yaml), which changes
batch size, strategy weights, population capacity, and parent temperature by cycle interval.
The parameter reference also explains stagnation-triggered redesign, defaults, and current limitations.

For balanced confidence, energy and geometry/sequence scoring, explicitly apply
the [default bounded policy](docs/examples/loss_presets/README.md#applying-the-policy)
to your reviewed workflow. It is not a standalone run file, a universal scientific
calibration, or a change to existing runs. Validate and complete a
[small real verification run](docs/pyrosetta.md#end-to-end-verification-checklist)
before increasing the workload.

### 3. Inspect results

```bash
ddeharness tracing
```

Open the printed URL and select **Protein design** to inspect sequences, structures, metrics and agent activity.

![Candidate sequences, structures and linked metrics](docs/assets/tracing_board_1.png)

![Design progress, agent activity and candidate lineage](docs/assets/tracing_board_2.png)

Started design tasks run in background processes and continue after the TUI
closes while the client host remains running. Task state and worker logs default
to `~/.opendde_harness/protein_design/<task_id>/`; set
`OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT` to change the task root. With remote compute,
original structure files are stored on the compute host, and the dashboard uses
structures captured in the client's tracing data. See the [dashboard guide](docs/tracing-board.md).

## Development & License

See the [repository rules](AGENTS.md) and [compute image build instructions](docker/README.md). Ordinary users consume a prebuilt image; Dockerfiles remain available for publishers and customization. The Harness is licensed under [Apache-2.0](LICENSE); dependency and model licenses still apply, including the separate [PyRosetta terms](docs/pyrosetta.md#installation). TUI and memory foundations are adapted from Raven, and long-term memory is served by EverOS. Computational results require experimental validation. Model calls and compute may incur costs.

## Citation and Acknowledgements

If you use OpenDDE Harness in your work, please cite this software and the technical report linked above. When using OpenDDE for structure prediction, also cite the [OpenDDE technical report](https://arxiv.org/abs/2607.03787) and follow its citation and acknowledgement guidance. Cite the original methods for other models and tools used in your experiments, including SolubleMPNN and ESM2 when applicable.

We acknowledge Raven, EverOS, and the upstream scientific tools and models used by the Harness. Their software and model licenses continue to apply; consult the dependencies' own notices when installing or redistributing them.

## Partnership and Collaboration

![Collaboration](docs/assets/hiring.png)
