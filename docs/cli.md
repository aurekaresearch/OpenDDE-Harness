# CLI usage

Use the CLI to validate a reviewed design YAML and launch a background task without
the interactive TUI. Complete [installation](installation.md) and
[compute configuration](onboarding.md) first.

## Command surface

Prefix each command below with `ddeharness`.

| Command | Purpose |
| --- | --- |
| `onboard` | Configure the LLM provider, long-term memory and Protein Design compute |
| `status` | Show client status |
| `doctor` | Report readiness (`--json`, `--timeout`, `--compute-only`, `--verify-hashes`); `--probe` also sends an LLM test message |
| `plugins` | List installed plugins and the active memory backend |
| `tracing` | Open the tracing dashboard; `tracing stop` shuts the background viewer down |
| `upgrade` | Legacy GitHub Release updater (`--check` only reports); use the installation guide for package updates |
| `compare` | Compare two candidate populations |
| `compute` | `prepare`, `serve`, `stop` |
| `provider` | `login`, `list`, `get`, `set`, `test`, `use`, `reset`, `show`, `endpoint` |
| `protein-design` | `context`, `validate`, `start` |
| `skill` | `list`, `get`, `block`, `unblock` |
| `tui` | Launch the native TUI (bare `ddeharness` does the same) |
| `sessions` | `create`, `list`, `resume`, `delete`, `fork`, `export` |

Every command accepts `--help`. `provider endpoint` has `add`, `remove`, and
`list` subcommands. The current `upgrade` implementation still queries
`OpenDDE-Harness-beta`; update a PyPI installation with
`uv tool upgrade opendde-harness`, or follow the
[source update instructions](installation.md#install-from-source).

## Find examples and defaults

```bash
ddeharness protein-design context --json
```

This command reports preparation defaults, example paths, and a read-only GPU
snapshot from the default compute service's `/health` endpoint. It does not start
a container or reserve a worker, and a returned inventory is not a full readiness
check. Installed wheels include the examples. To identify a source checkout, pass
`--repository-root /path/to/OpenDDE-Harness`.

Review and adapt an [example YAML](examples/) for your target, binder, paths and
folding mode. See the [YAML reference](../opendde_harness/plugin/protein_design/skills/protein-design/references/yaml-configuration.md)
for fields and [API folding requirements](protein-design.md#use-the-hosted-opendde-folding-api)
when using a remote folding backend.

## Validate

```bash
ddeharness protein-design validate --config /path/to/design.yaml
```

Validation checks the resolved configuration without starting a design or
querying the compute service. Explicit YAML fields override onboarding defaults.
Successful validation does not establish
that the compute service is healthy or that GPU inference will succeed.

## Start

```bash
ddeharness protein-design start --config /path/to/design.yaml
```

`start` validates the YAML and launches a detached task immediately, without an
additional confirmation prompt. Review the configuration before running it.
The output includes the task ID and compute URL. The detached worker continues
after the terminal closes while the client host remains running. A successful
launch does not mean the design has finished; avoid rerunning `start` merely to
check progress.

## Options for scripts

- `--json`: machine-readable output for `context`, `validate` and `start`.
- `--opendde-config /path/to/config.json`: load an explicit application configuration
  in the `context`, `validate`, or `start` command process. This is not a global
  option. The detached worker currently reloads the default
  `~/.opendde_harness/config.json`, so use that default configuration when launching
  a task; an alternate file is not forwarded to the worker.
- `--config` / `-c`: design YAML path for `validate` and `start`.

Application configuration and scientific YAML are different files. Keep service
credentials out of design YAML and source control.

```bash
ddeharness protein-design start --config /path/to/design.yaml --json
```

This command also launches immediately; `--json` changes output, not execution.

## Inspect results

```bash
ddeharness tracing --port 4318
```

Open the printed URL and select **Protein design**. Task state and worker logs
default to `~/.opendde_harness/protein_design/<task_id>/`;
`OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT` overrides that root. Original compute outputs
live on the compute host; the dashboard reads captured structures and metrics from
client tracing data. See the [dashboard guide](tracing-board.md).

For command-specific help:

```bash
ddeharness protein-design --help
ddeharness protein-design start --help
```
