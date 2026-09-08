# CLI usage

Use the CLI to validate a reviewed design YAML and launch a background task without
the interactive TUI. Complete [installation](installation.md) and
[compute configuration](onboarding.md) first.

## Command surface

| Command | Purpose |
| --- | --- |
| `onboard` | Configure the LLM provider, long-term memory and Protein Design compute |
| `status` | Show client status |
| `doctor` | Read-only readiness check (`--json`, `--probe`, `--timeout`, `--compute-only`, `--verify-hashes`) |
| `plugins` | List installed plugins and the active memory backend |
| `tracing` | Open the tracing dashboard; `tracing stop` shuts the background viewer down |
| `upgrade` | Install the latest stable release (`--check` only reports) |
| `compare` | Compare two candidate populations |
| `compute` | `prepare`, `serve`, `stop` |
| `provider` | `login`, `list`, `get`, `set`, `test`, `use`, `reset`, `show`, `endpoint` |
| `protein-design` | `context`, `validate`, `start` |
| `skill` | `list`, `get`, `block`, `unblock` |
| `tui` | Launch the native TUI (bare `ddeharness` does the same) |
| `sessions` | `create`, `list`, `resume`, `delete`, `fork`, `export` |

Every command accepts `--help`.

## Find examples and defaults

```bash
ddeharness protein-design context --json
```

This read-only command reports preparation defaults and available example paths;
it does not launch a task or check live compute readiness. If the source checkout
cannot be located, specify it with `--repository-root /path/to/opendde_harness`.

Review and adapt an [example YAML](examples/) for your target, binder, paths and
folding mode. See the [YAML reference](../opendde_harness/plugin/protein_design/skills/protein-design/references/yaml-configuration.md)
for fields and [API folding requirements](protein-design.md#use-the-hosted-opendde-folding-api)
when using a remote folding backend.

## Validate

```bash
ddeharness protein-design validate --config /path/to/design.yaml
```

Validation checks the resolved configuration without starting a design. Explicit
YAML fields override onboarding defaults. Successful validation does not establish
that the compute service is healthy or that GPU inference will succeed.

## Start

```bash
ddeharness protein-design start --config /path/to/design.yaml
```

`start` validates the YAML and launches a detached task immediately, without an
additional confirmation prompt. Review the configuration before running it.
The output includes the task ID and compute URL. Closing the terminal does not
stop the task; avoid rerunning `start` merely to check progress.

## Options for scripts

- `--json`: machine-readable output for `context`, `validate` and `start`.
- `--opendde-config /path/to/config.json`: use an explicit application configuration.
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

Open the printed URL and select **Protein design**. Default task state and results
live under `~/.opendde_harness/protein_design/<task_id>/`. See the
[dashboard guide](tracing-board.md) for progress, structures and metrics.

For command-specific help:

```bash
ddeharness protein-design --help
ddeharness protein-design start --help
```
