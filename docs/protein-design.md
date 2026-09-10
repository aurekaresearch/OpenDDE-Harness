# Protein design setup

OpenDDE Harness owns the conversational agent, structured analyze/design/reflection sessions,
deterministic cycle orchestration, scientific compute adapters, task tools, and
shared long-term memory hooks. A versioned REST contract separates the TUI task process
from the GPU compute service.

This integration is dedicated to antigen-antibody design. It accepts single-chain
VHH or scFv binders and paired VH/VL binders; other binder topologies and general
protein-design tasks are rejected before compute allocation.

## Configure OpenDDE Harness

Commands that reference `docs/examples/` assume a source checkout; a PyPI
installation resolves those paths with `ddeharness protein-design context`.
Hostnames ending in `example.com` and worker names below are placeholders, not
public services. Replace them with endpoints provided by your deployment. Never
commit credentials.

Use `ddeharness onboard` for provider/model setup, optional long-term memory, and the
Protein Design compute URL, token and folding defaults. Step 3 can start a local
container, reuse a running one of the installed release, or connect to an existing
service. For a local container it pulls a missing prebuilt image after confirmation,
prepares SolubleMPNN and ESM2 650M in both modes, and adds the OpenDDE
checkpoint/common data in local mode. See the
[onboarding guide](onboarding.md) for the three-step workflow and local/API modes.

The following JSON is an advanced configuration reference, not a required setup
step. If editing `~/.opendde_harness/config.json` manually, merge only the fields
you need; do not overwrite existing credentials or unrelated configuration:

```json
{
  "plugins": {
    "config": {
      "protein-design": {
        "compute_url": "http://127.0.0.1:8080",
        "fold_poll_interval": 2.0
      }
    }
  },
  "memory": {
    "backend": "longterm"
  }
}
```

Disable the integration by adding `"protein-design"` to `plugins.disabled`.
Configure your provider and model through onboarding or the existing application
configuration. Verify model access before submitting a long design run.

### Bind tasks to compute servers

One OpenDDE Harness TUI can run several design tasks at the same time.  Configure a
static worker registry when Docker services live on different servers:

```json
{
  "plugins": {
    "config": {
      "protein-design": {
        "compute_token": "replace-with-a-shared-secret",
        "compute_workers": [
          {
            "id": "gpu-worker-1",
            "url": "http://gpu-worker-1:8080",
            "profiles": ["gpu8"],
            "backends": ["opendde"]
          },
          {
            "id": "gpu-worker-2",
            "url": "http://gpu-worker-2:8080",
            "profiles": ["gpu8"],
            "backends": ["opendde"]
          }
        ]
      }
    }
  }
}
```

Select a server in a task YAML with at most one of these forms:

```yaml
compute:
  url: http://gpu-worker-2:8080
# Or: worker_id: gpu-worker-2
# Or: profile: gpu8
```

`url` and `worker_id` are explicit placement. A profile lets OpenDDE Harness probe the
matching workers and choose the lowest normalized queue load. Placement occurs
once at task start: all cycles, auxiliary model calls, population access, and
candidate downloads remain bound to that URL. The task status and Dashboard
show the resolved worker. If `compute` is omitted, the configured worker pool is
load-balanced; if no registry exists, the plugin-level `compute_url` (default
`http://127.0.0.1:8080`) is used.

### Place jobs on GPUs inside one worker

One compute worker sees the GPUs exposed to its container and leases them per job. A fold
holds its GPUs exclusively; ESM-2 and SolubleMPNN take one GPU each and share it
with other short jobs, up to `OPENDDE_HARNESS_SHARED_JOBS_PER_GPU` (default 2),
but never with a running fold. Jobs on disjoint GPUs run at the same time.

Placement is automatic: each job takes the least-loaded free GPUs. Pin GPUs only
when a run must stay on specific devices:

```yaml
compute:
  placement:
    fold: [1, 2, 3]
    esm: 0
    mpnn: 0
    cp_degree: 3
```

`fold` must list exactly `cp_degree` distinct indices, and more than one GPU
runs OpenDDE's Fold-CP context-parallel inference at that degree. `esm` and
`mpnn` take a single index each. A request for a busy GPU waits in the queue
rather than failing. `design.esm_device`, when set, still pins the ESM-2
scoring that runs inside a fold job. `GET /health` reports the inventory and the
current leases under `workers.gpu_leases`, and `protein_design_context` returns
the same inventory as `compute.gpus`.

An unused service exits after `OPENDDE_HARNESS_COMPUTE_IDLE_SECONDS` (default
600) without work, so its GPUs return to the host.

When serving compute directly on a prepared Linux host with `ddeharness compute
serve`, set `OPENDDE_HARNESS_PROTEIN_DESIGN_HOST` to its trusted cluster interface
(the default is `127.0.0.1`). Set the same non-empty
`OPENDDE_HARNESS_PROTEIN_DESIGN_TOKEN` on the service and `compute_token` in the
client. Locally managed containers remain bound to host loopback; an environment
variable alone does not expose their port remotely. Do not expose this API directly to an
untrusted network.

## Start compute

Install the client, obtain a trusted prebuilt compute image, and run
`ddeharness onboard`. For local folding, download OpenDDE data separately with
`ddeharness compute prepare --assets-only --mode local`, or let onboarding
prepare missing files after confirmation. See [installation](installation.md)
for image selection and paths.

The client runs agents, task orchestration, memory and tracing. The on-demand
compute service runs scientific tools and exits when idle. Local folding runs
OpenDDE inside that service; API folding submits fold/refold upstream while
auxiliary tools remain in the Harness compute service. Both modes run the same
prebuilt environment image, which carries no project code and no model weights:
the prepared code snapshot and the SolubleMPNN/ESM2 650M weights are mounted
read-only from the host.

### Start through onboarding

Onboarding reuses the running container of the installed release, or pulls a
missing image and starts one after confirmation. It never builds an image. Only
local folding needs external OpenDDE data. A running container of the installed
release is not restarted while it is busy. See
[container lifecycle](onboarding.md#container-lifecycle).

The `docker/` directory contains only maintainer build inputs. For ordinary
usage, [pull a prebuilt image](installation.md#select-a-compute-image) and use
onboarding to start or connect to compute. No Compose template is required.
Older custom deployments should retain their original configuration. Recreate
containers only after active tasks finish.

### Use the hosted OpenDDE folding API

OpenDDE itself does not have to be installed on the compute worker when folding
uses the asynchronous job API. The Harness compute service still runs ESM2, SolubleMPNN, structural analysis,
and population storage. The client applies gates and admission decisions, while fold/refold requests
are submitted remotely and their ZIP artifacts are copied into the task output:

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

API mode uses the external HTTPS gateway `https://api.aurekabio.cloud` by default.
Set `fold.api_url` to override it, or supply `OPENDDE_HARNESS_OPENDDE_API_URL`
to the compute service (explicit YAML takes precedence). No cluster-internal
service address or shared NFS access is needed. The wizard no longer asks for an upstream API
token and the compute service sends none; the hosted folding API is unauthenticated.
The remote service owns GPU scheduling and MSA/template preparation. Local GPU
and model paths do not control remote folding. Local `pairedMsaPath` or
`unpairedMsaPath` values and `use_msa: false` are rejected in API mode; remove
local MSA settings when adapting an example.

For API folding, use onboarding's API mode or connect to an existing prepared
compute service. A newly created local container still needs the prepared
SolubleMPNN and ESM2 650M weights mounted from the host, but no OpenDDE
checkpoint or common data. Merely installing the host client does not provision
this scientific runtime. Existing container environment settings and mounts are
not changed by a task YAML.

API result downloads automatically retry transport errors, HTTP 408/429/5xx,
and invalid ZIP archives up to five times after the initial attempt, waiting
three seconds between attempts. Each attempt downloads the same completed job
from the beginning; it does not resubmit prediction. The ZIP is checked before
replacing the destination file. Authentication and not-found errors fail
immediately. If retries are exhausted, the error includes the upstream job ID
for a later download. This policy does not change the separate fold-job wait
timeout. Restart or recreate an existing compute service to load updated code.

API requests default to `need_atom_confidence: true`. The downloaded ZIP includes
the matching `full_data_sample_*.json` with atom pLDDT, token PAE, and contact
tensors. API and local mode use the same 10-component weighted `loss`, minimized
by the design workflow. Do not disable atom confidence when optimizing loss;
missing confidence artifacts cause scoring to fail, not fall back to ipTM.

## Use from OpenDDE Harness

In the TUI, ask to run a named example, provide a YAML path, or describe a new antibody design. The `protein-design` Skill first calls the read-only `protein_design_context` tool for verified example paths and credential-free compute/folding defaults. The CLI equivalent is `ddeharness protein-design context --json`. Existing example settings are reused; new designs require resolving missing target facts and scaffold choices. Only unresolved choices are asked, not three mandatory review rounds. The final validated configuration is summarized with an explicit question approving it and authorizing launch. Only an affirmative reply lets the agent call `protein_design_start` with `user_confirmed: true`. Context queries `/health` for GPU information without starting a service or binding a worker. Validation checks configuration only; neither is a full readiness check. Status, adjustment, stop, candidates, and confirmed target-only MSA tools remain available. Each task runs in a detached worker under `~/.opendde_harness/protein_design/<task_id>`; closing or restarting the TUI does not stop it while the client host remains running. Long folding work runs as an asynchronous compute job and the task records design cases through the configured long-term memory backend.

Bundled YAML examples live in the repository's `docs/examples/` and
are available through the TUI or CLI: `crlf2_quickstart.yaml` and
`cacng1_quickstart.yaml`. The context tool resolves their actual absolute paths
without requiring an example README or searching the whole filesystem.
The final question combines configuration approval and launch authorization.

The same reviewed YAML can be launched without the TUI, from the repository root
(`validate` only checks; `start` launches without interactive confirmation):

```bash
ddeharness protein-design validate --config docs/examples/crlf2_quickstart.yaml
ddeharness protein-design start --config docs/examples/crlf2_quickstart.yaml
```

Add `--json` for scripts. When a TUI request supplies only a target name,
sequence, or design intent, the bundled `protein-design` skill uses its YAML
configuration reference to collect and confirm key information, create a file
under `~/.opendde_harness/protein_design/configs/`, and validate it. An earlier generic
request to run does not replace the final confirmation after the reviewed YAML
summary. The execution boundary remains a validated, explicitly confirmed
on-disk configuration rather than an inline mapping.

### Prepare a missing target MSA

MSA preparation is explicit and target-only. API mode uses service-managed MSA
and does not need local A3M searches. For local folding, combine an unresolved
MSA choice with other missing target information; do not re-ask settled choices:

1. disable MSA for this run;
2. provide existing paired and unpaired A3M paths;
3. search online with `protein_design_search_target_msa`.

The search tool may be called only after option 3 is explicitly confirmed. It
accepts one canonical antigen sequence and rejects binder/antibody use. Its
public compute placement is deliberately simple: omit `compute_url` to let the
configured pool/default select a worker, or pass the explicit URL already
present in the reviewed design. `compute_worker_id` and `compute_profile` remain
internal task-placement concepts and are not arguments of the MSA tool.

The selected compute service submits the sequence through Protenix's online MMseqs2
client, validates that the result contains homologs rather than only the query,
and stores `non_pairing.a3m` and `pairing.a3m` under:

```text
<compute-output-root>/msa/<target>-<chain>-<sequence-hash>/
```

Results are reused by sequence hash unless `force: true`. The response contains
both absolute paths, alignment depth, cache state, server URL/mode, and resolved
compute URL/worker ID. Copy the paths into the target chain configuration and
bind the final design YAML to that returned worker. The default public endpoint
is `https://protenix-server.com/api/msa`; configure a compatible private service with:

```dotenv
MMSEQS_SERVICE_HOST_URL=https://protenix-server.com/api/msa
OPENDDE_HARNESS_MSA_SERVER_MODE=protenix
OPENDDE_HARNESS_MSA_SEARCH_TIMEOUT=1800
```

Public service availability and rate limits are external to OpenDDE Harness.
The CLI launcher does not perform this interactive preparation step.

## External services

ProTrek and the target MSA service are external lookups used by the compute
service. Optional enrichment can report an unavailable result without stopping
the design. An explicitly requested target MSA search is required by default and
reports an error when it cannot complete. API folding also uses an external
service and is required when that folding mode is selected.

| Service | Default endpoint | Used by | Setting |
| --- | --- | --- | --- |
| ProTrek | `http://search-protrek.com/` | analysis-agent homolog search | `PROTREK_ENDPOINT` |
| Target MSA | `https://protenix-server.com/api/msa` | `protein_design_search_target_msa` | `MMSEQS_SERVICE_HOST_URL` |

Connection-level failures (connect timeout, refused connection, DNS failure) are
reported, not raised: the compute service answers HTTP 200 with
`{"available": false, "service": ..., "reason": ..., "endpoint": ...}`, the agent
sees plain text saying the search is unavailable, and the cycle is neither
retried nor counted as failed. Only an explicitly requested target MSA search
fails loudly, and its error names the endpoint and the proxy option.

The configured ProTrek default uses plain HTTP. Use the protocol supported by
your deployment; changing the scheme alone does not enable TLS. The
request carries only the query sequence, or the query structure file for a
structure search; no task identifiers, credentials or design context leave the
container. Point `PROTREK_ENDPOINT` at a private deployment to keep the traffic
inside your network, or set `PROTREK_ENDPOINT=""` to disable ProTrek entirely, in
which case the tool reports that it is disabled instead of attempting a call.

`ddeharness doctor` prints one line per optional service, and
`GET /health?probe_external=1` probes each endpoint from inside the container
with a three-second timeout and reports `reachable`, `endpoint` and `reason`.
Neither one fails the readiness check.

### Give the container egress

The compute container has its own network namespace, so a proxy listening on the
host's loopback address is not reachable at `127.0.0.1` from inside it.
Onboarding therefore copies `http_proxy`, `https_proxy`, `all_proxy` and
`no_proxy` (both letter cases) from the host into the container, rewrites
`127.0.0.1`, `localhost` and `::1` proxy hosts to `host.docker.internal`, adds
`--add-host=host.docker.internal:host-gateway`, and appends
`localhost,127.0.0.1,host.docker.internal` to `no_proxy`. Export the proxy
variables before running `ddeharness onboard`.

`PROTREK_ENDPOINT`, `MMSEQS_SERVICE_HOST_URL`, `OPENDDE_HARNESS_MSA_SERVER_MODE`
and `OPENDDE_HARNESS_MSA_SEARCH_TIMEOUT` are passed through from the host when
they are set. To pin them regardless of the host environment, add an explicit
mapping under `plugins.config["protein-design"].compute_docker` in
`~/.opendde_harness/config.json`. Merge this into the existing configuration:

```json
{
  "plugins": {
    "config": {
      "protein-design": {
        "compute_docker": {
          "env": {
            "PROTREK_ENDPOINT": "",
            "MMSEQS_SERVICE_HOST_URL": "https://msa.internal.example/api/msa",
            "https_proxy": "http://proxy.internal.example:3128"
          }
        }
      }
    }
  }
}
```

Those values win over the host environment. Variables onboarding owns itself
(the compute token, `PYTHONPATH`, `PROTEIN_DESIGN_OUTPUT_PATH` and the
`STRUCTPRED_*` paths) are rejected. Settings changes apply the next time the
container starts. Add this mapping after onboarding: the local wizard currently
rebuilds `compute_docker` without retaining `env`. Stop the idle container with
`ddeharness compute stop`, then let the next task start it with these overrides.

## Long-term memory scope

Memory scope is automatic and is not a scientific YAML option. Each record is
scoped by an application id and a project id. Ordinary OpenDDE Harness
conversation memory and SkillForge memory retrieval use:

```text
app opendde_harness, project general
```

Protein-design cases and learned skills use the target name exactly as written
in the YAML:

```text
app protein-design, project <target.name>
```

For example, `target.name: CRLF2` maps to project `CRLF2` under the
`protein-design` application. Runs with the
same exact target name share long-term cases and learned skills, while different
names remain isolated. OpenDDE Harness does not normalize case, punctuation, or biological
aliases. A run's `task_id` is the memory session id inside that target scope.

Meaningful cycle events are appended as `protein_design_case_v2` records. A v2
case captures the selected parent, pre-cycle global best, primary and learned
skills, representative mutations, gate and population outcomes, objective and
loss-component deltas, structure evidence, and a deterministic lesson. The
normal consolidation boundary is Reflection: non-final events may accumulate
before Reflection flushes the session and lets the memory service extract, cluster, and
update learned skills.

## Monitor a design task

The TUI's **Design Tasks** area shows running and recent detached tasks without
streaming fold logs through the chat input. Start OpenDDE Harness's tracing viewer and open the `Protein design` workspace. The run
selector contains only tasks launched by OpenDDE Harness, ordered with active tasks first
and recent history after them. The viewer automatically selects the newest
active task, or the newest completed task when none is active.

The workspace reads OpenDDE Harness tracing. The candidate workspace uses 3Dmol.js for
structures and shows configured CDR regions, sequences and available metrics.
Design loop steps expose the recorded Input/Output; the adjacent search tree
shows parent–child lineage without persistent node labels. Metric charts show
current-cycle best and global objective-best candidates, with synchronized
cycle hover. `Open trace` switches to the underlying run/cycle/phase trace.
See the [Dashboard guide](tracing-board.md) for interactions and missing-data behavior.

The compute service exposes `GET /structure?path=...` for this transfer. It only
reads `.pdb`, `.cif`, and `.mmcif` regular files below the configured output
root, rejects symlink escape, and enforces a byte limit. Structure capture is
best effort and cannot fail the scientific design run.

## Validation

Run `uv run pytest -q` from the source checkout. The tests cover REST contracts,
job scheduling, structured-output handling, orchestration, memory hooks, and
plugin discovery. Real model inference and external-service availability require
validation on the intended compute deployment.

### Live fold result visibility

Initial scoring (cycle -1), every search cycle, and terminal post-refolding
(cycle equal to the configured cycle count) publish scores and structure artifacts
as soon as folding returns. Search-cycle checkpoints are updated under the same
trace span after quality assessment, so normal completion does not duplicate a
round. Early checkpoints do not claim population admission or final selection.
Results remain visible if subsequent speculation, quality assessment, population
persistence, reflection, or memory work blocks or fails. Terminal refold results
are visible before pose analysis and final filtering. These changes require a
new worker process; they do not backfill historical runs automatically.
