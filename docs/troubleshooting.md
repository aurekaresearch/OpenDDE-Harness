# Troubleshooting

Start with the error and the affected layer: application, provider, Harness compute
service, or scientific backend. Do not repeatedly submit a design or MSA search
until you have checked whether the previous request is still running.

## Common symptoms

| Symptom | What to check |
| --- | --- |
| `ddeharness: command not found` | For an editable checkout installed with `make install`, run `uv run ddeharness` from the repository root. For a uv tool installation, open a new shell and check `command -v ddeharness`; ensure the uv tool binary directory is on PATH. |
| `ui-tui/dist` missing or stale and npm unavailable | Source builds require Node.js 22.19 or newer and npm in the same shell. Run `npm --prefix ui-tui ci` and `npm --prefix ui-tui run build`, then retry installation. |
| TUI bundle built, then Python installation fails | `built .../ui-tui/dist/entry.js` only confirms frontend compilation. Fix the Python dependency/index error and rerun `uv tool install --python 3.12 --reinstall .` from the source checkout. |
| The terminal UI is garbled, colorless, or left on the alternate screen | See the [TUI guide](tui.md#troubleshooting); `reset` recovers a terminal a killed process left behind. |
| Client updated but compute image unchanged | Expected: client installation does not change compute. Pull the publisher's new image and rerun `ddeharness onboard` after the running container's tasks finish. |
| Image pull fails | Confirm the full published image reference, registry access, credentials and disk space. Set `OPENDDE_HARNESS_COMPUTE_IMAGE` before onboarding a new container. The default image is `aurekaresearch/opendde-harness:v1`. No automatic build fallback occurs. |
| PyRosetta image build cannot copy `pyproject.toml`, or reports `Excluded dependency present: pyrosetta` | Use the corrected build context and opt-in doctor together; see [PyRosetta runtime troubleshooting](../docker/README.md#optional-pyrosetta-runtime). A successful `--dry-run` does not validate the image build. |
| PyRosetta imports on the host but fails in a task | Verify the selected worker image contains PyRosetta, then select it during onboarding after active tasks finish. The host `.venv` is separate from the worker; see [Docker-managed installation](pyrosetta.md#docker-managed-compute). |
| Config fails schema validation with unknown keys | Keys from earlier releases are not migrated. Remove the named keys from `~/.opendde_harness/config.json`, or run `ddeharness onboard` to write a fresh one. The error names the full key path. |
| Example YAML not found | Run `ddeharness protein-design context --json` for verified paths to the two YAMLs in `docs/examples/`. The workspace may not be the checkout. Supply the real checkout with `--repository-root` if needed, rather than repeatedly searching global wildcards. |
| Compute connection refused | Check the task's resolved compute URL, service status, authentication and network access. `127.0.0.1` means the host of the calling process, not necessarily your laptop. |
| Compute route returns 404 | Compare the selected worker URL and deployed client/server versions. A health response alone does not prove a specific route exists. |
| Missing checkpoint or CCD cache | Check host files and their container-visible mounts. Follow the persistent layout in the setup guide; do not replace paths with another machine's absolute paths. |
| OpenDDE data download fails | Rerun `ddeharness compute prepare --assets-only` with the same data roots. Verified files and partial downloads are retained; an interrupted transfer resumes where it stopped. A checksum mismatch needs inspection; existing mismatched files are not overwritten. |
| ESM2 or SolubleMPNN unavailable | These weights live on the host, not in the image. Run `ddeharness compute prepare --assets-only` and check the Harness weights root (`~/.cache/opendde-harness`, or `OPENDDE_HARNESS_WEIGHTS_DIR`); `ddeharness doctor --verify-hashes` reports which required file is missing or mismatched. |
| Docker or NVIDIA runtime unavailable | Install/start Docker and configure NVIDIA Container Toolkit on the compute host. The client installer does not provision these system components. |
| MSA search times out | Inspect the existing job and upstream response before retrying. Confirm alignment files and depth before binding them to the design. |
| Design task fails at cycle 0 on a ProTrek call | The configured default is `http://search-protrek.com/`; use the protocol supported by your service instead of only changing the URL scheme. Check that first, then whether the compute container has any route to it. Optional services no longer fail a run, so update the compute code first. Then check `ddeharness doctor --compute-only`, which prints one line per optional external service, and give the container egress: export `http_proxy`/`https_proxy` before `ddeharness onboard` (a host loopback proxy is rewritten to `host.docker.internal`), or set `plugins.config["protein-design"].compute_docker.env` in `~/.opendde_harness/config.json` after onboarding, then restart idle compute with the saved settings. Set `PROTREK_ENDPOINT=""` to disable the search instead. |
| Target MSA search reports the server is unreachable | The error names the endpoint and the proxy option. Confirm the container's egress the same way as above, then rerun; `MMSEQS_SERVICE_HOST_URL` selects a private MMseqs service. |
| Provider rejects a request | Record the model, operation and sanitized error. Verify the provider supports the requested parameters and tools; do not silently change scientific settings. |
| Missing dashboard structure or I/O | Check task artifacts and recorded events. Historical payloads cannot be recovered merely by refreshing the UI. |
| Dashboard port unavailable | Use the URL printed by `ddeharness tracing`, or choose another port with `--port`. Do not terminate an unidentified listener. |

## PyRosetta and bounded-loss setup

- **Import succeeds on the host but fails in a task:** trace the task's resolved
  worker and its actual Python subprocess. A registry entry or YAML compute
  override may select a different service. Inspect image ID and mounted source;
  follow [runtime verification](pyrosetta.md#verify-the-selected-runtime). Do not
  assume rebuilding an already working image will correct placement.
- **Worker is busy after changing image/code:** the service refuses replacement
  to preserve active tasks. Wait for completion; do not force-stop it or start a
  competing GPU container as a workaround. Correct settings apply to a later start.
- **Preset file is missing after installation:** older published releases may not
  contain this feature. Use the intended revision and the
  [installed resource location](examples/loss_presets/README.md#applying-the-policy).
  The preset is a fragment, not a complete workflow accepted by `start`.
- **Anchor/group validation fails:** replace all four loss-policy fields together.
  Partial `loss_weights` mappings retain omitted built-in coefficients. Every
  positive term needs one anchor and one group; zero-weight terms need neither.
  Budgets must sum to one, and good/bad order must match preference direction.
- **Small or negative linear loss:** inspect signed contributions; cancellation
  can be valid. Opt into [fixed bounded scoring](pyrosetta.md#opt-in-bounded-objective-with-fixed-group-budgets)
  in a new configuration rather than silently rescaling existing results.
- **Min ipAE unavailable:** inspect `metadata.min_ipae` and the matching confidence
  artifact/chain mapping. It is not mean `i_pae` or ipSAE; old runs without the raw
  metric are not backfilled. A positive Min ipAE loss term requires the measurement.
- **ipSAE is zero:** distinguish `metadata.ipsae.status: unavailable` with an
  explicit fallback from `status: success` with a measured zero. Under the default
  bounded policy, either raw zero incurs the full configured ipSAE penalty.
- **Lines coloured by Contacts instead of age:** choose **Color by → Age / cycle**
  above the properties chart. Reload after a static-asset update; do not restart
  compute. The selector is independent of the 3D structure colour controls.

## Delay before the first reply

Skill selection makes no model call. A catalogue of at most 40 advertisable
skills is listed whole, in a query-independent order, so the block is
byte-identical from turn to turn and the provider's prefix cache survives.
A larger catalogue, or any catalogue reached through a memory backend, is
narrowed to 5 entries by BM25 over the message you just sent. Either way the
prompt carries names, one-line descriptions and qualified ids only; the agent
loads a body by calling `use_skill`, which reads it from disk.

A skill whose declared `requires` are unmet is not advertised — check
`ddeharness skill list` if one you expect is missing, and install the binary or
export the environment variable it names. `skillForge.enabled=false` (or
`skillForge.router.enabled=false`) drops the block entirely.

What is left before the first token is provider queueing and response latency.

## Check local compute

The locally managed compute container is ephemeral: it starts on demand, is
named `opendde-compute-<code-id>` after the installed code release, and removes
itself when idle (see the [container lifecycle](onboarding.md#container-lifecycle)).
`ddeharness doctor` prints whether it is running, its name, port, code id, jobs,
idle countdown and leases; `~/.opendde_harness/compute/local.json` records the
same instance. Do not use Compose commands to manage it.

While it runs, read its log on the compute host:

```bash
ddeharness doctor --compute-only
docker logs --tail=100 CONTAINER_NAME
```

Replace `CONTAINER_NAME` with the name printed by `doctor`, such as
`opendde-compute-` followed by its code ID. Once the container has exited, its
Docker log is gone; reproduce the failure
by starting a task or rerunning onboarding, which waits up to 90 seconds for
readiness and prints the service's last error. Check readiness and authentication through onboarding
at the configured Harness compute URL. For an older custom or Compose
deployment, use its original configuration. The Harness compute URL and hosted
OpenDDE API URL are distinct endpoints.

Restarting compute can interrupt in-flight work. `ddeharness compute stop` refuses
while jobs or task leases are active, printing the running/queued job and lease
counts and exiting non-zero; `--force` stops it regardless and drains running jobs first;
even so, inspect active tasks, save their IDs, and arrange a maintenance window.
Closing the TUI is not a stop command (it only lets an idle container exit);
ask it to stop the specific task ID and verify the returned status.

## Report a reproducible issue

Include the repository commit (`git rev-parse HEAD`), operating system, installation
method, compute mode, image version, sanitized configuration, exact command or TUI
request, expected behavior, `ddeharness doctor --json` output, and the smallest
relevant error excerpt. For dashboard
issues, include the browser version and a screenshot.

Never publish API keys, bearer tokens, `.env` files, private endpoints, proprietary
sequences, or complete task archives without reviewing their contents. Replace
sensitive inputs with a minimal public example where possible.

See [compute setup](protein-design.md), [examples](examples/)
and the [dashboard guide](tracing-board.md) for the supported workflows.

## Web search and fetching need no key

`web_search` is served by the model provider's own search when it has one --
the Codex login, OpenAI's Responses API, Anthropic, and any LiteLLM vendor that
supports it -- and the answer cites what the model read. On an endpoint without
one (a relay, DeepSeek, a local deployment) the tool queries Brave's Search
API when `tools.web.braveApiKey` is set (the onboarding wizard asks for it;
free tier 2,000 queries a month) and DuckDuckGo otherwise, which answers
without a key; `content=true` adds an excerpt of each result's page.
DuckDuckGo answers a burst of queries from one address with a puzzle page; the
tool reports that as a refusal once and refuses the same call for two minutes
rather than letting the model retry it all turn. Result quality on that path
is the engine's; a provider with a hosted search (the Codex login, OpenAI,
Anthropic) searches better, and a fixed data source is not a search at all
(see the lookup skills below).

`web_fetch` returns an API's JSON, TSV or FASTA as it is and reduces an HTML
page to its article locally (trafilatura). When the local read cannot say what
the page says -- the article is thin because the site renders client-side, or
the site refused the fetch -- the page is read through Jina Reader, which
renders it in a browser and is free without a key; `tools.web.jinaApiKey` only
raises Jina's rate limit. `tools.web.proxy` applies to both tools. A config written before this release carries a `tools.web.search`
object; it is ignored on load and can be deleted.

For the data sources the antibody workflow reads all the time -- RCSB PDB,
UniProt, PubMed / Europe PMC -- the `pdb-lookup`, `uniprot-lookup` and
`pubmed-lookup` skills give the model the exact keyless endpoints and the
fields worth asking for, so a structure, sequence or paper question is one
`web_fetch` rather than a search.

## Server-side compaction on the Codex login

On the Codex login the model's own backend can compact a conversation: once a
call's prompt reaches `context.server_compact_ratio` (default 0.8) of the
model's window, the loop sends the prompt back with a `compaction_trigger`
and stores the opaque `compaction` item the backend answers with as a session
message. The next turn on the same model replays the last user messages plus
that item in place of the earlier history, at the backend's own fidelity; any
other model, or a session resumed elsewhere, ignores the marker and reads the
local history, which stays in the session as before. It costs one call on the
full prompt at compaction time. Set the ratio to 0 to keep local handling only.

The share must be between 0 and 1; a value outside that range (or a NaN) is
refused at config load rather than read as a working setting. Everything else
follows the turn's last call as it was actually made -- the model that answered
it, that model's window and reported prompt size, and the tool catalogue that
call sent -- so a routing strategy or a fallback hop compacts on the model that
is filling up, not on the session's default. A call the vendor reported no
prompt usage for is never compacted. What the marker keeps in clear beside the
opaque item is recent user *text*, bounded by a real token count; an attachment
is left out, the same way the session log leaves one out.

Once a marker exists it is never trimmed away to make a prompt fit: the local
history stays in the session in full for every other model, but on the model
that made the marker the request carries the marker and what follows it, so
dropping the marker is the one edit that makes the request bigger rather than
smaller.
