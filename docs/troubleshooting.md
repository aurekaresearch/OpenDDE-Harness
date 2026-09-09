# Troubleshooting

Start with the error and the affected layer: application, provider, Harness compute
service, or scientific backend. Do not repeatedly submit a design or MSA search
until you have checked whether the previous request is still running.

## Common symptoms

| Symptom | What to check |
| --- | --- |
| `ddeharness: command not found` | Open a new shell after installation and check `command -v ddeharness`; ensure the uv tool binary directory is on PATH. |
| TUI bundle built, then Python installation fails | `built .../ui-tui/dist/entry.js` only confirms frontend compilation. Fix the Python dependency/index error and rerun `uv tool install --python 3.12 --reinstall .` from the source checkout. |
| Client updated but compute image unchanged | Expected: client installation does not change compute. Pull the publisher's new image and rerun `ddeharness onboard` after the running container's tasks finish. |
| Image pull fails | Confirm the full published image reference, registry access, credentials and disk space. Set `OPENDDE_HARNESS_COMPUTE_IMAGE` before onboarding a new container. The default image is `aurekaresearch/opendde-harness:v1`. No automatic build fallback occurs. |
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
