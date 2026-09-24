# Workspace and internal storage

`workspace` is the default Shell, file-tool and child-agent execution directory.
Harness no longer initializes internal state or assistant templates there.
Project `AGENTS.md` is read according to the existing project-instruction rules;
Harness does not create or migrate it. User work such as `designs/` stays in place.

The active config file's parent is the instance data root. `scope` is a stable
SHA-256 digest of the resolved workspace path (including symlink resolution).
It isolates sessions and runtime state; it does not change memory app IDs.

| Data | Path relative to instance data root |
| --- | --- |
| Assistant instructions | `assistant/{soul.md,agent.md,TOOLS.md}` |
| Custom skills | `skills/` |
| Sessions | `sessions/<scope>/` |
| Default exports | `exports/<scope>/` |
| Memory queue, cursor and lock | `state/<scope>/memory/` |
| Shadow Git | `state/<scope>/checkpoint/shadow.git` |
| Local fallback memory | `memory/host/<scope>/` |
| EverOS | `memory/` (unchanged) |

External memory is the sole long-term recall source when enabled. The local
backend is an alternative, not a second injection lane. With memory disabled,
neither recall nor extraction is performed. Merely opening a client does not
create a local user profile.

Checkpoint paths are relative to the scoped checkpoint directory, not the
workspace. Absolute paths and escaping paths are refused. If the instance data
root is inside the execution workspace, checkpoints are disabled rather than
risk snapshotting internal data or credentials.

## Explicit migration

Ordinary startup only warns about legacy paths. It never moves their contents.
Preview before starting the updated client or initializing assistant templates:

```bash
ddeharness workspace migrate --workspace /path/to/workspace --config /path/to/config.json
```

Stop clients, Python workers and memory services first, then explicitly apply:

```bash
ddeharness workspace migrate --workspace /path/to/workspace --config /path/to/config.json --apply
```

Preview reports paths, sizes and conflicts, never file contents. Apply refuses
conflicting destinations, symlinks and active runtimes. Process verification is
currently supported on Linux; other platforms fail closed. The check is
conservative and may require stopping unrelated same-user Python processes.
Automatic live migration is not supported.

Original bytes and permissions are backed up under `migrations/<scope>/originals/`.
The manifest records verified mappings. Queues and cursors are copied while an
instance-wide pending marker blocks startup; sources are archived only after all
copies verify. If interrupted, rerun the same apply command to resume. Do not
manually remove the pending marker to bypass an unfinished migration.

Historical `user.md`, episodic notes and unclassified procedural material are
archived for review, **not imported into EverOS** and not silently discarded.
No paid model call is made. The command reports the number requiring review.
Custom checkpoint locations outside the historical default are not automatically
moved; retain them and review their mapping before deployment.

Rollback requires all related runtimes stopped and refuses changed/new data:

```bash
ddeharness workspace migrate --rollback /path/to/instance/migrations/SCOPE/manifest.json
```

Backups are retained after rollback. Migration and rollback are deployment
operations; installation alone does not perform either.
