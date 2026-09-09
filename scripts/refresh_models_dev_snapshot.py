"""Regenerate the bundled models.dev snapshot.

OpenDDE Harness ships a trimmed copy of the models.dev catalogue so a fresh
install can label models, price a call and size a context window without a
network round trip, and so tests never depend on one. This script is how that
copy is produced -- editing it by hand would leave no way to tell what it was
trimmed from.

    uv run python scripts/refresh_models_dev_snapshot.py

This is a maintainer-time tool: run it before a release and commit the output.
Nothing at runtime reaches the network for this data. The harness has to work
on machines with no route to sites outside mainland China, where a runtime
fetch fails silently and the reader falls back to a guess -- so the file the
wheel carries is the only source, and this script is the only thing that ever
fetches.

Source: the project's own repository rather than its ``/api.json`` endpoint.
Both are models.dev; the repository is the one that can be pinned. api.json is
a rendering of these files served by a small project's app, and an outage there
is the failure this snapshot exists to survive -- so the refresh should not
depend on it either. Pinning also makes a refresh reproducible: the recorded
commit sha says exactly which catalogue a snapshot came from, where "whatever
the site returned that day" said nothing.

What is kept and why:

* only the providers OpenDDE Harness can reach -- either LiteLLM maps the vendor
  (so a prefixed id routes) or the registry carries a ``ProviderSpec`` for it.
  The full catalogue is 181 providers, most of which the harness has no way to
  talk to, and the repository rejects additions over 1 MiB;
* per provider row: the fields a person reads when choosing a model, the cost
  that prices a finished call, and ``limit`` -- the window and output ceiling
  *as that provider serves the model*, which a reseller may cap below the
  vendor's own figure;
* a second, provider-agnostic table of the vendor's native ``limit`` per model
  (the repository's ``models/<vendor>/<id>.toml``), stored under ``_models``.
  It answers for a model reached through a provider the catalogue has no row
  for, and is an upper bound by construction: a provider row, when one exists,
  is consulted first.

Wire-format capability flags stay out: ``ProviderSpec`` answers those. The
one model capability kept is ``reasoning``, because the request shape that
switches thinking on is only worth sending to a model that thinks.
"""

from __future__ import annotations

import copy
import io
import json
import sys
import tarfile
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

REPO = "anomalyco/models.dev"
REF = "dev"
COMMIT_API = f"https://api.github.com/repos/{REPO}/commits/{REF}"


def tarball_url(sha: str) -> str:
    """The archive of exactly the commit the snapshot records, not the branch tip."""
    return f"https://codeload.github.com/{REPO}/tar.gz/{sha}"


SNAPSHOT = Path(__file__).resolve().parents[1] / "opendde_harness" / "providers" / "data" / "models_dev.json"

#: OpenDDE Harness's provider name -> models.dev's name for the same vendor.
#: Only the ones that differ; a matching name needs no entry. Absent vendors
#: (VolcEngine, a local Ollama) and harness-only sections (custom, hosted_vllm)
#: have no upstream row by nature, not by oversight.
PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
    "dashscope": "alibaba",
    "moonshot": "moonshotai",
    "azure_openai": "azure",
    "github_copilot": "github-copilot",
    "minimax_global": "minimax",
    "minimax_cn": "minimax-cn",
}

#: Provider-row fields carried over. ``name``/``description`` are what a picker
#: renders; ``cost`` prices a finished call; ``limit`` sizes the next request;
#: ``reasoning`` says whether an effort is worth sending (providers/compat).
KEEP = ("name", "description", "cost", "limit", "reasoning")

#: Canonical-row fields carried over. Limit and the reasoning flag: labels and
#: prices are read from the provider rows, which know what the serving party
#: charges.
KEEP_CANONICAL = ("limit", "reasoning")


def upstream_name(provider: str) -> str:
    return PROVIDER_ALIASES.get(provider, provider)


def reachable_providers(catalogue: dict[str, dict]) -> set[str]:
    """Harness-side names worth carrying rows for.

    Two ways to be reachable and neither contains the other: LiteLLM maps ~130
    vendors the registry has no spec for (a prefixed id routes to them with only
    a key), and the registry carries specs for gateways and regional instances
    LiteLLM has never heard of (aihubmix, siliconflow, minimax_cn). Taking only
    the first silently drops those three -- 123 models -- while every total in
    the summary still goes up, which is why ``tests/test_providers_catalog.py``
    asserts a labelled provider list rather than a total.
    """
    from opendde_harness.providers.registry import PROVIDERS

    try:
        from opendde_harness.providers.litellm_setup import import_litellm

        # ``provider_list`` holds ``LlmProviders`` members, whose ``str()`` is
        # "LlmProviders.OPENAI" -- comparing that to a vendor name matches nothing
        # and silently keeps the union empty.
        known = {getattr(p, "value", str(p)) for p in getattr(import_litellm(), "provider_list", [])}
    except Exception:  # pragma: no cover - litellm ships with the project
        known = set()

    inverse = {v: k for k, v in PROVIDER_ALIASES.items()}
    wanted = {spec.name for spec in PROVIDERS}
    for upstream in catalogue:
        ours = inverse.get(upstream, upstream)
        # Either spelling counts: the alias table exists because the two sources
        # name the same vendor differently, and LiteLLM sides with either one.
        if ours in known or upstream in known:
            wanted.add(ours)
    return wanted


def _limit(entry: dict) -> dict | None:
    """The row's ``limit`` with only positive integer fields, or None."""
    limit = entry.get("limit")
    if not isinstance(limit, dict):
        return None
    kept = {k: int(v) for k, v in limit.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0}
    return kept or None


def _row(entry: dict, keep: tuple[str, ...]) -> dict:
    row = {k: entry[k] for k in keep if k in entry}
    limit = _limit(entry)
    if limit is None:
        row.pop("limit", None)
    elif "limit" in row:
        row["limit"] = limit
    return row


def build(providers: dict[str, dict], *, wanted: set[str]) -> dict:
    """One section per reachable provider, keyed the way the runtime asks.

    The section key is the normalized provider name (``nano_gpt``), not
    models.dev's or LiteLLM's spelling (``nano-gpt``): ``split_model_id``
    normalizes the prefix of every stored id, and ``catalog._provider_row``
    looks the section up by that -- a hyphenated key is a section nothing can
    read, which is how a third of the file was once dead weight.
    """
    from opendde_harness.providers.registry import normalize_provider_name

    out: dict[str, dict] = {}
    for name in sorted(wanted):
        upstream = providers.get(upstream_name(name))
        if not upstream:
            continue
        models = {model_id: _row(entry, KEEP) for model_id, entry in (upstream.get("models") or {}).items()}
        if models:
            out[normalize_provider_name(name)] = {"models": models}
    return out


def build_canonical(canonical: dict[str, dict]) -> dict[str, dict]:
    """The native limit and reasoning flag per vendor model id, where stated."""
    out: dict[str, dict] = {}
    for ref, entry in canonical.items():
        row = _row(entry, KEEP_CANONICAL)
        if row.get("limit"):
            out[ref] = row
    return out


def read_catalogue(archive: bytes) -> tuple[dict[str, dict], dict[str, dict]]:
    """Parse the repository into api.json's provider shape plus the canonical rows.

    One ``provider.toml`` names the vendor; everything under ``models/`` is one
    row keyed by its path, because the id a vendor publishes can itself contain a
    slash -- a gateway files ``moonshotai/Kimi-K2.6`` two directories deep, and
    reading only the flat level drops every one of them (siliconflow and
    openrouter are entirely nested, so both came back empty).
    """
    providers: dict[str, dict] = {}
    #: ``models/<vendor>/<id>.toml`` -- the vendor's own definition of a model,
    #: shared by every provider that resells it. A provider row states only what
    #: differs and points ``base_model`` here for the rest.
    canonical: dict[str, dict] = {}

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".toml"):
                continue
            parts = member.name.split("/")
            if len(parts) < 4 or parts[1] not in {"providers", "models"}:
                continue
            handle = tar.extractfile(member)
            if handle is None:  # pragma: no cover - directories filtered above
                continue
            try:
                data: dict[str, Any] = tomllib.loads(handle.read().decode("utf-8"))
            except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
                print(f"  skipping {member.name}: {exc}", file=sys.stderr)
                continue

            if parts[1] == "models":
                canonical["/".join(parts[2:]).removesuffix(".toml")] = data
                continue

            vendor = providers.setdefault(parts[2], {"models": {}})
            if parts[3] == "provider.toml":
                vendor["name"] = data.get("name") or parts[2]
            elif parts[3] == "models" and len(parts) > 4:
                vendor["models"]["/".join(parts[4:]).removesuffix(".toml")] = data

    resolve_inheritance(providers, canonical)
    return providers, canonical


def resolve_inheritance(providers: dict[str, dict], canonical: dict[str, dict]) -> None:
    """Fill in what a row inherits from the model it declares as its base.

    A provider reselling someone else's model states only what differs -- its own
    price, or a smaller window -- and points ``base_model`` at the vendor's
    definition for the rest. 2915 of the catalogue's rows are written that way,
    including every one of github_copilot's and most of azure's. The published
    api.json resolves this before serving; reading the files directly does not,
    and the failure is quiet: every row is present and every total looks right,
    the rows just have no names. That is why ``tests/test_providers_catalog.py``
    asserts a label per provider and not a count.

    The merge follows the upstream generator (``mergeBaseModel``): the row's
    own fields win, and a nested table such as ``[limit]`` or ``[cost]`` merges
    key by key rather than replacing the base's, so a provider that restates
    only ``limit.context`` keeps the inherited ``limit.output``. A top-level
    replace lost that field on 298 rows and answered ``openrouter``'s
    ``deepseek-v4-flash`` with a 16384-token ceiling against a real 384000.
    ``base_model_omit`` then removes inherited dot-paths the provider says do
    not apply to it (``limit.input`` on most of the 95 rows that use it).
    """
    resolved: dict[str, dict] = {}

    def entry_for(ref: str) -> dict | None:
        # The shared definition first: `base_model = "anthropic/claude-opus-5"`
        # names the vendor's model, which is a different file from that vendor's
        # own provider row and is the only place some of them exist.
        if ref in canonical:
            return canonical[ref]
        provider, _, model_id = ref.partition("/")
        return providers.get(provider, {}).get("models", {}).get(model_id)

    def resolve(ref: str, seen: frozenset[str]) -> dict:
        if ref in resolved:
            return resolved[ref]
        entry = entry_for(ref)
        if entry is None:
            return {}
        base_ref = entry.get("base_model")
        # A base can itself derive, so this recurses; `seen` stops a cycle
        # from doing it forever.
        merged = entry if not base_ref or ref in seen else inherit(resolve(str(base_ref), seen | {ref}), entry)
        resolved[ref] = merged
        return merged

    for provider, vendor in providers.items():
        for model_id in list(vendor["models"]):
            ref = f"{provider}/{model_id}"
            # A provider row and the shared definition can share a ref; the row
            # is the one being resolved, so it seeds `seen` rather than being
            # looked up through `entry_for`, which would return the other file.
            entry = vendor["models"][model_id]
            base_ref = entry.get("base_model")
            if base_ref:
                entry = inherit(resolve(str(base_ref), frozenset({ref})), entry)
            vendor["models"][model_id] = entry


def inherit(base: dict, entry: dict) -> dict:
    """``entry`` laid over ``base``, tables merged key by key, then omits applied."""
    overrides = {k: v for k, v in entry.items() if k not in ("base_model", "base_model_omit")}
    inherited = {k: v for k, v in base.items() if k not in ("base_model", "base_model_omit")}
    # A deep copy, not a shared reference: the omit below mutates, and a
    # table inherited untouched is the canonical row's own dict.
    merged = _deep_merge(copy.deepcopy(inherited), overrides)
    for path in entry.get("base_model_omit") or []:
        _omit(merged, str(path).split("."))
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            base[key] = _deep_merge(current, value)
        else:
            base[key] = value
    return base


def _omit(target: dict, parts: list[str]) -> None:
    for part in parts[:-1]:
        target = target.get(part)
        if not isinstance(target, dict):
            return
    target.pop(parts[-1], None)


def _fetch(url: str, *, accept: str = "*/*") -> bytes:
    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "opendde-harness-model-catalog-refresh", "Accept": accept}
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - a pinned https URL
        return response.read()


def main() -> int:
    print(f"resolving {REPO}@{REF} ...")
    sha = json.loads(_fetch(COMMIT_API, accept="application/vnd.github+json"))["sha"]
    tarball = tarball_url(sha)
    print(f"fetching {tarball} ...")
    providers, canonical = read_catalogue(_fetch(tarball))
    print(f"  catalogue: {len(providers)} providers, {len(canonical)} canonical models")

    wanted = reachable_providers(providers)
    snapshot = build(providers, wanted=wanted)
    native = build_canonical(canonical)
    from opendde_harness.providers.registry import normalize_provider_name

    missing = sorted(w for w in wanted if normalize_provider_name(w) not in snapshot)

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    # Sorted: the upstream returns models in an unstable order, so an unsorted
    # dump makes every refresh a diff of the whole file with no change in it.
    # The sha rides along so a snapshot can be traced to the commit it came from.
    payload = {"_source": {"repo": REPO, "ref": REF, "sha": sha}, "_models": native, **snapshot}
    SNAPSHOT.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    size_mib = SNAPSHOT.stat().st_size / 1048576
    models = sum(len(v["models"]) for v in snapshot.values())
    print(
        f"wrote {SNAPSHOT.relative_to(Path.cwd())}: {len(snapshot)} providers, {models} models, "
        f"{len(native)} native limits, {size_mib:.3f} MiB"
    )
    if missing:
        print(f"  reachable but not in the catalogue ({len(missing)}): {', '.join(missing)}")
    if size_mib > 1:
        print("ERROR: over the 1 MiB gate; trim KEEP or PROVIDER_ALIASES", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
