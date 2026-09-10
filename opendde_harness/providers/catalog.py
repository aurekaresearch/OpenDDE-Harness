"""What a model is called, what it costs, and how much context it takes.

A picker showing `anthropic/claude-sonnet-4-6` is showing an identifier. What a
person choosing a model wants is its name, roughly what it is good at, how much
context it takes and how recent it is -- none of which LiteLLM's table carries,
because that table exists to price and route.

So there are two catalogue sources, and the ladder in ``providers.rates`` asks
them in a fixed order:

* LiteLLM's own table decides prices and limits used in a request. It ships with
  the dependency and needs no network. It also carries capability flags, which
  OpenDDE Harness deliberately does not read: its `supports_prompt_caching` asks whether a
  model caches at all, while what a request needs to know is whether the provider
  accepts `cache_control` blocks -- `ProviderSpec`'s field of the same name.
* the models.dev snapshot decides labels, prices a finished call, and answers
  for a window LiteLLM's table does not carry. It holds a row per provider
  (name, description, the provider's own cost, and the limit *as that provider
  serves the model*) and, under ``_models``, the vendor's native limit per
  canonical model id -- an upper bound, asked only after the provider row.

The snapshot answering for a window at all is a deliberate change of position.
It used to be kept out on the grounds that a community-maintained file which
goes stale or wrong would shape the next request rather than cost a label. What
that bought in practice was a constant: every model the LiteLLM table lacked --
which included the shipped default -- was sized at 65,536 tokens, which is
wrong for all of them and drove the trimmer and the curator's slow path at a
fraction of the real window. The concern that motivated the exclusion -- a
self-hosted ``hosted_vllm/qwen3-32b`` reading OpenRouter's ``qwen/qwen3-32b``
-- is real, and it is met by the matching rule rather than by discarding the
data: every lookup here is an exact key match on the id as stored, with no
cross-provider, wire-prefix or basename fallback, so a deployment the snapshot
has no row for is answered with nothing and the caller says so.

The snapshot ships with OpenDDE Harness so a fresh install works offline and tests never
reach the network; the machines it has to run on cannot reach the catalogue's
site, so nothing at runtime fetches it. Regenerate with
``scripts/refresh_models_dev_snapshot.py`` and commit the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opendde_harness.config.schema import ModelOverlay

SNAPSHOT = Path(__file__).parent / "data" / "models_dev.json"

#: Where a row's facts came from, kept on the row so a surface can tell a
#: label it can trust from an id it is falling back to.
SOURCE_SNAPSHOT = "snapshot"
SOURCE_ID_ONLY = "id-only"
#: The user described it themselves, which beats any catalogue.
SOURCE_OVERLAY = "overlay"


@dataclass(frozen=True)
class ModelRow:
    """One model, as a person reads it."""

    ref: str
    provider: str
    label: str
    source: str
    description: str = ""

    @property
    def described(self) -> bool:
        return self.source != SOURCE_ID_ONLY


@lru_cache(maxsize=1)
def _snapshot() -> dict[str, dict]:
    """The bundled labels, or nothing when they cannot be read.

    Never raises: a missing or corrupt snapshot must cost labels, not startup.
    """
    try:
        return json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - only on a damaged install
        logger.debug(f"model label snapshot unavailable: {exc}")
        return {}


def describe(provider: str, model: str, *, overlay: "ModelOverlay | None" = None) -> ModelRow:
    """Everything known about this model for display purposes.

    Falls back to the id as its own label, so a caller can render the result
    unconditionally: a model the snapshot has never heard of -- one released
    since the last refresh, or served by a local deployment -- still comes back
    as a row rather than as nothing to show.
    """
    from opendde_harness.providers.registry import canonical_provider_name
    from opendde_harness.providers.wire import split_model_id, stored_model_id

    provider = canonical_provider_name(provider)
    ref = stored_model_id(provider, model)
    entry = _snapshot().get(provider, {}).get("models", {}).get(_vendor_id(provider, model))

    if not entry:
        # A relay or a self-hosted server serves the vendor's model under the
        # vendor's own name; the name is what the user reads, so the label
        # comes from whichever provider row carries it.
        entry = _any_provider_row(_vendor_id(provider, model))

    if entry:
        row = ModelRow(
            ref=ref,
            provider=provider,
            label=entry.get("name") or ref,
            source=SOURCE_SNAPSHOT,
            description=entry.get("description") or "",
        )
    else:
        row = ModelRow(ref=ref, provider=provider, label=split_model_id(ref)[1] or ref, source=SOURCE_ID_ONLY)

    return _with_overlay(row, overlay)


def model_cost(model: str) -> dict | None:
    """The vendor's own published rates for this model, or None.

    Keyed by provider, which is the point: reading a price out of a flat
    cross-vendor table answers a self-hosted deployment with a hosted vendor's
    figure. ``model`` is a stored id, so the provider it names is the one asked.
    Costs are published per million tokens.
    """
    entry = _provider_row(model)
    cost = entry.get("cost") if entry else None
    return cost if isinstance(cost, dict) else None


#: Rows for a provider models.dev does not list, kept by hand the way pi
#: keeps its Codex catalogue: the ChatGPT backend serves OpenAI's models
#: behind a 272k window (128k for the spark tier) and a 128k output ceiling,
#: whatever the API's own row for the same id says -- and the API row says
#: 1,050,000, which is the direction that refuses requests. Exact ids only,
#: like every other table here: a slug not listed is unknown, not 272k.
#: Keyed by the registry's provider name, then by the vendor id.
_CODEX_LIMIT = {"context": 272_000, "output": 128_000}
_CODEX_SPARK_LIMIT = {"context": 128_000, "output": 128_000}
_BUILTIN_ROWS: dict[str, dict[str, dict]] = {
    "openai_codex": {
        "gpt-6-astra": {"limit": _CODEX_LIMIT, "reasoning": True},
        "gpt-5.3-codex-spark": {"limit": _CODEX_SPARK_LIMIT, "reasoning": True},
        "gpt-5.4": {"limit": _CODEX_LIMIT, "reasoning": True},
        "gpt-5.4-mini": {"limit": _CODEX_LIMIT, "reasoning": True},
        "gpt-5.5": {"limit": _CODEX_LIMIT, "reasoning": True},
        "gpt-5.6-luna": {"limit": _CODEX_LIMIT, "reasoning": True},
        "gpt-5.6-sol": {"limit": _CODEX_LIMIT, "reasoning": True},
        "gpt-5.6-terra": {"limit": _CODEX_LIMIT, "reasoning": True},
    },
}


def builtin_limit(model: str) -> dict | None:
    """The limit this harness declares for a provider no catalogue lists, or None."""
    return _limit_of(_builtin_row(model))


def _builtin_row(model: str) -> dict | None:
    from opendde_harness.providers.registry import find_by_model, split_model_id

    if not split_model_id(model)[0]:
        return None
    spec = find_by_model(model)
    rows = _BUILTIN_ROWS.get(spec.name) if spec else None
    return rows.get(_vendor_id(spec.name, model)) if rows else None


def served_limit(model: str) -> dict | None:
    """The limit the provider named by this id serves the model with, or None.

    A reseller may cap a model's window below the vendor's own figure, and
    models.dev records that on the provider's row -- so this is asked before
    :func:`native_limit`, and only ever about the provider the id names.
    """
    return _limit_of(_provider_row(model))


def native_limit(model: str) -> dict | None:
    """The vendor's own limit for this model, from the canonical table, or None.

    Matched by exact key first, over the ids LiteLLM's table files this model
    under -- the stored id, or the vendor's own spelling for a regional
    instance of that vendor. Then by the model's own name: a relay or a
    self-hosted server serves ``deepseek-v4-flash`` under its own prefix,
    and the prefix is only where it is reached, not what it is. The name is
    matched exactly (no case folding) and only when one vendor carries it,
    which every name in the bundled table satisfies. A relay that caps a
    model below the vendor's window declares the cap in the model's
    overlay, which is read before this. Never through the wire id: that
    names the driver, and ``custom/x`` going out as ``openai/x`` says
    nothing about ``x``.
    """
    return _limit_of(_native_row_for(model))


def _native_row_for(model: str) -> dict | None:
    from opendde_harness.providers.wire import metadata_candidates

    if not model:
        return None
    table = _snapshot().get("_models")
    if not isinstance(table, dict):
        return None
    for key in metadata_candidates(model):
        entry = table.get(key)
        if isinstance(entry, dict):
            return entry
    return _bare_row(model)


def _bare_row(model: str) -> dict | None:
    """The canonical row whose model name is this prefixed id's own, or None."""
    from opendde_harness.providers.registry import split_model_id

    provider, name = split_model_id(model)
    if not provider or not name:
        return None
    return _bare_index().get(name)


_bare_cache: tuple[int, dict[str, dict]] | None = None


def _bare_index() -> dict[str, dict]:
    """Model name -> canonical row, for the names exactly one vendor carries."""
    global _bare_cache
    table = _snapshot().get("_models")
    if not isinstance(table, dict):
        return {}
    # Keyed on the table object, so a test that swaps the snapshot is indexed too.
    if _bare_cache is not None and _bare_cache[0] == id(table):
        return _bare_cache[1]
    rows: dict[str, dict | None] = {}
    for key, entry in table.items():
        _, _, name = key.partition("/")
        if not name or not isinstance(entry, dict):
            continue
        rows[name] = None if name in rows else entry
    index = {name: entry for name, entry in rows.items() if entry is not None}
    _bare_cache = (id(table), index)
    return index


def _any_provider_row(name: str) -> dict | None:
    """The first provider row carrying this model name, for its label only."""
    for section, block in _snapshot().items():
        if section == "_models" or not isinstance(block, dict):
            continue
        entry = (block.get("models") or {}).get(name)
        if isinstance(entry, dict):
            return entry
    return None


def model_reasoning(model: str) -> bool | None:
    """Whether the catalogue says this model can reason; None when it has no row.

    Provider row first, vendor row second, like the limits: a relay may serve
    a family's thinking variant under its own id, and the vendor's row is the
    answer for a provider the catalogue does not carry.
    """
    from opendde_harness.providers.wire import metadata_candidates

    rows = (
        _builtin_row(model),
        _provider_row(model),
        *(_native_row(key) for key in metadata_candidates(model or "")),
        _bare_row(model or ""),
    )
    for entry in rows:
        if isinstance(entry, dict) and isinstance(entry.get("reasoning"), bool):
            return entry["reasoning"]
    return None


def _native_row(key: str) -> dict | None:
    table = _snapshot().get("_models")
    entry = table.get(key) if isinstance(table, dict) else None
    return entry if isinstance(entry, dict) else None


def _limit_of(entry: dict | None) -> dict | None:
    limit = entry.get("limit") if isinstance(entry, dict) else None
    return limit if isinstance(limit, dict) and limit else None


def _provider_row(model: str) -> dict | None:
    """This model's row under the provider its id names, or None.

    The id has to name its provider. `find_by_model` falls back to keyword
    matching for a bare id, which reads across vendors: "qwen3-32b" matched
    DashScope and was priced at DashScope's rate whoever was actually serving
    it. That is the same borrowing the openrouter tier was gated to stop, one
    tier down.

    A bare id left by an older version therefore reads as unknown. That is not
    worth a compatibility path: both surfaces that write a model now store it
    qualified, so picking the model once restores the figure.
    """
    from opendde_harness.providers.registry import canonical_provider_name, find_by_model, split_model_id

    if not split_model_id(model)[0]:
        return None
    spec = find_by_model(model)
    provider = spec.name if spec else split_model_id(model)[0]
    if not provider:
        return None
    entry = _snapshot().get(canonical_provider_name(provider), {}).get("models", {}).get(_vendor_id(provider, model))
    return entry if isinstance(entry, dict) else None


def overlay_for(overlays: "Mapping[str, ModelOverlay]", model: str) -> "ModelOverlay | None":
    """The overlay declared for this stored id, from a map keyed by ``merge_key``.

    The provider is the one the id names, resolved the way routing resolves it,
    so a declaration under a section is found for every spelling of that
    section's ids -- and never for another provider's, which is what keeps a
    window declared for a self-hosted model from answering for a hosted one of
    the same name.
    """
    from opendde_harness.providers.registry import find_by_model, split_model_id
    from opendde_harness.providers.wire import merge_key

    if not overlays or not model:
        return None
    spec = find_by_model(model)
    provider = spec.name if spec else split_model_id(model)[0]
    if not provider:
        return None
    return overlays.get(merge_key(provider, model))


def _with_overlay(row: ModelRow, overlay: "ModelOverlay | None") -> ModelRow:
    """Let what the user stated beat what a catalogue guessed.

    The user is describing their own deployment, so they are the authority on
    it -- and for a model no catalogue carries, they are the only one. Fields
    left unset in the overlay keep the catalogue's answer rather than blanking
    it, so stating one fact does not erase the rest.
    """
    if overlay is None:
        return row

    changed = {
        "label": overlay.label or row.label,
        "description": overlay.description or row.description,
    }
    described = row.described or bool(overlay.label or overlay.description)
    return replace(row, **changed, source=SOURCE_OVERLAY if described else row.source)


def _vendor_id(provider: str, model: str) -> str:
    """The vendor's own id, which is how the snapshot is keyed.

    A stored id names its provider and the snapshot does not repeat that, so the
    prefix comes off before the lookup -- including a gateway's, whose rows are
    filed under the upstream vendor's id.

    ``head`` is always normalized (``split_model_id`` runs it through
    ``normalize_provider_name``), so ``provider`` must be too before the
    fallback comparison -- the same normalization ``wire.merge_key`` applies to
    both sides of its own identity check. Comparing raw missed a provider
    OpenDDE Harness carries no spec for whenever it was spelled differently from its
    model prefix, e.g. hyphenated ``provider`` against an underscored prefix.
    """
    from opendde_harness.providers.registry import find_by_name, normalize_provider_name
    from opendde_harness.providers.wire import split_model_id

    spec = find_by_name(provider)
    head, rest = split_model_id(model or "")
    if head and spec and head in spec.route_names:
        return rest
    if head and head == normalize_provider_name(provider):
        return rest
    return model or ""
