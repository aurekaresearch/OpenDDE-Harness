"""LocalSkillCatalog — the single owner of the local skill pool.

Absorbs what used to be ``SkillService``: it builds the
:class:`SkillRegistry` + :class:`LocalPool`, runs the SKILL.md file
watcher, and renders skills for the prompt (always-skills, injection,
XML summary).

Retrieval is **not** here — that lives in :class:`LocalSkillSource`
(which reuses this catalog's ``pool`` + ``registry``) and is fused
with the remote sources by :class:`SkillForgeRouter`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opendde_harness.memory_engine.skill_local.local_pool import LocalPool
from opendde_harness.memory_engine.skill_local.registry import SkillRegistry
from opendde_harness.memory_engine.skill_local.types import SkillMeta

log = logging.getLogger(__name__)


def normalize_blocklist(names: "Iterable[str] | None") -> frozenset[str]:
    """Casefold + trim a user-supplied blocklist into a match set."""
    if not names:
        return frozenset()
    return frozenset(n.strip().casefold() for n in names if n and n.strip())


def is_blocked(blocklist: frozenset[str], *identifiers: str | None) -> bool:
    """True when any identifier (name / slug / native id) is on the blocklist."""
    return any(i and i.strip().casefold() in blocklist for i in identifiers)


if TYPE_CHECKING:
    from collections.abc import Iterable

    from opendde_harness.memory_engine.skill_local.watcher import SkillFileWatcher
    from opendde_harness.providers.base import LLMProvider


class LocalSkillCatalog:
    """File access + directory rendering over the local skill pool."""

    def __init__(
        self,
        workspace: Path,
        config: Any = None,
        builtin_skills_dir: Path | None = None,
        llm_provider: "LLMProvider | None" = None,  # accepted for caller compat; unused
        *,
        start_watcher: bool = True,
        skills_dir: Path | None = None,
    ):
        # R1: build extra_dirs from config.local_dirs. List order = priority
        # (later overrides earlier on name collision). Each tuple:
        # (path, display_name, always_enabled).
        extra_dirs: list[tuple[Path, str, bool]] = []
        if config is not None and getattr(config, "enabled", False):
            seen_names: dict[str, int] = {}
            for entry in getattr(config, "local_dirs", []) or []:
                entry_path = Path(os.path.expanduser(getattr(entry, "path", "") or "")).resolve()
                entry_enabled = getattr(entry, "enabled", True)
                if not entry_enabled or not entry_path or not entry_path.is_dir():
                    if entry_enabled and getattr(entry, "path", ""):
                        log.warning(
                            "local_dirs: path does not exist or is not a directory: %s",
                            entry_path,
                        )
                    continue
                entry_name = getattr(entry, "name", None) or entry_path.name
                seen_names[entry_name] = seen_names.get(entry_name, 0) + 1
                if seen_names[entry_name] > 1:
                    entry_name = f"{entry_name}_{seen_names[entry_name]}"
                entry_always_enabled = getattr(entry, "always_enabled", True)
                extra_dirs.append((entry_path, entry_name, entry_always_enabled))

        self._registry = SkillRegistry(
            workspace,
            builtin_skills_dir=builtin_skills_dir,
            extra_dirs=extra_dirs,
            scan_max_depth=int(getattr(config, "scan_max_depth", 5) if config else 5),
            skills_dir=skills_dir,
        )

        self._config = config
        self._blocklist = normalize_blocklist(getattr(config, "blocklist", None))

        # Local-pool BM25 retrieval over file-based skills (workspace +
        # builtin). Always available — no model, no GPU. Blocklisted
        # skills are hidden from the index via a filtered registry view
        # so they never surface through retrieval; the router pool drop
        # in SkillsSegmentBuilder stays the backstop for other sources.
        pool_registry: Any = self._registry
        if self._blocklist:
            skipped = sorted({m.name for m in self._registry.list_all() if self._is_blocked(m.name)})
            if skipped:
                log.info(
                    "skill blocklist: hiding blocked skill(s) from the local pool: %s",
                    ", ".join(skipped),
                )
            pool_registry = _BlocklistRegistryView(self._registry, self._blocklist)
        self._local_pool = LocalPool(pool_registry)

        # Background SKILL.md watcher. Auto-started by default so the
        # common long-lived consumer (ContextBuilder) picks up hand-edits
        # to ``<workspace>/skills/**/SKILL.md`` without anyone needing to
        # remember a separate call. The watcher runs in a daemon thread
        # so process exit cleans it up; when ``watchfiles`` is missing
        # this collapses to a no-op + one INFO log.
        #
        # Short-lived consumers (a single CLI command, a one-shot
        # ``build_skills_summary()`` for a subagent) should pass
        # ``start_watcher=False``: starting costs ~25ms, and the watcher's
        # daemon thread holds a strong reference back through the
        # ``on_change`` bound method that keeps the catalog alive past its
        # single use — a real thread/handle leak in long-lived parents
        # that spawn many subagents.
        self._file_watcher: "SkillFileWatcher | None" = None
        if start_watcher:
            self.start_file_watcher()

    # ── Pool/registry access (for LocalSkillSource) ──────────────────

    def _is_blocked(self, name: str) -> bool:
        return is_blocked(self._blocklist, name)

    @property
    def registry(self) -> SkillRegistry:
        return self._registry

    @property
    def pool(self) -> LocalPool:
        return self._local_pool

    def invalidate_skill_cache(self, source: str | None = None) -> None:
        """Invalidate the file-registry cache and refresh the local BM25 index.

        When ``source`` is provided, only that source's slice is
        rebuilt and merged back inside the registry — saves the cost of
        re-walking the unchanged builtin / workspace / external trees.
        Pass ``None`` for a hard reset (rare, only when something off-band
        has rewritten multiple sources at once).

        After the registry update, the local BM25 index is rebuilt
        eagerly so file-watcher events flow straight through to retrieval
        — ``search`` callers never pay the index build on the hot path.
        """
        if source is None:
            self._registry.invalidate_cache()
        else:
            self._registry.invalidate_source(source)
        self._local_pool.rebuild_index()

    def start_file_watcher(self) -> bool:
        """Start the background SKILL.md filesystem watcher.

        Called once automatically from :meth:`__init__`, so consumers
        normally never invoke it directly — it's still public for tests
        and for the rare case of restarting the watcher after an
        external ``stop_file_watcher`` call.

        Idempotent: the first call wires up a daemon thread
        (``watchfiles``-backed) that pipes per-source invalidations into
        :meth:`invalidate_skill_cache` whenever a SKILL.md is added,
        edited or removed under ``<workspace>/skills``. Subsequent calls
        are no-ops and return ``False``.

        Returns ``True`` only when a new watcher thread is now running.
        Returns ``False`` — and the rest of the catalog still works in
        manual-invalidation mode — when:

          - ``watchfiles`` is missing from the install (it's a declared
            dependency, so this only happens in stripped / partial
            installs),
          - the workspace skills directory does not exist,
          - a watcher is already running.

        Builtin / external layers are intentionally **not** watched:
        they are read-only mirrors in this codebase, and the builtin
        layer can carry ~80K files — recursive watching would blow past
        Linux's default inotify watch limit.

        Never raises.
        """
        if self._file_watcher is not None:
            return False
        from opendde_harness.memory_engine.skill_local.watcher import SkillFileWatcher

        watcher = SkillFileWatcher(
            roots=[self._registry.workspace_skills],
            on_change=self.invalidate_skill_cache,
            resolve_source=self._registry.resolve_source_for_path,
        )
        if not watcher.start():
            # Failure to start (missing dep / missing root) — leave
            # ``_file_watcher`` unset so a later call can retry once
            # the prerequisite is fixed (e.g. workspace materialized).
            return False
        self._file_watcher = watcher
        return True

    def stop_file_watcher(self) -> None:
        """Signal the watcher thread to exit and best-effort join.

        Safe to call when no watcher was ever started.
        """
        watcher = self._file_watcher
        if watcher is None:
            return
        watcher.stop()
        self._file_watcher = None

    # ------------------------------------------------------------------
    # Legacy ``SkillsLoader`` API (signature-compatible drop-in)
    # ------------------------------------------------------------------

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """All skills as legacy-shape dicts ``{name, path, source}``."""
        metas = self._registry.list_all()
        if filter_unavailable:
            metas = [m for m in metas if self._registry.check_available(m.name, source=m.source)]
        return [{"name": m.name, "path": str(m.path), "source": m.source} for m in metas]

    def load_skill(self, name: str) -> str | None:
        """Full SKILL.md content, or ``None`` if absent."""
        return self._registry.get_body(name)

    def get_always_skills(self) -> list[SkillMeta]:
        """Skills flagged ``always: true`` whose requirements are met.

        R3: truncation order is by local_dirs list order (which maps to
        source priority in the registry) + alphabetical within each
        source. WARN lists dropped skill names.
        """
        if getattr(self._config, "disable_always", False):
            return []
        # Registry list_all already returns skills ordered by layer
        # iteration (workspace → extra_dirs in order → builtin), and
        # within each layer by discovery order.  Sort stably by
        # (source priority, name) so truncation is predictable.
        all_always = [
            m
            for m in self._registry.list_all()
            if m.always and not self._is_blocked(m.name) and self._registry.check_available(m.name, source=m.source)
        ]
        all_always.sort(key=lambda m: (m.source, m.name))
        cap = int(getattr(self._config, "always_max", 5) or 5)
        if len(all_always) > cap:
            kept = all_always[:cap]
            dropped = all_always[cap:]
            log.warning(
                "always skills (%d) exceed always_max (%d), dropped: %s",
                len(all_always),
                cap,
                ", ".join(m.name for m in dropped),
            )
            return kept
        return all_always

    def get_skill_metadata(self, name: str) -> dict | None:
        """Top-level frontmatter dict, or ``None`` if absent."""
        return self._registry.get_raw_metadata(name)

    def build_skills_summary(self, only: "list[SkillMeta] | list[str] | None" = None) -> str:
        """XML-formatted skill directory.

        Args:
            only: When provided, only these skills are included. Accepts
                either ``list[SkillMeta]`` (canonical) or ``list[str]`` of
                skill names (backward-compat for older callers). ``None``
                preserves legacy behavior (full local directory), used as
                a fallback when no selector picked.

        Local skills render via their on-disk ``meta.path``; LLM reads the
        full body via ``read``. ``available`` reflects ``requires``
        checks on the registry.
        """
        if only is None:
            metas: list[SkillMeta] = self._registry.list_all()
        else:
            # Backward-compat: accept list[str] of skill names — resolve to
            # SkillMeta via the registry, drop unknowns.
            if only and isinstance(only[0], str):
                resolved: list[SkillMeta] = []
                for name in only:
                    meta = self._lookup_meta(name, None)
                    if meta is not None:
                        resolved.append(meta)
                only = resolved
            metas = list(only)
        metas = [m for m in metas if not self._is_blocked(m.name)]
        if not metas:
            return ""

        lines: list[str] = ["<skills>"]
        for m in metas:
            name_x = _escape_xml(m.name)
            desc_x = _escape_xml(m.description or m.name)
            available = self._registry.check_available(m.name, source=m.source)
            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{name_x}</name>")
            lines.append(f"    <description>{desc_x}</description>")
            lines.append(f"    <location>{m.path}</location>")
            if not available:
                missing = self._registry.get_missing_requirements(
                    m.name,
                    source=m.source,
                )
                if missing:
                    lines.append(f"    <requires>{_escape_xml(missing)}</requires>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def gather_all_skills(self) -> list[SkillMeta]:
        """All skills visible to this catalog — the local file registry
        (workspace, builtin, external mirrors).

        Used by CLI ``skill list`` / inspection helpers — gathers
        everything unranked, deliberately including blocklisted skills so
        inspection can show a blocked-but-present skill. Hot-path
        retrieval goes through :class:`SkillForgeRouter` instead.
        """
        return self._registry.list_all()

    def _lookup_meta(
        self,
        name: str,
        source: str | None,
    ) -> SkillMeta | None:
        """Resolve a (name, source) to a ``SkillMeta`` via the local registry."""
        return self._registry.get(name, source=source)


class _BlocklistRegistryView:
    """Registry proxy handed to :class:`LocalPool` — hides blocklisted
    skills from the BM25 index while delegating everything else to the
    real registry, so index rebuilds keep flowing through unchanged."""

    def __init__(self, registry: SkillRegistry, blocklist: frozenset[str]) -> None:
        self._registry = registry
        self._blocklist = blocklist

    def list_all(self) -> list[SkillMeta]:
        return [m for m in self._registry.list_all() if not is_blocked(self._blocklist, m.name)]

    def __getattr__(self, item: str) -> Any:
        return getattr(self._registry, item)


# ----------------------------------------------------------------------
# Rendering helpers (module-level, stateless)
# ----------------------------------------------------------------------


def _escape_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


__all__ = ["LocalSkillCatalog"]
