"""Host and workflow skill access share loading without widening task scope."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from opendde_harness.agent.tools.use_skill import UseSkillTool
from opendde_harness.memory_engine.skill_forge.documents import SkillDocument, SkillDocumentCatalog
from opendde_harness.memory_engine.skill_forge.loader import SkillLoader
from opendde_harness.memory_engine.skill_forge.memory_source import MemorySkillSource
from opendde_harness.memory_engine.skill_forge.router import SkillForgeRouter
from opendde_harness.plugin.protein_design.core.memory import DesignMemory


def document(name="chosen", source="builtin", native_id=None):
    return SkillDocument(
        name,
        Path("/unused"),
        "---\ndescription: |\n  A short\n  description\n---\nBODY",
        source=source,
        native_id=native_id,
    )


def test_snapshot_is_an_allowlist_not_a_registry_fallback():
    registry = SimpleNamespace(get=lambda *args, **kwargs: pytest.fail("must not consult global registry"))
    loader = SkillLoader(registry, documents=[document()])
    assert loader.load("local/chosen") == loader.load("chosen")
    for forbidden in ("local/other", "memory/chosen", "../chosen"):
        with pytest.raises(ValueError, match="disallowed"):
            loader.load(forbidden)
    with pytest.raises(ValueError, match="disallowed"):
        SkillLoader(registry, documents=[]).load("chosen")


def test_memory_documents_do_not_touch_disk_or_cross_sources():
    doc = document("advice", "memory", "record-123")
    loader = SkillLoader(documents=[doc])
    loaded = loader.load("memory/record-123")
    assert loaded == loader.load("advice")
    assert loaded.skill_dir is None
    assert "BODY" in loaded.render()
    with pytest.raises(ValueError):
        loader.load("local/advice")


@pytest.mark.parametrize("blocked", ["chosen", "local/chosen"])
def test_blocklist_applies_to_qualified_and_alias_ids(blocked):
    loader = SkillLoader(documents=[document()], blocklist=[blocked])
    for identifier in ("chosen", "local/chosen"):
        with pytest.raises(ValueError, match="blocklist"):
            loader.load(identifier)


def test_host_tool_uses_same_loader_and_keeps_resources(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "guide.md").write_text("guide")
    body = "Read [guide](references/guide.md)."
    meta = SimpleNamespace(name="chosen", content=body, path=tmp_path / "SKILL.md")
    registry = SimpleNamespace(get=lambda *args, **kwargs: meta)
    expected = SkillLoader(registry).load("local/chosen").render()
    actual = asyncio.run(UseSkillTool(registry).execute("local/chosen"))
    assert actual == expected
    assert str(tmp_path / "scripts") in actual
    assert str(tmp_path / "references" / "guide.md") in actual


def test_memory_lookup_does_not_fall_back_to_local_registry():
    calls = []

    def get(name, source=None):
        calls.append((name, source))
        if source == "memory":
            return None
        pytest.fail("memory lookup must not resolve unrelated local skill")

    result = asyncio.run(UseSkillTool(SimpleNamespace(get=get)).execute("memory/missing"))
    assert result.startswith("Error:")
    assert calls == [("missing", "memory")]


def test_shared_catalog_uses_multiline_description_and_filters_role():
    first = document()
    other = SkillDocument("advice", Path("memory/x"), "BODY", source="memory", roles=("quality",))
    catalog = SkillDocumentCatalog([first, other])
    assert first.as_hit().meta["description"] == "A short description"
    assert catalog.for_role("design", catalog.names()) == (first,)
    assert catalog.select(["chosen", "chosen"]) == (first,)


def test_design_skill_retrieval_preserves_scope_and_uses_public_router(monkeypatch):
    backend = SimpleNamespace(
        recall=AsyncMock(side_effect=AssertionError("unscoped recall forbidden")),
        recall_scoped=AsyncMock(
            return_value=[
                SimpleNamespace(
                    text="skill instructions", score=1.0, metadata={"id": "record", "name": "advice", "type": "skill"}
                ),
                SimpleNamespace(text="ordinary fact", score=0.5, metadata={"type": "fact"}),
            ]
        ),
    )
    calls = []
    original = SkillForgeRouter.select

    async def select(self, *args, **kwargs):
        calls.append(True)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(SkillForgeRouter, "select", select)
    result = asyncio.run(DesignMemory(backend, agent_id="designer").retrieve_skills("target-A", "lessons"))
    assert calls == [True]
    assert [item.native_id for item in result] == ["record"]
    assert backend.recall_scoped.call_args.kwargs["app_id"] == "protein-design"
    assert backend.recall_scoped.call_args.kwargs["project_id"] == "target-A"
    assert backend.recall_scoped.call_args.kwargs["agent_id"] == "designer"
    assert backend.recall_scoped.call_args.args[0] == "target=target-A; lessons"


def test_memory_source_unscoped_default_is_unchanged_and_failures_degrade():
    backend = SimpleNamespace(recall=AsyncMock(return_value=[]))
    asyncio.run(MemorySkillSource(backend, "host").search("query", [], 3))
    backend.recall.assert_awaited_once_with("query", agent_id="host", top_k=3)
    backend.recall.side_effect = RuntimeError("offline")
    assert asyncio.run(DesignMemory(backend, agent_id="designer").retrieve_skills("target", "query")) == []
