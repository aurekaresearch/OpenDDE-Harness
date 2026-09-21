"""Every workflow role uses the shared engine with isolated, lazy skill context."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from opendde_harness.context_engine.assembler import ContextAssembler
from opendde_harness.context_engine.history_trimmer import ContextBudgetError
from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole, profile_for
from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
from opendde_harness.plugin.protein_design.agents.skills import SkillDocument
from opendde_harness.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from opendde_harness.providers.messages import text_of


class Output(BaseModel):
    report: str


class Provider(LLMProvider):
    def __init__(self, responses, window=None):
        self.responses = iter(responses)
        self.sent = []
        self.window = window

    def context_window(self):
        return self.window

    def get_default_model(self):
        return "test"

    async def chat(self, messages, **kwargs):
        self.sent.append(deepcopy(messages))
        return next(self.responses)

    async def chat_with_retry(self, messages, **kwargs):
        return await self.chat(messages, **kwargs)


def skill(name="chosen", body="UNIQUE SKILL BODY"):
    return SkillDocument(name, Path("/unused"), f"---\ndescription: Short summary\n---\n{body}", ())


def load(name="chosen"):
    return LLMResponse(None, tool_calls=[ToolCallRequest("load-1", "use_skill", {"skill_id": name})])


def answer():
    return LLMResponse('{"report":"ok"}')


@pytest.mark.parametrize("role", list(AgentRole))
@pytest.mark.parametrize("design_type", ["vhh", "minibinder"])
def test_all_roles_use_engine_and_load_only_selected_skill(role, design_type, monkeypatch):
    assembled = []
    original = ContextAssembler.assemble

    async def capture(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        assembled.append(deepcopy(result))
        return result

    monkeypatch.setattr(ContextAssembler, "assemble", capture)
    provider = Provider([load(), answer(), load(), answer()])
    session = OpenDDEHarnessStructuredSession(provider, "test")
    profile = replace(profile_for(role, design_type), output_schema=Output)

    async def run():
        for cycle in (1, 2):
            await session.run(profile, f"cycle {cycle}", skills=[skill(), skill(f"unused-{cycle}", "UNUSED BODY")])

    asyncio.run(run())
    assert len(assembled) == 2
    assert all(item.metadata["engine"] == "context_assembler" for item in assembled)
    first, loaded, second, _ = provider.sent
    assert first[0] == second[0]
    assert len(first) == len(second) == 2
    assert "cycle 1" not in str(second)
    assert "UNIQUE SKILL BODY" not in str(first)
    assert "UNIQUE SKILL BODY" in str(loaded)
    assert "UNUSED BODY" not in str(provider.sent)
    assert "Short summary" in text_of(first[1])
    assert "# Available tools" not in text_of(first[0])
    assert text_of(first[1]).startswith("cycle 1")
    assert assembled[0].messages == first


def test_budget_is_checked_again_after_loading_skill():
    provider = Provider([load(), answer()], window=2000)
    profile = replace(AGENT_PROFILES[AgentRole.ANALYZE], output_schema=Output, max_tokens=100)
    session = OpenDDEHarnessStructuredSession(provider, "test")
    with pytest.raises(ContextBudgetError):
        asyncio.run(session.run(profile, "evidence", skills=[skill(body="large result " * 6000)]))
    assert len(provider.sent) == 1


def test_oversized_initial_context_does_not_reach_provider():
    provider = Provider([answer()], window=1000)
    profile = replace(AGENT_PROFILES[AgentRole.ANALYZE], output_schema=Output, max_tokens=100)
    with pytest.raises(ContextBudgetError):
        asyncio.run(OpenDDEHarnessStructuredSession(provider, "test").run(profile, "evidence " * 5000))
    assert not provider.sent


def test_unknown_window_does_not_invent_a_limit():
    provider = Provider([answer()])
    profile = replace(AGENT_PROFILES[AgentRole.ANALYZE], output_schema=Output)
    asyncio.run(OpenDDEHarnessStructuredSession(provider, "test").run(profile, "evidence " * 5000))
    assert len(provider.sent) == 1


def test_concurrent_roles_do_not_share_context():
    provider = Provider([answer(), answer()])
    session = OpenDDEHarnessStructuredSession(provider, "test")

    async def run():
        await asyncio.gather(
            *(
                session.run(replace(AGENT_PROFILES[role], output_schema=Output), role.value)
                for role in (AgentRole.ANALYZE, AgentRole.QUALITY)
            )
        )

    asyncio.run(run())
    assert len(provider.sent) == 2
    assert {text_of(messages[1]) for messages in provider.sent} == {"analyze", "quality"}
    assert all(len(messages) == 2 for messages in provider.sent)
    assert provider.sent[0][0] != provider.sent[1][0]
