"""G4 — a default turn makes one model request per assistant message, and no more.

The skill catalogue used to be chosen by two model calls in front of every
turn: a query rewriter and an LLM gate. Both are gone, and so is the context
curator's slow path. What is left has to be provable from the outside, which is
what this is: every request the service was handed is counted, and the count is
the turn's own assistant messages. A gate, a rewriter or a curator coming back
adds a request that no assistant message answers for.

The catalogue is also deterministic now -- BM25 only above forty skills, and
never a model call -- so the block it renders is the same bytes on the second
turn as on the first.
"""

from __future__ import annotations

import pytest

from opendde_harness.providers import messages as msg
from tests._gate import (
    MODEL,
    Echo,
    Events,
    block_of,
    declare,
    gate_config,
    loop_for,
    no_injections,
    request,
    requires_service,
    service,  # noqa: F401  -- the fixture the gate shares
    system_prompt,
)

pytestmark = [requires_service, pytest.mark.usefixtures("service")]

KEY = "tui:aux"


def _workspace(tmp_path):
    """A workspace with one skill of its own, so the catalogue is not empty.

    An empty ``# Skills`` block would make the byte-identity assertion true for
    the wrong reason.
    """
    from opendde_harness.config.paths import get_workspace_storage

    skill = get_workspace_storage(tmp_path).skills / "assay-notes"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: assay-notes\ndescription: how this project records assay results\n---\n\nWrite them down.\n",
        encoding="utf-8",
    )
    return tmp_path


async def test_one_request_per_assistant_message_and_nothing_else(tmp_path, service):  # noqa: F811
    """No auxiliary call anywhere in the turn, and no compaction request."""
    config = gate_config(
        _workspace(tmp_path), model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 2}
    )
    loop = loop_for(service, config, tools=[Echo()])
    events = Events()

    try:
        await loop.run_turn(request("ping", KEY), events, no_injections, stream=True)
    finally:
        await loop.close_mcp()

    stored = loop.sessions.get_or_create(KEY).messages
    assistants = [m for m in stored if m.get("role") == msg.ASSISTANT]
    assert len(assistants) == 3, "two tool-calling calls and the wrap-up, which is the whole turn"
    assert service.streams == len(assistants), "one request per assistant message: no gate, no rewriter, no curator"
    assert service.compacted == [], "nothing asked the backend for a summary"
    assert {asked for asked in service.models_asked} == {("faux", "echo")}, "every request went to the turn's model"


async def test_the_skills_block_is_the_same_bytes_on_the_next_turn(tmp_path, service):  # noqa: F811
    """A deterministic catalogue renders deterministically. A model call in
    front of the block would make the second turn's bytes depend on what a
    model said about the second question."""
    config = gate_config(
        _workspace(tmp_path), model=MODEL, providers=declare("faux"), defaults={"maxToolIterations": 1}
    )
    loop = loop_for(service, config, tools=[Echo()])
    events = Events()

    try:
        await loop.run_turn(request("first question", KEY), events, no_injections, stream=True)
        first = service.streams
        await loop.run_turn(request("a completely different second question", KEY), events, no_injections, stream=True)
    finally:
        await loop.close_mcp()

    blocks = [block_of(system_prompt(context), "# Skills") for context in service.contexts]
    assert blocks[0], "the catalogue was advertised at all"
    assert "assay-notes" in blocks[0], "the workspace's own skill is in it"
    assert blocks[0] == blocks[first], "the same bytes, whatever the second question asked"
    assert len(set(blocks)) == 1, "and the same on every request of both turns"
