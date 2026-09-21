"""Role-scoped segments for the shared context engine.

No conversation identity, runtime timestamps or ordinary-chat memory is injected
into a workflow agent. Its task state and allowed skill catalogue live in user
context; the role/task instructions are a stable system prefix.
"""

from dataclasses import dataclass

from opendde_harness.context_engine.base import AssemblyContext, Segment


@dataclass(frozen=True)
class WorkflowInstructions:
    text: str
    name: str = "workflow_instructions"
    order: int = 0

    async def build(self, ctx: AssemblyContext) -> Segment:
        return Segment(text=self.text)
