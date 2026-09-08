"""Explicit, non-blocking long-term memory hooks for protein design runs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from opendde_harness.memory_engine.backend import MemoryBackend
from opendde_harness.plugin.protein_design.agents.skills import LearnedSkill

logger = logging.getLogger(__name__)

_COLLAPSED_ENGLISH_TOKEN = re.compile(r"(?=[A-Za-z]{48,})(?=[A-Za-z]*[a-z])[A-Za-z]+")


class DesignMemory:
    def __init__(
        self,
        backend: MemoryBackend | None,
        *,
        agent_id: str,
        app_id: str = "protein-design",
        project_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._agent_id = agent_id
        self._app_id = app_id
        self._project_id = project_id

    async def retrieve(self, target: str, query: str, *, top_k: int = 8) -> list[str]:
        hits = await self._recall(target, query, top_k=top_k)
        return [
            hit.text for hit in hits
            if (hit.metadata or {}).get("type") != "skill" and hit.text.strip()
        ]

    async def retrieve_skills(
        self,
        target: str,
        query: str,
        *,
        top_k: int = 8,
    ) -> list[LearnedSkill]:
        hits = await self._recall(target, query, top_k=top_k)
        learned: list[LearnedSkill] = []
        for hit in hits:
            metadata = hit.metadata or {}
            if metadata.get("type") != "skill" or not hit.text.strip():
                continue
            if self._has_collapsed_english_prose(hit.text):
                logger.warning(
                    "Ignoring malformed learned skill %r: English word boundaries "
                    "were removed during generation",
                    metadata.get("name") or metadata.get("id"),
                )
                continue
            name = str(metadata.get("name") or self._frontmatter_value(hit.text, "name") or "")
            roles_value = metadata.get("roles") or self._frontmatter_value(hit.text, "role") or "design"
            roles = tuple(part.strip() for part in re.split(r"[,|]", str(roles_value)) if part.strip())
            learned.append(
                LearnedSkill(
                    name=name,
                    content=hit.text,
                    roles=roles,
                    native_id=str(metadata.get("id") or name),
                    score=float(hit.score),
                )
            )
        return learned

    async def _recall(self, target: str, query: str, *, top_k: int) -> list[Any]:
        if self._backend is None:
            return []
        try:
            scoped_recall = getattr(self._backend, "recall_scoped", None)
            if callable(scoped_recall):
                return await scoped_recall(
                    f"target={target}; {query}",
                    agent_id=self._agent_id,
                    top_k=top_k,
                    app_id=self._app_id,
                    project_id=self._project_id or target,
                )
            return await self._backend.recall(
                f"target={target}; {query}",
                agent_id=self._agent_id,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning("protein-design memory recall failed: %s", exc)
            return []

    @staticmethod
    def _frontmatter_value(content: str, key: str) -> str | None:
        match = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*([^\n]+)$", content)
        return match.group(1).strip() if match else None

    @staticmethod
    def _has_collapsed_english_prose(content: str) -> bool:
        """Reject unusable LLM output whose English spaces were stripped.

        A long mixed/lower-case alphabetic token is not valid prose.  Uppercase
        protein sequences are deliberately excluded so a skill may still quote
        a full amino-acid sequence without being rejected.
        """
        return _COLLAPSED_ENGLISH_TOKEN.search(content) is not None

    async def record(
        self,
        task_id: str,
        cycle: int,
        target: str,
        outcome: dict[str, Any],
        insights: list[str],
        *,
        finalize: bool = True,
    ) -> bool:
        if self._backend is None:
            return False
        action = outcome.get("action") if isinstance(outcome.get("action"), dict) else {}
        identity = outcome.get("identity") if isinstance(outcome.get("identity"), dict) else {}
        lesson = outcome.get("lesson") if isinstance(outcome.get("lesson"), dict) else {}
        quality = outcome.get("quality") if isinstance(outcome.get("quality"), dict) else {}
        selected_skill = str(
            outcome.get("selected_skill_id")
            or action.get("primary_skill")
            or "protein-design-proposal"
        )
        task_intent = str(
            outcome.get("task_intent")
            or f"Improve {identity.get('objective_key', 'the design objective')} "
            f"for {identity.get('target', target)} while preserving configured constraints."
        )
        learned_skill_ids = outcome.get("applied_learned_skill_ids")
        if not isinstance(learned_skill_ids, list):
            learned_skill_ids = action.get("learned_skills", [])
        tool_call_id = f"design-cycle-{cycle}"
        messages = [
            {
                "role": "user",
                "content": task_intent,
            },
            {
                "role": "assistant",
                "content": "Apply the selected design strategy and evaluate its outcome.",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": selected_skill,
                            "arguments": json.dumps(
                                {
                                    "target": target,
                                    "cycle": cycle,
                                    "applied_learned_skill_ids": learned_skill_ids,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": selected_skill,
                "content": json.dumps(outcome, ensure_ascii=False, default=str),
            },
            {
                "role": "assistant",
                "content": (
                    "\n".join(insights)
                    or str(outcome.get("key_insight") or lesson.get("summary") or "")
                ),
            },
        ]
        try:
            return await self._backend.store(
                task_id,
                messages,
                metadata={
                    "app_id": self._app_id,
                    "project_id": self._project_id or target,
                    "target": target,
                    "cycle": cycle,
                    "memory_type": "design_case",
                    "memory_schema": outcome.get("memory_schema"),
                    "case_triggers": identity.get("triggers", outcome.get("triggers", [])),
                    "case_verdict": lesson.get("verdict"),
                    "case_quality_score": quality.get("score"),
                    "is_final": finalize,
                },
            )
        except Exception as exc:
            logger.warning("protein-design memory store failed: %s", exc)
            return False
