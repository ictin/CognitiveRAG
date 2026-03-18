from __future__ import annotations

from CognitiveRAG.llm.prompts import CRITIC_SYSTEM, critic_user_prompt
from CognitiveRAG.llm.schemas import CriticOutput
from CognitiveRAG.schemas.agent import Critique


class CriticAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def run(self, query: str, answer: str) -> Critique:
        try:
            structured = await self.llm_client.ainvoke_structured(
                system_prompt=CRITIC_SYSTEM,
                user_prompt=critic_user_prompt(query, answer),
                schema=CriticOutput,
            )
            issues = getattr(structured, 'issues', []) or []
            approved = getattr(structured, 'approved', None)
            # map alternative fields if model returned a different structure
            if approved is None:
                # check for grounded/complete/actionable style
                grounded = getattr(structured, 'grounded', None)
                complete = getattr(structured, 'complete', None)
                actionable = getattr(structured, 'actionable', None)
                if grounded is not None:
                    approved = bool(grounded)
                    # build issues list from flags
                    if not bool(grounded):
                        issues.append('not_grounded')
                    if complete is not None and not bool(complete):
                        issues.append('incomplete')
                    if actionable is not None and not bool(actionable):
                        issues.append('not_actionable')
            if approved is None:
                approved = False
            follow_up = []
            if hasattr(structured, 'follow_up_actions'):
                follow_up = getattr(structured, 'follow_up_actions') or []
            return Critique(approved=approved, issues=issues, follow_up_actions=follow_up)
        except Exception as e:
            # parsing failed; attempt to capture raw content logged by ainvoke_structured
            try:
                from CognitiveRAG.core.logging import logger
                logger.error('CRITIC_STRUCTURED_PARSE_FAILED: %s', e)
            except Exception:
                pass
            return Critique(approved=False, issues=['parse_error'], follow_up_actions=['Parsing of critic output failed; see logs for raw content'])
