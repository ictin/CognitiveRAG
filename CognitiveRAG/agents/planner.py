from __future__ import annotations

from CognitiveRAG.llm.prompts import PLANNER_SYSTEM, planner_user_prompt
from CognitiveRAG.llm.schemas import PlannerOutput
from CognitiveRAG.schemas.agent import Plan, PlanStep


class PlannerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def run(self, query: str) -> Plan:
        # try structured planner output
        structured = None
        try:
            structured = await self.llm_client.ainvoke_structured(
                system_prompt=PLANNER_SYSTEM,
                user_prompt=planner_user_prompt(query),
                schema=PlannerOutput,
            )
        except Exception:
            structured = None

        if structured is None:
            # fallback minimal Plan
            return Plan(objective=query, steps=[PlanStep(step_id="step_1", description="Read the retrieved context")])

        raw_objective = getattr(structured, 'objective', None) or query
        # make objective short and direct: use first line or a concise fallback
        if isinstance(raw_objective, str):
            objective = raw_objective.strip().splitlines()[0]
            # if the model returned a planning instruction, convert to concise objective
            if objective.lower().startswith('plan') or 'plan how' in objective.lower():
                objective = "Determine the document's main topic"
            if len(objective) > 120:
                objective = objective[:117].rsplit(' ', 1)[0] + '...'
        else:
            objective = query
        raw_steps = getattr(structured, 'steps', []) or []
        steps = [PlanStep(step_id=f"step_{i+1}", description=step) for i, step in enumerate(raw_steps)]
        if not steps:
            steps = [PlanStep(step_id="step_1", description="Read the retrieved context"), PlanStep(step_id="step_2", description="Identify the main topic"), PlanStep(step_id="step_3", description="Answer briefly")]
        return Plan(objective=objective, steps=steps)
