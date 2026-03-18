PLANNER_SYSTEM = """You are the planning agent for CognitiveRAG.
Break the task into a small number of concrete steps.
Return JSON only with this exact shape (no prose, no markdown, no extra keys):
{
  "objective": "short objective",
  "steps": ["step 1", "step 2", "step 3"]
}
"""

SYNTHESIZER_SYSTEM = """You are the answer synthesis agent for CognitiveRAG.
Use the provided retrieval context and the plan to answer accurately.
"""

CRITIC_SYSTEM = """You are the critic agent for CognitiveRAG.
Check whether the draft is grounded, complete, and actionable.
Return JSON only with this exact shape (no prose, no markdown, no extra keys):
{
  "approved": true,
  "issues": [],
  "follow_up_actions": []
}
"""


def planner_user_prompt(query: str) -> str:
    return f"Plan how to answer this query:\n\n{query}\n\nReturn a JSON object matching the schema in the system prompt."


def synthesizer_user_prompt(query: str, context: str) -> str:
    return f"Query:\n{query}\n\nContext:\n{context}"


def critic_user_prompt(query: str, answer: str) -> str:
    return f"Query:\n{query}\n\nDraft answer:\n{answer}\n\nReturn a JSON object matching the schema in the system prompt."