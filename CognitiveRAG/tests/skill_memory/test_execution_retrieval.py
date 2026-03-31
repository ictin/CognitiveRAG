from pathlib import Path

from CognitiveRAG.crag.skill_memory.execution_retrieval import retrieve_similar_execution_cases
from CognitiveRAG.crag.skill_memory.execution_schema import build_execution_case
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore


def test_similar_case_retrieval_prefers_text_and_artifact_overlap(tmp_path: Path):
    store = SkillExecutionStore(tmp_path / "skill_exec.sqlite3")
    case_a = build_execution_case(
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        language="en",
        request_text="Write a hook for a quick ramen recipe.",
        selected_artifact_ids=["skill:principle:hook", "skill:template:recipe"],
        output_text="Hook text for ramen.",
        success_flag=True,
    )
    case_b = build_execution_case(
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        language="en",
        request_text="Draft storyboard transitions for finance explainer.",
        selected_artifact_ids=["skill:principle:scene"],
        output_text="Storyboard output.",
        success_flag=True,
    )
    store.upsert_many([case_a, case_b])

    ranked = retrieve_similar_execution_cases(
        store=store,
        query="quick ramen hook",
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        language="en",
        artifact_ids=["skill:template:recipe"],
        top_k=2,
    )
    assert ranked
    assert ranked[0].case.execution_case_id == case_a.execution_case_id

