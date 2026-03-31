from pathlib import Path

from CognitiveRAG.crag.skill_memory.execution_retrieval import retrieve_similar_execution_cases
from CognitiveRAG.crag.skill_memory.execution_schema import build_execution_case
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore


def test_execution_retrieval_respects_agent_task_channel_filters(tmp_path: Path):
    store = SkillExecutionStore(tmp_path / "skill_exec.sqlite3")
    store.upsert_case(
        build_execution_case(
            agent_type="script_agent",
            task_type="recipe_short",
            channel_type="short_video",
            language="en",
            request_text="script one",
            selected_artifact_ids=["a1"],
            success_flag=True,
        )
    )
    store.upsert_case(
        build_execution_case(
            agent_type="storyboard_agent",
            task_type="short_explainer",
            channel_type="youtube",
            language="en",
            request_text="storyboard one",
            selected_artifact_ids=["b1"],
            success_flag=True,
        )
    )

    ranked = retrieve_similar_execution_cases(
        store=store,
        query="storyboard one",
        agent_type="storyboard_agent",
        task_type="short_explainer",
        channel_type="youtube",
        language="en",
        top_k=5,
    )
    assert len(ranked) == 1
    assert ranked[0].case.agent_type == "storyboard_agent"
    assert ranked[0].case.task_type == "short_explainer"
    assert ranked[0].case.channel_type == "youtube"

