from pathlib import Path

from CognitiveRAG.crag.skill_memory.execution_schema import ExecutionProvenance, build_execution_case
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore


def test_execution_store_persists_and_recovers_case(tmp_path: Path):
    store = SkillExecutionStore(tmp_path / "skill_exec.sqlite3")
    case = build_execution_case(
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        language="en",
        request_text="Create intro for quick omelette.",
        selected_artifact_ids=["skill:principle:a", "skill:template:b"],
        output_text="Hook: 3-egg omelette in 5 minutes.",
        success_flag=True,
        notes="Human accepted with minor edit.",
        provenance=ExecutionProvenance(session_id="sess-1", run_id="run-1"),
    )
    store.upsert_case(case)

    loaded = store.get_case(case.execution_case_id)
    assert loaded is not None
    assert loaded.execution_case_id == case.execution_case_id
    assert loaded.selected_artifact_ids == ["skill:principle:a", "skill:template:b"]
    assert loaded.success_flag is True
    assert loaded.provenance.run_id == "run-1"

