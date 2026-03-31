from CognitiveRAG.crag.skill_memory.execution_schema import ExecutionProvenance, build_execution_case


def test_execution_case_schema_has_required_fields():
    case = build_execution_case(
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        language="en",
        request_text="Write a 30-second recipe hook.",
        selected_artifact_ids=["skill:principle:1", "skill:template:2"],
        pack_summary="2 principles, 1 template",
        output_text="Hook: Make pasta in 10 minutes.",
        success_flag=True,
        provenance=ExecutionProvenance(session_id="s1", run_id="r1"),
    )
    assert case.execution_case_id.startswith("exec:")
    assert case.agent_type == "script_agent"
    assert case.task_type == "recipe_short"
    assert case.channel_type == "short_video"
    assert case.language == "en"
    assert case.request_text
    assert case.selected_artifact_ids == ["skill:principle:1", "skill:template:2"]
    assert case.created_at
    assert case.updated_at
    assert case.provenance.session_id == "s1"

