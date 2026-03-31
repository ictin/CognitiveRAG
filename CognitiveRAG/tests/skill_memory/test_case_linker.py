from CognitiveRAG.crag.skill_memory.case_linker import case_artifact_overlap, normalize_artifact_ids
from CognitiveRAG.crag.skill_memory.execution_schema import build_execution_case


def test_case_linker_normalizes_and_scores_artifact_overlap():
    case = build_execution_case(
        agent_type="storyboard_agent",
        task_type="short_explainer",
        request_text="Build a 5-scene explainer storyboard.",
        selected_artifact_ids=["a", "b", "b", " c "],
        success_flag=False,
    )
    assert normalize_artifact_ids(case.selected_artifact_ids) == ["a", "b", "c"]

    overlap = case_artifact_overlap(case, ["b", "z"])
    assert overlap == 0.5

