from CognitiveRAG.crag.skill_memory.evaluation_linker import (
    evaluation_link_row,
    has_valid_execution_link,
    normalize_execution_case_id,
)
from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore


def test_evaluation_linker_preserves_execution_case_link():
    case = build_evaluation_case(
        execution_case_id=" exec:run-1 ",
        agent_type="storyboard_agent",
        task_type="short_explainer",
        criterion_scores=[RubricCriterionScore(criterion_id="scene", label="Scene", score=4, max_score=5)],
    )
    assert has_valid_execution_link(case)
    assert normalize_execution_case_id(case.execution_case_id) == "exec:run-1"
    row = evaluation_link_row(case)
    assert row[0] == case.evaluation_case_id
    assert row[1] == "exec:run-1"

