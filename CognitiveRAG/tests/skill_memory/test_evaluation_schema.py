from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore


def test_evaluation_schema_captures_required_quality_fields():
    case = build_evaluation_case(
        execution_case_id="exec:abc",
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        language="en",
        rubric_id="rubric:script:v1",
        criterion_scores=[
            RubricCriterionScore(criterion_id="hook", label="Hook", score=4, max_score=5),
            RubricCriterionScore(criterion_id="pacing", label="Pacing", score=3, max_score=5),
        ],
        anti_pattern_hits=["slow_intro"],
        strengths=["clear promise"],
        weaknesses=["weak cta"],
        improvement_notes=["tighten ending"],
    )
    assert case.evaluation_case_id.startswith("eval:")
    assert case.execution_case_id == "exec:abc"
    assert case.rubric_id == "rubric:script:v1"
    assert len(case.criterion_scores) == 2
    assert case.overall_score > 0
    assert case.anti_pattern_hits == ["slow_intro"]
    assert case.strengths == ["clear promise"]
    assert case.weaknesses == ["weak cta"]
    assert case.improvement_notes == ["tighten ending"]

