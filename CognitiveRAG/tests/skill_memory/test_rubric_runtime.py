from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore, RubricRuntime, compute_weighted_score


def test_rubric_runtime_structure_and_weighted_score():
    criteria = [
        RubricCriterionScore(criterion_id="hook", label="Hook clarity", score=4, max_score=5, weight=2),
        RubricCriterionScore(criterion_id="pacing", label="Pacing", score=3, max_score=5, weight=1),
    ]
    runtime = RubricRuntime(rubric_id="rubric:script:v1", rubric_ref="script_v1", criteria=criteria)
    assert runtime.rubric_id == "rubric:script:v1"
    assert len(runtime.criteria) == 2

    overall = compute_weighted_score(criteria)
    assert 0.0 <= overall <= 1.0
    assert round(overall, 3) == round((4 * 2 + 3 * 1) / (5 * 2 + 5 * 1), 3)

