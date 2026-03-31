from pathlib import Path

from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore


def test_evaluation_store_persists_case_with_execution_link(tmp_path: Path):
    store = SkillEvaluationStore(tmp_path / "skill_eval.sqlite3")
    case = build_evaluation_case(
        execution_case_id="exec:run1",
        agent_type="script_agent",
        task_type="recipe_short",
        criterion_scores=[RubricCriterionScore(criterion_id="hook", label="Hook", score=5, max_score=5)],
        pass_flag=True,
        strengths=["great opening"],
    )
    store.upsert_case(case)

    loaded = store.get_case(case.evaluation_case_id)
    assert loaded is not None
    assert loaded.execution_case_id == "exec:run1"
    assert loaded.pass_flag is True
    linked = store.list_by_execution_case("exec:run1")
    assert len(linked) == 1
    assert linked[0].evaluation_case_id == case.evaluation_case_id

