from pathlib import Path

from CognitiveRAG.crag.skill_memory.evaluation_retrieval import retrieve_best_evaluations
from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore


def test_evaluation_retrieval_respects_agent_task_channel_language_filters(tmp_path: Path):
    store = SkillEvaluationStore(tmp_path / "skill_eval.sqlite3")
    store.upsert_case(
        build_evaluation_case(
            execution_case_id="exec:1",
            agent_type="script_agent",
            task_type="recipe_short",
            channel_type="short_video",
            language="en",
            criterion_scores=[RubricCriterionScore(criterion_id="hook", label="Hook", score=5, max_score=5)],
            pass_flag=True,
        )
    )
    store.upsert_case(
        build_evaluation_case(
            execution_case_id="exec:2",
            agent_type="storyboard_agent",
            task_type="short_explainer",
            channel_type="youtube",
            language="en",
            criterion_scores=[RubricCriterionScore(criterion_id="scene", label="Scene", score=5, max_score=5)],
            pass_flag=True,
        )
    )

    hits = retrieve_best_evaluations(
        store=store,
        agent_type="storyboard_agent",
        task_type="short_explainer",
        channel_type="youtube",
        language="en",
        top_k=5,
    )
    assert len(hits) == 1
    assert hits[0].case.execution_case_id == "exec:2"

