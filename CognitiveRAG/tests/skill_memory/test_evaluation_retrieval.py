from pathlib import Path

from CognitiveRAG.crag.skill_memory.evaluation_retrieval import retrieve_best_evaluations, retrieve_weak_evaluations
from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore


def test_retrieval_returns_best_and_weak_prior_evaluations(tmp_path: Path):
    store = SkillEvaluationStore(tmp_path / "skill_eval.sqlite3")
    good = build_evaluation_case(
        execution_case_id="exec:good",
        agent_type="script_agent",
        task_type="recipe_short",
        criterion_scores=[RubricCriterionScore(criterion_id="hook", label="Hook", score=5, max_score=5)],
        pass_flag=True,
        strengths=["hook is clear"],
    )
    weak = build_evaluation_case(
        execution_case_id="exec:weak",
        agent_type="script_agent",
        task_type="recipe_short",
        criterion_scores=[RubricCriterionScore(criterion_id="hook", label="Hook", score=1, max_score=5)],
        pass_flag=False,
        anti_pattern_hits=["slow_intro"],
        weaknesses=["no payoff"],
    )
    store.upsert_many([good, weak])

    best = retrieve_best_evaluations(store=store, agent_type="script_agent", task_type="recipe_short", top_k=2)
    assert best
    assert best[0].case.evaluation_case_id == good.evaluation_case_id

    weak_hits = retrieve_weak_evaluations(store=store, agent_type="script_agent", task_type="recipe_short", top_k=2)
    assert weak_hits
    assert weak_hits[0].case.evaluation_case_id == weak.evaluation_case_id

