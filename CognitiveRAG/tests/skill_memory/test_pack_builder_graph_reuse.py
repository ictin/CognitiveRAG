from pathlib import Path

from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.execution_schema import build_execution_case
from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore
from CognitiveRAG.crag.skill_memory.pack_builder import build_skill_pack
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore
from CognitiveRAG.crag.skill_memory.schemas import SkillPackRequest, SkillSourceRef, build_artifact
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


def _seed_base_artifacts(store: SkillMemoryStore) -> tuple[str, str]:
    ref = SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id="seed")
    strong = build_artifact(
        artifact_type="principle",
        source_ref=ref,
        canonical_text="Principle: Open with concrete promise and pacing.",
        confidence=0.75,
        metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
    )
    weak = build_artifact(
        artifact_type="principle",
        source_ref=ref,
        canonical_text="Principle: Start vaguely and explain later.",
        confidence=0.76,
        metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
    )
    store.upsert(strong)
    store.upsert(weak)
    return strong.artifact_id, weak.artifact_id


def test_pack_builder_uses_graph_evaluation_signal_to_boost_and_penalize(tmp_path: Path):
    store = SkillMemoryStore(tmp_path / "skills.sqlite3")
    exec_store = SkillExecutionStore(tmp_path / "skill_exec.sqlite3")
    eval_store = SkillEvaluationStore(tmp_path / "skill_eval.sqlite3")
    strong_id, weak_id = _seed_base_artifacts(store)

    for idx in range(2):
        good_exec = build_execution_case(
            agent_type="script_agent",
            task_type="recipe_short",
            request_text=f"good run {idx}",
            selected_artifact_ids=[strong_id],
            success_flag=True,
        )
        exec_store.upsert_case(good_exec)
        good_eval = build_evaluation_case(
            execution_case_id=good_exec.execution_case_id,
            agent_type="script_agent",
            task_type="recipe_short",
            criterion_scores=[RubricCriterionScore(criterion_id="hook", label="Hook", score=5, max_score=5)],
            pass_flag=True,
        )
        eval_store.upsert_case(good_eval)

    for idx in range(2):
        bad_exec = build_execution_case(
            agent_type="script_agent",
            task_type="recipe_short",
            request_text=f"bad run {idx}",
            selected_artifact_ids=[weak_id],
            success_flag=False,
        )
        exec_store.upsert_case(bad_exec)
        bad_eval = build_evaluation_case(
            execution_case_id=bad_exec.execution_case_id,
            agent_type="script_agent",
            task_type="recipe_short",
            criterion_scores=[RubricCriterionScore(criterion_id="hook", label="Hook", score=1, max_score=5)],
            pass_flag=False,
        )
        eval_store.upsert_case(bad_eval)

    pack = build_skill_pack(
        store=store,
        request=SkillPackRequest(
            query="hook and pacing",
            agent_type="script_agent",
            task_type="recipe_short",
            channel_type="short_video",
            language="en",
            max_items=6,
        ),
    )
    assert pack.selected_artifact_ids
    assert pack.selected_artifact_ids[0] == strong_id
    strong_explain = pack.selection_explanations.get(strong_id, {})
    weak_explain = pack.selection_explanations.get(weak_id, {})
    assert float(strong_explain.get("adjustment", 0.0)) > 0.0
    assert float(weak_explain.get("adjustment", 0.0)) < 0.0
    assert str(strong_explain.get("signal_mode")) == "signal_applied"
    assert str(weak_explain.get("signal_mode")) == "signal_applied"


def test_pack_builder_falls_back_with_sparse_history_and_is_deterministic(tmp_path: Path):
    store = SkillMemoryStore(tmp_path / "skills.sqlite3")
    first_id, _second_id = _seed_base_artifacts(store)

    request = SkillPackRequest(
        query="hook and pacing",
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        language="en",
        max_items=6,
    )

    first = build_skill_pack(store=store, request=request)
    second = build_skill_pack(store=store, request=request)
    assert first.selected_artifact_ids == second.selected_artifact_ids
    assert first.selection_explanations == second.selection_explanations
    explain = first.selection_explanations.get(first_id, {})
    assert str(explain.get("signal_mode")) == "fallback_sparse_history"
    assert float(explain.get("adjustment", 0.0)) == 0.0
