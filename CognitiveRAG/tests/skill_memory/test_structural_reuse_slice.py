from pathlib import Path

from CognitiveRAG.crag.skill_memory.evaluation_schema import build_evaluation_case
from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore
from CognitiveRAG.crag.skill_memory.execution_schema import build_execution_case
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore
from CognitiveRAG.crag.skill_memory.pack_builder import build_skill_pack
from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore
from CognitiveRAG.crag.skill_memory.schemas import SkillPackRequest, SkillSourceRef, build_artifact
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


def test_structural_skill_memory_reuse_includes_lessons_with_provenance(tmp_path: Path):
    store = SkillMemoryStore(tmp_path / "skills.sqlite3")
    exec_store = SkillExecutionStore(tmp_path / "skill_exec.sqlite3")
    eval_store = SkillEvaluationStore(tmp_path / "skill_eval.sqlite3")

    ref = SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id="c-1")
    store.upsert(
        build_artifact(
            artifact_type="template",
            source_ref=ref,
            canonical_text="Template: {hook} -> {proof} -> {cta}",
            title="recipe template",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        )
    )
    store.upsert(
        build_artifact(
            artifact_type="rubric",
            source_ref=ref,
            canonical_text="Rubric: hook clarity; proof quality; CTA strength",
            title="recipe rubric",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        )
    )
    store.upsert(
        build_artifact(
            artifact_type="style_note",
            source_ref=ref,
            canonical_text="Style: concise, direct, practical",
            title="style note",
            metadata={"agent_type": "script_agent", "task_type": "recipe_short"},
        )
    )

    execution_case = build_execution_case(
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        request_text="Write a sharp ramen hook with proof and CTA.",
        selected_artifact_ids=["skill:template:recipe", "skill:rubric:recipe"],
        output_text="Hook with proof and CTA",
        success_flag=True,
    )
    exec_store.upsert_case(execution_case)

    evaluation_case = build_evaluation_case(
        execution_case_id=execution_case.execution_case_id,
        agent_type="script_agent",
        task_type="recipe_short",
        channel_type="short_video",
        criterion_scores=[RubricCriterionScore(criterion_id="hook", label="Hook", score=2, max_score=5)],
        pass_flag=False,
        weaknesses=["hook too abstract"],
        improvement_notes=["state concrete benefit in first sentence"],
        anti_pattern_hits=["vague_intro"],
    )
    eval_store.upsert_case(evaluation_case)

    pack = build_skill_pack(
        store=store,
        request=SkillPackRequest(
            query="reuse prior execution lesson and evaluation lesson to improve ramen hook",
            agent_type="script_agent",
            task_type="recipe_short",
            channel_type="short_video",
            max_items=12,
        ),
    )

    assert "template" in pack.grouped_artifacts
    assert "execution_lesson" in pack.grouped_artifacts
    assert "evaluation_lesson" in pack.grouped_artifacts
    assert "style_note" in pack.grouped_artifacts

    execution_lesson = pack.grouped_artifacts["execution_lesson"][0]
    evaluation_lesson = pack.grouped_artifacts["evaluation_lesson"][0]

    assert execution_lesson.metadata.get("source_class") == "skill_execution_memory"
    assert execution_lesson.metadata.get("execution_case_id") == execution_case.execution_case_id
    assert execution_lesson.source_refs and execution_lesson.source_refs[0].source_path.startswith("skill_execution:")

    assert evaluation_lesson.metadata.get("source_class") == "skill_evaluation_memory"
    assert evaluation_lesson.metadata.get("evaluation_case_id") == evaluation_case.evaluation_case_id
    assert evaluation_lesson.metadata.get("execution_case_id") == execution_case.execution_case_id
    assert evaluation_lesson.source_refs and evaluation_lesson.source_refs[0].source_path.startswith("skill_evaluation:")

    explain_exec = pack.selection_explanations.get(execution_lesson.artifact_id, {})
    explain_eval = pack.selection_explanations.get(evaluation_lesson.artifact_id, {})
    assert explain_exec
    assert explain_eval
