from __future__ import annotations

from typing import Dict, List
from pathlib import Path

from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore
from CognitiveRAG.crag.skill_memory.ranking import RankedArtifact, rank_artifacts
from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact, SkillPackRequest, SkillSourceRef, build_artifact
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


TYPED_NAMESPACES = [
    "craft_principles",
    "craft_templates",
    "craft_examples",
    "craft_rubrics",
    "craft_antipatterns",
    "craft_workflows",
    "style_profiles",
    "craft_execution_lessons",
    "craft_evaluation_lessons",
]

RAW_NAMESPACES = ["craft_raw", "content_books_raw"]


def _load_namespace(store: SkillMemoryStore, namespace: str) -> List[SkillArtifact]:
    rows = store.list_namespace(namespace)
    return [SkillArtifact.model_validate(row) for row in rows]


def _derived_execution_lessons(store: SkillMemoryStore, request: SkillPackRequest) -> List[SkillArtifact]:
    db = Path(store.db_path).parent / "skill_exec.sqlite3"
    if not db.exists():
        return []
    rows = SkillExecutionStore(db).list_cases(
        agent_type=request.agent_type,
        task_type=request.task_type,
        channel_type=request.channel_type or None,
        limit=20,
    )
    out: List[SkillArtifact] = []
    for case in rows:
        status = "successful" if case.success_flag else "unsuccessful"
        lesson = (
            f"Execution lesson: For {case.task_type}, a {status} run used artifacts "
            f"{', '.join(case.selected_artifact_ids[:4]) or 'none'}; "
            f"request='{case.request_text.strip()[:160]}'; "
            f"output='{case.output_text.strip()[:160]}'."
        )
        ref = SkillSourceRef(
            source_kind="craft",
            source_path=f"skill_execution:{case.execution_case_id}",
            chunk_id=case.execution_case_id,
        )
        out.append(
            build_artifact(
                artifact_type="execution_lesson",
                source_ref=ref,
                canonical_text=lesson,
                title=f"Execution lesson {case.execution_case_id}",
                confidence=0.85 if case.success_flag else 0.65,
                tags=["execution_lesson", case.agent_type, case.task_type],
                metadata={
                    "execution_case_id": case.execution_case_id,
                    "success_flag": bool(case.success_flag),
                    "selected_artifact_ids": list(case.selected_artifact_ids),
                    "source_class": "skill_execution_memory",
                },
            )
        )
    return out


def _derived_evaluation_lessons(store: SkillMemoryStore, request: SkillPackRequest) -> List[SkillArtifact]:
    db = Path(store.db_path).parent / "skill_eval.sqlite3"
    if not db.exists():
        return []
    rows = SkillEvaluationStore(db).list_cases(
        agent_type=request.agent_type,
        task_type=request.task_type,
        channel_type=request.channel_type or None,
        language=request.language or None,
        limit=20,
    )
    out: List[SkillArtifact] = []
    for case in rows:
        weaknesses = ", ".join(case.weaknesses[:3]) or "none"
        improvements = ", ".join(case.improvement_notes[:3]) or "none"
        lesson = (
            f"Evaluation lesson: score={case.overall_score:.2f}, pass={case.pass_flag}; "
            f"weaknesses={weaknesses}; improvements={improvements}; "
            f"anti_patterns={', '.join(case.anti_pattern_hits[:3]) or 'none'}."
        )
        ref = SkillSourceRef(
            source_kind="craft",
            source_path=f"skill_evaluation:{case.evaluation_case_id}",
            chunk_id=case.evaluation_case_id,
        )
        out.append(
            build_artifact(
                artifact_type="evaluation_lesson",
                source_ref=ref,
                canonical_text=lesson,
                title=f"Evaluation lesson {case.evaluation_case_id}",
                confidence=0.88,
                tags=["evaluation_lesson", case.agent_type, case.task_type],
                metadata={
                    "evaluation_case_id": case.evaluation_case_id,
                    "execution_case_id": case.execution_case_id,
                    "pass_flag": bool(case.pass_flag),
                    "overall_score": float(case.overall_score),
                    "source_class": "skill_evaluation_memory",
                },
            )
        )
    return out


def retrieve_skill_artifacts(
    *,
    store: SkillMemoryStore,
    request: SkillPackRequest,
    include_raw: bool = True,
    max_candidates: int = 60,
) -> List[RankedArtifact]:
    seen: Dict[str, SkillArtifact] = {}

    for namespace in TYPED_NAMESPACES:
        for artifact in _load_namespace(store, namespace):
            seen[artifact.artifact_id] = artifact

    for artifact in _derived_execution_lessons(store, request):
        seen[artifact.artifact_id] = artifact
    for artifact in _derived_evaluation_lessons(store, request):
        seen[artifact.artifact_id] = artifact

    if include_raw:
        for namespace in RAW_NAMESPACES:
            for artifact in _load_namespace(store, namespace):
                seen[artifact.artifact_id] = artifact

    ranked = rank_artifacts(seen.values(), request)
    return ranked[: max(1, int(max_candidates))]
