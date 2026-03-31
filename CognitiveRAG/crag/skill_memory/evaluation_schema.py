from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from pydantic import BaseModel, Field

from CognitiveRAG.crag.skill_memory.rubric_runtime import RubricCriterionScore, compute_weighted_score


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationProvenance(BaseModel):
    evaluator: str = "skill_evaluation"
    source: str = "skill_evaluation_memory"
    metadata: Dict[str, object] = Field(default_factory=dict)


class SkillEvaluationCase(BaseModel):
    evaluation_case_id: str
    execution_case_id: str
    agent_type: str
    task_type: str
    channel_type: str = ""
    language: str = ""
    rubric_id: str = ""
    rubric_ref: str = ""
    criterion_scores: List[RubricCriterionScore] = Field(default_factory=list)
    overall_score: float = 0.0
    pass_flag: bool = False
    anti_pattern_hits: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    human_edits_summary: str = ""
    improvement_notes: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    provenance: EvaluationProvenance = Field(default_factory=EvaluationProvenance)


def build_evaluation_case_id(*, execution_case_id: str, rubric_id: str, created_at: str) -> str:
    seed = f"{execution_case_id}|{rubric_id}|{created_at}|{uuid.uuid4()}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]
    return f"eval:{digest}"


def build_evaluation_case(
    *,
    execution_case_id: str,
    agent_type: str,
    task_type: str,
    criterion_scores: List[RubricCriterionScore],
    channel_type: str = "",
    language: str = "",
    rubric_id: str = "",
    rubric_ref: str = "",
    pass_flag: bool | None = None,
    anti_pattern_hits: List[str] | None = None,
    strengths: List[str] | None = None,
    weaknesses: List[str] | None = None,
    human_edits_summary: str = "",
    improvement_notes: List[str] | None = None,
    provenance: EvaluationProvenance | None = None,
) -> SkillEvaluationCase:
    created = utc_now_iso()
    overall = compute_weighted_score(criterion_scores)
    resolved_pass = bool(pass_flag) if pass_flag is not None else overall >= 0.7
    return SkillEvaluationCase(
        evaluation_case_id=build_evaluation_case_id(
            execution_case_id=execution_case_id,
            rubric_id=rubric_id or rubric_ref or "rubric",
            created_at=created,
        ),
        execution_case_id=execution_case_id,
        agent_type=agent_type,
        task_type=task_type,
        channel_type=channel_type,
        language=language,
        rubric_id=rubric_id,
        rubric_ref=rubric_ref,
        criterion_scores=criterion_scores,
        overall_score=overall,
        pass_flag=resolved_pass,
        anti_pattern_hits=sorted(set(anti_pattern_hits or [])),
        strengths=list(strengths or []),
        weaknesses=list(weaknesses or []),
        human_edits_summary=human_edits_summary,
        improvement_notes=list(improvement_notes or []),
        created_at=created,
        updated_at=created,
        provenance=provenance or EvaluationProvenance(),
    )

