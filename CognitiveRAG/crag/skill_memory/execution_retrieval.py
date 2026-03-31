from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from CognitiveRAG.crag.skill_memory.case_linker import case_artifact_overlap, text_overlap_score
from CognitiveRAG.crag.skill_memory.execution_schema import SkillExecutionCase
from CognitiveRAG.crag.skill_memory.execution_store import SkillExecutionStore


@dataclass(frozen=True)
class RankedExecutionCase:
    case: SkillExecutionCase
    score: float
    reasons: List[str]


def _score_case(
    *,
    case: SkillExecutionCase,
    query: str,
    agent_type: str | None,
    task_type: str | None,
    channel_type: str | None,
    language: str | None,
    artifact_ids: Iterable[str] | None,
) -> RankedExecutionCase:
    score = 0.0
    reasons: List[str] = []

    if agent_type and case.agent_type == agent_type:
        score += 6.0
        reasons.append("agent_match")
    if task_type and case.task_type == task_type:
        score += 6.0
        reasons.append("task_match")
    if channel_type and case.channel_type == channel_type:
        score += 4.0
        reasons.append("channel_match")
    if language and case.language == language:
        score += 3.0
        reasons.append("language_match")

    txt = text_overlap_score(query, case)
    if txt > 0:
        score += txt * 12.0
        reasons.append("text_match")

    if artifact_ids:
        overlap = case_artifact_overlap(case, artifact_ids)
        if overlap > 0:
            score += overlap * 10.0
            reasons.append("artifact_overlap")

    if case.success_flag:
        score += 1.5
        reasons.append("success_bias")

    return RankedExecutionCase(case=case, score=score, reasons=reasons)


def retrieve_similar_execution_cases(
    *,
    store: SkillExecutionStore,
    query: str,
    agent_type: str | None = None,
    task_type: str | None = None,
    channel_type: str | None = None,
    language: str | None = None,
    artifact_ids: Iterable[str] | None = None,
    top_k: int = 5,
    prefetch_limit: int = 120,
) -> List[RankedExecutionCase]:
    cases = store.list_cases(
        agent_type=agent_type,
        task_type=task_type,
        channel_type=channel_type,
        limit=max(top_k, prefetch_limit),
    )
    ranked = [
        _score_case(
            case=case,
            query=query,
            agent_type=agent_type,
            task_type=task_type,
            channel_type=channel_type,
            language=language,
            artifact_ids=artifact_ids,
        )
        for case in cases
    ]
    ranked = [r for r in ranked if r.score > 0]
    ranked.sort(key=lambda r: (-r.score, r.case.execution_case_id))
    return ranked[: max(1, int(top_k))]

