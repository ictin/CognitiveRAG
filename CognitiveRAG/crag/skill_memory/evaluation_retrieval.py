from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from CognitiveRAG.crag.skill_memory.evaluation_schema import SkillEvaluationCase
from CognitiveRAG.crag.skill_memory.evaluation_store import SkillEvaluationStore


_WORD_RE = re.compile(r"\W+")


@dataclass(frozen=True)
class RankedEvaluationCase:
    case: SkillEvaluationCase
    score: float
    reasons: List[str]


def _text_overlap(query: str, case: SkillEvaluationCase) -> float:
    q = {t for t in _WORD_RE.split((query or "").lower()) if t}
    if not q:
        return 0.0
    blob = " ".join(
        [
            " ".join(case.strengths),
            " ".join(case.weaknesses),
            " ".join(case.improvement_notes),
            case.human_edits_summary,
            " ".join(case.anti_pattern_hits),
        ]
    ).lower()
    c = {t for t in _WORD_RE.split(blob) if t}
    if not c:
        return 0.0
    return len(q & c) / max(1, len(q))


def _score(case: SkillEvaluationCase, *, query: str = "", prefer_weak: bool = False) -> RankedEvaluationCase:
    reasons: List[str] = []
    if prefer_weak:
        score = (1.0 - float(case.overall_score)) * 10.0
        if not case.pass_flag:
            score += 5.0
            reasons.append("failed_case")
        if case.anti_pattern_hits:
            score += min(3.0, len(case.anti_pattern_hits))
            reasons.append("anti_pattern_hits")
    else:
        score = float(case.overall_score) * 10.0
        if case.pass_flag:
            score += 3.0
            reasons.append("passed_case")
    overlap = _text_overlap(query, case)
    if overlap > 0:
        score += overlap * 6.0
        reasons.append("text_match")
    return RankedEvaluationCase(case=case, score=score, reasons=reasons)


def retrieve_best_evaluations(
    *,
    store: SkillEvaluationStore,
    agent_type: str | None = None,
    task_type: str | None = None,
    channel_type: str | None = None,
    language: str | None = None,
    top_k: int = 5,
) -> List[RankedEvaluationCase]:
    cases = store.list_cases(
        agent_type=agent_type,
        task_type=task_type,
        channel_type=channel_type,
        language=language,
        pass_flag=True,
        limit=max(top_k * 3, 20),
    )
    ranked = [_score(c, prefer_weak=False) for c in cases]
    ranked.sort(key=lambda r: (-r.score, r.case.evaluation_case_id))
    return ranked[: max(1, int(top_k))]


def retrieve_weak_evaluations(
    *,
    store: SkillEvaluationStore,
    agent_type: str | None = None,
    task_type: str | None = None,
    channel_type: str | None = None,
    language: str | None = None,
    top_k: int = 5,
) -> List[RankedEvaluationCase]:
    cases = store.list_cases(
        agent_type=agent_type,
        task_type=task_type,
        channel_type=channel_type,
        language=language,
        pass_flag=False,
        limit=max(top_k * 3, 20),
    )
    ranked = [_score(c, prefer_weak=True) for c in cases]
    ranked.sort(key=lambda r: (-r.score, r.case.evaluation_case_id))
    return ranked[: max(1, int(top_k))]


def retrieve_evaluations_for_execution(
    *, store: SkillEvaluationStore, execution_case_id: str, query: str = "", top_k: int = 5
) -> List[RankedEvaluationCase]:
    cases = store.list_by_execution_case(execution_case_id, limit=max(top_k * 3, 20))
    ranked = [_score(c, query=query, prefer_weak=not c.pass_flag) for c in cases]
    ranked.sort(key=lambda r: (-r.score, r.case.evaluation_case_id))
    return ranked[: max(1, int(top_k))]

