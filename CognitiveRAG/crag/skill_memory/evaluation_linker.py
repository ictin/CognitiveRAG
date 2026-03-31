from __future__ import annotations

from CognitiveRAG.crag.skill_memory.evaluation_schema import SkillEvaluationCase


def normalize_execution_case_id(execution_case_id: str) -> str:
    return str(execution_case_id or "").strip()


def has_valid_execution_link(evaluation_case: SkillEvaluationCase) -> bool:
    return bool(normalize_execution_case_id(evaluation_case.execution_case_id))


def evaluation_link_row(evaluation_case: SkillEvaluationCase) -> tuple[str, str]:
    return (evaluation_case.evaluation_case_id, normalize_execution_case_id(evaluation_case.execution_case_id))

