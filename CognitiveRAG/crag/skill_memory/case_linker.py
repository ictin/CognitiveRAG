from __future__ import annotations

import re
from typing import Iterable, List, Set

from CognitiveRAG.crag.skill_memory.execution_schema import SkillExecutionCase


_WORD_RE = re.compile(r"\W+")


def normalize_artifact_ids(artifact_ids: Iterable[str]) -> List[str]:
    return sorted({str(a).strip() for a in artifact_ids if str(a).strip()})


def case_artifact_overlap(case: SkillExecutionCase, artifact_ids: Iterable[str]) -> float:
    left: Set[str] = set(normalize_artifact_ids(case.selected_artifact_ids))
    right: Set[str] = set(normalize_artifact_ids(artifact_ids))
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(right))


def case_text_blob(case: SkillExecutionCase) -> str:
    return " ".join(
        [
            case.request_text or "",
            case.output_text or "",
            case.notes or "",
            case.pack_summary or "",
            " ".join(case.human_edits),
            case.agent_type,
            case.task_type,
            case.channel_type,
            case.language,
        ]
    ).lower()


def text_overlap_score(query: str, case: SkillExecutionCase) -> float:
    q_tokens = {t for t in _WORD_RE.split((query or "").lower()) if t}
    if not q_tokens:
        return 0.0
    c_tokens = {t for t in _WORD_RE.split(case_text_blob(case)) if t}
    if not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / max(1, len(q_tokens))

