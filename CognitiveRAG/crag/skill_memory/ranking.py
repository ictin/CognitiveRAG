from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact, SkillPackRequest


_WORD_RE = re.compile(r"\W+")

TYPE_PRIORITY: Dict[str, int] = {
    "principle": 70,
    "template": 60,
    "example": 50,
    "rubric": 40,
    "anti_pattern": 30,
    "workflow": 20,
    "style_gist": 15,
    "raw_chunk": 1,
}


@dataclass(frozen=True)
class RankedArtifact:
    artifact: SkillArtifact
    score: float
    reasons: List[str]


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.split((text or "").lower()) if t}


def _artifact_text_blob(artifact: SkillArtifact) -> str:
    metadata_bits = [str(v) for v in artifact.metadata.values()]
    return " ".join(
        [
            artifact.title or "",
            artifact.canonical_text or "",
            " ".join(artifact.tags),
            " ".join(metadata_bits),
            artifact.namespace,
        ]
    ).lower()


def score_artifact(artifact: SkillArtifact, request: SkillPackRequest) -> RankedArtifact:
    reasons: List[str] = []
    score = float(TYPE_PRIORITY.get(artifact.artifact_type, 0))
    reasons.append(f"type:{artifact.artifact_type}")

    score += max(0.0, min(1.0, float(artifact.confidence))) * 10.0
    reasons.append("confidence")

    query_tokens = _tokens(request.query)
    blob = _artifact_text_blob(artifact)
    artifact_tokens = _tokens(blob)
    if query_tokens:
        overlap = len(query_tokens & artifact_tokens) / max(1, len(query_tokens))
        if overlap > 0:
            score += overlap * 16.0
            reasons.append("query_match")

    if request.agent_type and request.agent_type.replace("_", " ") in blob:
        score += 6.0
        reasons.append("agent_match")
    if request.task_type and request.task_type.replace("_", " ") in blob:
        score += 6.0
        reasons.append("task_match")
    if request.channel_type and request.channel_type.replace("_", " ") in blob:
        score += 3.0
        reasons.append("channel_match")
    if request.language and request.language.lower() in blob:
        score += 3.0
        reasons.append("language_match")
    if request.style_profile and request.style_profile.replace("_", " ") in blob:
        score += 4.0
        reasons.append("style_match")

    if artifact.artifact_type == "raw_chunk":
        score -= 8.0
        reasons.append("raw_penalty")

    return RankedArtifact(artifact=artifact, score=score, reasons=reasons)


def rank_artifacts(artifacts: Iterable[SkillArtifact], request: SkillPackRequest) -> List[RankedArtifact]:
    ranked = [score_artifact(artifact, request) for artifact in artifacts]
    return sorted(ranked, key=lambda r: (-r.score, r.artifact.artifact_id))

