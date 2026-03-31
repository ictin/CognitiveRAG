from __future__ import annotations

from typing import Dict, List

from CognitiveRAG.crag.skill_memory.retrieval import retrieve_skill_artifacts
from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact, SkillPack, SkillPackRequest
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


TYPE_ORDER = ["principle", "template", "example", "rubric", "anti_pattern", "workflow", "raw_chunk"]

AGENT_QUOTAS: Dict[str, Dict[str, int]] = {
    "script_agent": {
        "principle": 5,
        "template": 2,
        "example": 3,
        "rubric": 1,
        "anti_pattern": 1,
        "workflow": 1,
        "raw_chunk": 2,
    },
    "storyboard_agent": {
        "principle": 5,
        "template": 2,
        "example": 3,
        "rubric": 1,
        "anti_pattern": 1,
        "workflow": 1,
        "raw_chunk": 2,
    },
}


def _quota_for(request: SkillPackRequest) -> Dict[str, int]:
    return dict(AGENT_QUOTAS.get(request.agent_type, AGENT_QUOTAS["script_agent"]))


def _append(grouped: Dict[str, List[SkillArtifact]], artifact: SkillArtifact) -> None:
    grouped.setdefault(artifact.artifact_type, []).append(artifact)


def build_skill_pack(*, store: SkillMemoryStore, request: SkillPackRequest) -> SkillPack:
    ranked = retrieve_skill_artifacts(store=store, request=request, include_raw=True)
    quotas = _quota_for(request)
    grouped: Dict[str, List[SkillArtifact]] = {}
    selected_ids: List[str] = []
    used = set()
    max_items = max(1, int(request.max_items))
    typed_selected = 0

    for art_type in TYPE_ORDER:
        quota = quotas.get(art_type, 0)
        if quota <= 0:
            continue
        for candidate in ranked:
            art = candidate.artifact
            if art.artifact_type != art_type:
                continue
            if art.artifact_id in used:
                continue
            if len(grouped.get(art_type, [])) >= quota:
                continue
            if len(selected_ids) >= max_items:
                break
            if art_type == "raw_chunk" and typed_selected >= 6:
                continue
            _append(grouped, art)
            used.add(art.artifact_id)
            selected_ids.append(art.artifact_id)
            if art_type != "raw_chunk":
                typed_selected += 1
        if len(selected_ids) >= max_items:
            break

    warnings: List[str] = []
    if typed_selected < 4:
        warnings.append("Skill pack coverage is thin; insufficient distilled artifacts for this request.")
    for required in ("principle", "template", "example"):
        if not grouped.get(required):
            warnings.append(f"Missing preferred artifact type: {required}")

    return SkillPack(
        query=request.query,
        agent_type=request.agent_type,
        task_type=request.task_type,
        channel_type=request.channel_type,
        language=request.language,
        style_profile=request.style_profile,
        selected_artifact_ids=selected_ids,
        grouped_artifacts=grouped,
        warnings=warnings,
    )

