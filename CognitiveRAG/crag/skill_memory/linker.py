from __future__ import annotations

from typing import Iterable, List

from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact


def link_artifacts(artifacts: Iterable[SkillArtifact]) -> List[SkillArtifact]:
    items = list(artifacts)
    raw_by_chunk: dict[str, str] = {}
    for art in items:
        if art.artifact_type != "raw_chunk":
            continue
        for ref in art.source_refs:
            if ref.chunk_id:
                raw_by_chunk[ref.chunk_id] = art.artifact_id

    for art in items:
        chunk_links: list[str] = []
        for ref in art.source_refs:
            if ref.chunk_id and ref.chunk_id in raw_by_chunk and art.artifact_id != raw_by_chunk[ref.chunk_id]:
                chunk_links.append(raw_by_chunk[ref.chunk_id])
        if chunk_links:
            art.links = sorted(set([*art.links, *chunk_links]))
    return items

