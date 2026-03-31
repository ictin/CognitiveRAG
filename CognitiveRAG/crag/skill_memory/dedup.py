from __future__ import annotations

from typing import Iterable, List

from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact


def dedup_artifacts(artifacts: Iterable[SkillArtifact]) -> List[SkillArtifact]:
    by_key: dict[tuple[str, str, str], SkillArtifact] = {}
    for artifact in artifacts:
        key = (artifact.namespace, artifact.artifact_type, artifact.normalized_key)
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = artifact
            continue
        winner = artifact if artifact.confidence >= prev.confidence else prev
        loser = prev if winner is artifact else artifact
        winner.source_refs = list(
            {
                (
                    ref.source_kind,
                    ref.source_path,
                    ref.book_id,
                    ref.chapter_id,
                    ref.section_id,
                    ref.chunk_id,
                    ref.char_start,
                    ref.char_end,
                ): ref
                for ref in [*winner.source_refs, *loser.source_refs]
            }.values()
        )
        winner.links = sorted(set([*winner.links, *loser.links]))
        winner.tags = sorted(set([*winner.tags, *loser.tags]))
        winner.metadata = {
            **dict(winner.metadata or {}),
            "dedup_count": int(dict(prev.metadata or {}).get("dedup_count", 1)) + 1,
        }
        by_key[key] = winner
    return sorted(by_key.values(), key=lambda a: (a.namespace, a.artifact_type, a.normalized_key))

