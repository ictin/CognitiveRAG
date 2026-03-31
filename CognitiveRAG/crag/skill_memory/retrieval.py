from __future__ import annotations

from typing import Dict, List

from CognitiveRAG.crag.skill_memory.ranking import RankedArtifact, rank_artifacts
from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact, SkillPackRequest
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


TYPED_NAMESPACES = [
    "craft_principles",
    "craft_templates",
    "craft_examples",
    "craft_rubrics",
    "craft_antipatterns",
    "craft_workflows",
    "style_profiles",
]

RAW_NAMESPACES = ["craft_raw", "content_books_raw"]


def _load_namespace(store: SkillMemoryStore, namespace: str) -> List[SkillArtifact]:
    rows = store.list_namespace(namespace)
    return [SkillArtifact.model_validate(row) for row in rows]


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

    if include_raw:
        for namespace in RAW_NAMESPACES:
            for artifact in _load_namespace(store, namespace):
                seen[artifact.artifact_id] = artifact

    ranked = rank_artifacts(seen.values(), request)
    return ranked[: max(1, int(max_candidates))]

