from pathlib import Path

from CognitiveRAG.crag.skill_memory.retrieval import retrieve_skill_artifacts
from CognitiveRAG.crag.skill_memory.schemas import SkillPackRequest, SkillSourceRef, build_artifact
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


def test_retrieval_prefers_matching_typed_artifacts(tmp_path: Path):
    store = SkillMemoryStore(tmp_path / "skills.sqlite3")
    craft_ref = SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id="c-1")
    store.upsert(
        build_artifact(
            artifact_type="principle",
            source_ref=craft_ref,
            canonical_text="Principle: storyboard beats need clear transitions.",
            confidence=0.9,
            metadata={"agent_type": "storyboard_agent", "task_type": "short_explainer", "language": "en"},
        )
    )
    store.upsert(
        build_artifact(
            artifact_type="raw_chunk",
            source_ref=craft_ref,
            canonical_text="Raw notes about random scripts.",
            confidence=0.95,
        )
    )
    request = SkillPackRequest(
        query="transitions for explainer storyboard",
        agent_type="storyboard_agent",
        task_type="short_explainer",
        language="en",
    )

    ranked = retrieve_skill_artifacts(store=store, request=request, include_raw=True, max_candidates=10)
    assert ranked
    assert ranked[0].artifact.artifact_type == "principle"

