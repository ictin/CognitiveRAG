from pathlib import Path

from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore
from CognitiveRAG.crag.skill_memory.schemas import SkillSourceRef, build_artifact


def test_store_persists_and_searches_skill_artifacts(tmp_path: Path):
    db = tmp_path / "skill_memory.sqlite3"
    store = SkillMemoryStore(db)
    ref = SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id="chunk-1")
    artifact = build_artifact(
        artifact_type="workflow",
        source_ref=ref,
        canonical_text="Workflow: ideate -> draft -> revise -> QA",
        title="workflow",
        confidence=0.88,
    )
    store.upsert(artifact)

    rows = store.list_namespace("craft_workflows")
    assert len(rows) == 1
    assert rows[0]["artifact_type"] == "workflow"

    hits = store.search("revise QA", namespace="craft_workflows", top_k=3)
    assert len(hits) == 1
    assert hits[0]["artifact_id"] == artifact.artifact_id

