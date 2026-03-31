from pathlib import Path

from CognitiveRAG.crag.skill_memory.pack_builder import build_skill_pack
from CognitiveRAG.crag.skill_memory.schemas import SkillPackRequest, SkillSourceRef, build_artifact
from CognitiveRAG.crag.skill_memory.store import SkillMemoryStore


def test_raw_chunks_only_used_when_typed_coverage_is_sparse(tmp_path: Path):
    store = SkillMemoryStore(tmp_path / "skills.sqlite3")
    ref = SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id="raw-1")
    store.upsert(
        build_artifact(
            artifact_type="raw_chunk",
            source_ref=ref,
            canonical_text="Raw craft notes about script pacing and hook structures.",
            confidence=0.9,
        )
    )

    pack = build_skill_pack(
        store=store,
        request=SkillPackRequest(
            query="script pacing",
            agent_type="script_agent",
            task_type="recipe_short",
            max_items=6,
        ),
    )
    assert pack.grouped_artifacts.get("raw_chunk")
    assert any("thin" in warning.lower() for warning in pack.warnings)

