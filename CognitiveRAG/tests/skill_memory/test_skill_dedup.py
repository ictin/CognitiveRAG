from CognitiveRAG.crag.skill_memory.dedup import dedup_artifacts
from CognitiveRAG.crag.skill_memory.schemas import SkillSourceRef, build_artifact


def test_skill_artifact_dedup_is_deterministic():
    a_ref = SkillSourceRef(source_kind="craft", source_path="/a.md", chunk_id="a")
    b_ref = SkillSourceRef(source_kind="craft", source_path="/b.md", chunk_id="b")
    a = build_artifact(
        artifact_type="principle",
        source_ref=a_ref,
        canonical_text="Always lead with one clear promise.",
        confidence=0.7,
    )
    b = build_artifact(
        artifact_type="principle",
        source_ref=b_ref,
        canonical_text="Always   lead with one clear promise.",
        confidence=0.9,
    )
    out = dedup_artifacts([a, b])
    assert len(out) == 1
    assert out[0].confidence == 0.9
    assert len(out[0].source_refs) == 2
    assert int(out[0].metadata.get("dedup_count", 1)) >= 2

