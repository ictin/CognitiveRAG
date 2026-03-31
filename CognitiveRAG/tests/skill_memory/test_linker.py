from CognitiveRAG.crag.skill_memory.linker import link_artifacts
from CognitiveRAG.crag.skill_memory.schemas import SkillSourceRef, build_artifact


def test_linker_connects_typed_artifacts_to_raw_chunk():
    ref = SkillSourceRef(source_kind="craft", source_path="/craft.md", chunk_id="chunk-1")
    raw = build_artifact(artifact_type="raw_chunk", source_ref=ref, canonical_text="raw")
    principle = build_artifact(artifact_type="principle", source_ref=ref, canonical_text="Principle: be concrete.")
    out = link_artifacts([raw, principle])
    by_type = {a.artifact_type: a for a in out}
    assert raw.artifact_id in by_type["principle"].links

