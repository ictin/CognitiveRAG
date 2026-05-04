from CognitiveRAG.crag.skill_memory.schemas import SkillSourceRef, build_artifact


def test_schema_builds_all_required_artifact_types():
    ref = SkillSourceRef(source_kind="craft", source_path="/books/craft.md", book_id="b1", chapter_id="c1", section_id="s1", chunk_id="ch1")
    types = [
        "raw_chunk",
        "principle",
        "template",
        "example",
        "rubric",
        "anti_pattern",
        "workflow",
        "style_note",
        "style_gist",
        "execution_lesson",
        "evaluation_lesson",
    ]
    built = [build_artifact(artifact_type=t, source_ref=ref, canonical_text=f"{t} text", title=t) for t in types]
    assert {x.artifact_type for x in built} == set(types)
    assert all(x.artifact_id for x in built)
    assert all(x.namespace.startswith("craft_") or x.namespace == "style_profiles" for x in built)
