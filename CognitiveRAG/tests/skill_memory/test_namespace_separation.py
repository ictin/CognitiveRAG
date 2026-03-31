from CognitiveRAG.crag.skill_memory.distill import distill_chunk
from CognitiveRAG.crag.skill_memory.schemas import SkillSourceRef


def test_craft_and_content_namespaces_are_separate():
    craft_ref = SkillSourceRef(source_kind="craft", source_path="/craft/source.md", chunk_id="craft-1")
    content_ref = SkillSourceRef(source_kind="content", source_path="/content/source.md", chunk_id="content-1")

    craft = distill_chunk(
        text="Principle: Keep hooks specific. Workflow: research -> draft -> revise.",
        source_ref=craft_ref,
        include_raw=True,
    )
    content = distill_chunk(
        text="A content chapter excerpt with story details.",
        source_ref=content_ref,
        include_raw=True,
    )

    craft_namespaces = {a.namespace for a in craft}
    content_namespaces = {a.namespace for a in content}

    assert "content_books_raw" not in craft_namespaces
    assert content_namespaces == {"content_books_raw"}
    assert any(ns.startswith("craft_") or ns == "style_profiles" for ns in craft_namespaces)

