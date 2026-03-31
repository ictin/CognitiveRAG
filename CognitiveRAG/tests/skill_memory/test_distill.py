from CognitiveRAG.crag.skill_memory.distill import distill_chunk
from CognitiveRAG.crag.skill_memory.schemas import SkillSourceRef


def test_distill_extracts_required_types_from_craft_chunk():
    text = (
        "Principle: Always lead with a single clear promise. "
        "Template: Hook {audience} with {pain} then reveal {outcome}. "
        "Example before: weak opener. Example after: concrete high-retention opener. "
        "Rubric criteria: clarity; novelty; payoff. score 1-5. "
        "Anti-pattern: avoid vague intros that hide the core promise. "
        "Workflow: research -> draft -> tighten -> QA. "
        "Style: tone concise, voice direct, pacing fast."
    )
    ref = SkillSourceRef(source_kind="craft", source_path="/craft/book.md", book_id="craft-1", chapter_id="ch2", section_id="s3", chunk_id="chunk-7")
    artifacts = distill_chunk(text=text, source_ref=ref, include_raw=True)
    kinds = {a.artifact_type for a in artifacts}
    assert "raw_chunk" in kinds
    assert "principle" in kinds
    assert "template" in kinds
    assert "example" in kinds
    assert "rubric" in kinds
    assert "anti_pattern" in kinds
    assert "workflow" in kinds
    assert "style_gist" in kinds
    assert all(a.source_refs for a in artifacts)
    assert all(a.source_refs[0].chunk_id == "chunk-7" for a in artifacts)


def test_content_chunk_stays_raw_only():
    ref = SkillSourceRef(source_kind="content", source_path="/content/book.md", book_id="content-1", chunk_id="content-chunk")
    artifacts = distill_chunk(text="This is content material, not craft guidance.", source_ref=ref, include_raw=True)
    assert len(artifacts) == 1
    assert artifacts[0].artifact_type == "raw_chunk"
    assert artifacts[0].namespace == "content_books_raw"

