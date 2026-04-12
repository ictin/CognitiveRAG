from CognitiveRAG.crag.promotion.bridge import promote_summaries_to_patterns


def test_bridge_emits_normalized_patterns():
    session_id = "bridge_sess"
    summaries = [
        {
            "chunk_index": 0,
            "created_at": "2026-04-12T10:00:00Z",
            "summary": "The user said that the user prefers concise answers. "
            "We concluded the successful workflow was ingest -> validate -> deploy.",
        },
        {"chunk_index": 1, "summary": "We concluded the successful workflow was ingest -> validate -> deploy."},
    ]
    patterns = promote_summaries_to_patterns(session_id, summaries)
    assert len(patterns) == 2
    subtypes = {p.memory_subtype for p in patterns}
    assert "profile_preference" in subtypes or "stable_fact" in subtypes
    assert "workflow_pattern" in subtypes or "procedure_pattern" in subtypes
    for p in patterns:
        assert p.pattern_id
        assert p.item_id
        assert p.normalized_text
        assert p.provenance
        prov = p.provenance[0]
        assert "source_refs" in prov
