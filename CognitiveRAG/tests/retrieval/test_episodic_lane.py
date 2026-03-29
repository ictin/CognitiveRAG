from CognitiveRAG.crag.retrieval.episodic_lane import retrieve


def test_episodic_lane_prefers_fresh_and_marks_must_include():
    hits = retrieve(
        session_id="s3",
        fresh_tail=[{"index": 4, "text": "newest message", "message_id": "m4"}],
        older_raw=[{"index": 0, "text": "older message", "message_id": "m0"}],
        top_k=10,
    )
    assert hits
    assert hits[0].id.startswith("fresh:")
    assert any(h.must_include for h in hits)
