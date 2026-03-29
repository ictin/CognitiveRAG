from CognitiveRAG.crag.retrieval.lexical_lane import retrieve


def test_lexical_lane_returns_stable_hits_with_provenance_tokens():
    hits = retrieve(
        query="youtube secrets",
        session_id="s1",
        fresh_tail=[{"text": "talked about youtube secrets yesterday"}],
        older_raw=[{"text": "other message"}],
        summaries=[{"summary": "youtube secrets summary"}],
        top_k=5,
    )
    assert hits
    assert all(h.lane.value == "lexical" for h in hits)
    assert all(h.tokens > 0 for h in hits)
    assert all(isinstance(h.provenance, dict) for h in hits)
