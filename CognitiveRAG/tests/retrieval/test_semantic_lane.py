from CognitiveRAG.crag.retrieval.semantic_lane import retrieve


def test_semantic_lane_contract_shape():
    hits = retrieve(
        query="memory organization",
        session_id="s2",
        fresh_tail=[{"text": "memory organized in layers"}],
        older_raw=[],
        summaries=[{"summary": "layered memory overview"}],
        top_k=4,
    )
    assert hits
    first = hits[0]
    assert first.lane.value == "semantic"
    assert first.semantic_score >= 0
    assert first.tokens > 0
