def test_retrieval_policy_defaults_populate_provenance_and_exactness():
    from CognitiveRAG.schemas.retrieval import RetrievedChunk, RetrievalBundle

    exact_chunk = RetrievedChunk(
        chunk_id="exact-1",
        text="exact text",
        source_type="document",
        exactness="exact",
        summarizable=True,
    ).with_policy_defaults()

    derived_chunk = RetrievedChunk(
        chunk_id="derived-1",
        text="derived text",
        source_type="vector",
    ).with_policy_defaults()

    bundle = RetrievalBundle(
        query="hello",
        intent="test",
        chunks=[exact_chunk, derived_chunk],
        provenance={"query": "hello", "intent": "test"},
    )

    assert exact_chunk.provenance == {
        "chunk_id": "exact-1",
        "document_id": None,
        "source_type": "document",
    }
    assert exact_chunk.exactness == "exact"
    assert exact_chunk.summarizable is False

    assert derived_chunk.provenance == {
        "chunk_id": "derived-1",
        "document_id": None,
        "source_type": "vector",
    }
    assert derived_chunk.exactness == "derived"
    assert derived_chunk.summarizable is True

    assert bundle.provenance == {"query": "hello", "intent": "test"}
    assert len(bundle.chunks) == 2
