def test_reasoning_records_are_retrievable(tmp_path):
    from CognitiveRAG.memory.reasoning_store import ReasoningStore
    from CognitiveRAG.schemas.memory import ReasoningPattern
    from CognitiveRAG.retriever import set_memory_stores
    from CognitiveRAG.retrieval_modes import RetrievalMode
    from CognitiveRAG.retrieval_contract import RetrievedChunk
    from CognitiveRAG.retriever import retriever

    dbpath = tmp_path / "reasoning.sqlite3"
    rs = ReasoningStore(dbpath)

    rp = ReasoningPattern(
        pattern_id="rp_test1",
        problem_signature="how to reuse reasoning",
        reasoning_steps=["think", "test"],
        solution_summary="use promote and retrieve",
        confidence=0.9,
        provenance=["cite://example"]
    )
    rs.upsert(rp)

    # inject reasoning store into retriever
    set_memory_stores(reasoning_store=rs)

    # documents_only should exclude reasoning
    docs_only = retriever.retrieve("reuse reasoning", top_k=5, mode=RetrievalMode.DOCUMENTS_ONLY)
    assert all(getattr(r, 'metadata', {}).get('source_type') != 'reasoning' for r in docs_only)

    # task_memory should include reasoning results
    results = retriever.retrieve("reuse reasoning", top_k=5, mode=RetrievalMode.TASK_MEMORY)
    assert isinstance(results, list)
    found = False
    for item in results:
        # normalized metadata should expose provenance
        prov = item.metadata.get('provenance')
        if prov and 'cite://example' in prov:
            found = True
            assert item.metadata.get('source_type') == 'reasoning' or item.metadata.get('source_type') == 'unknown'
    assert found, "Reasoning provenance not found in retrieval results"

    # cleanup injection
    set_memory_stores(reasoning_store=None)
