def test_retriever_returns_retrievedchunk():
    from CognitiveRAG.retriever import retriever
    from CognitiveRAG.retrieval_contract import RetrievedChunk

    # Call retrieve with a simple query; do not rely on external web search
    results = retriever.retrieve("test query", top_k=1)
    assert isinstance(results, list), "Retriever should return a list"
    if results:
        first = results[0]
        assert isinstance(first, RetrievedChunk), "Items must be RetrievedChunk instances"
        assert hasattr(first, 'page_content'), "RetrievedChunk must have page_content"
        assert hasattr(first, 'metadata'), "RetrievedChunk must have metadata"
        assert hasattr(first, 'rank'), "RetrievedChunk must have rank"
        assert hasattr(first, 'final_score'), "RetrievedChunk must have final_score"
        assert hasattr(first, 'augmentation_decision'), "RetrievedChunk must have augmentation_decision"
