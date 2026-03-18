def test_retrieval_metadata_fields_present():
    from CognitiveRAG.retriever import retriever
    from CognitiveRAG.retrieval_contract import RetrievedChunk

    results = retriever.retrieve("metadata test", top_k=3)
    assert isinstance(results, list)
    if results:
        for item in results:
            assert isinstance(item, RetrievedChunk)
            assert item.rank is not None
            assert item.final_score is not None
            assert item.ranking_reason is not None and isinstance(item.ranking_reason, str)
            assert isinstance(item.augmentation_decision, dict)
            assert 'used' in item.augmentation_decision
