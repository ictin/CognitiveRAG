def test_metadata_normalization_on_retrieval():
    from CognitiveRAG.retriever import retriever
    from CognitiveRAG.retrieval_contract import RetrievedChunk

    results = retriever.retrieve('meta test', top_k=2)
    assert isinstance(results, list)
    if results:
        r = results[0]
        assert isinstance(r.metadata, dict)
        # canonical keys present
        for key in ['source_type','project','created_at','updated_at','origin_id']:
            assert key in r.metadata
        assert isinstance(r.metadata.get('topic_tags', []), list)
        assert isinstance(r.metadata.get('entity_tags', []), list)
