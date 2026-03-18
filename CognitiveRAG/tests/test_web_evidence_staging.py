def test_web_evidence_staging_behaviour():
    from CognitiveRAG.retriever import retriever
    from CognitiveRAG.retrieval_modes import RetrievalMode

    # Run a query in full mode — may or may not return web results depending on DDGS availability
    results_full = retriever.retrieve('openclaw test web evidence', top_k=3, mode=RetrievalMode.FULL_MEMORY)
    # Any web results present should have source_type == 'web_evidence'
    for r in results_full:
        if r.metadata.get('source_type') == 'web_evidence':
            assert r.metadata.get('document_kind') == 'web_capture'

    # In documents_only mode, no web_evidence should be returned
    results_docs = retriever.retrieve('openclaw test web evidence', top_k=3, mode=RetrievalMode.DOCUMENTS_ONLY)
    assert all(r.metadata.get('source_type') != 'web_evidence' for r in results_docs)

    # In task_memory mode, web_evidence should also be excluded
    results_task = retriever.retrieve('openclaw test web evidence', top_k=3, mode=RetrievalMode.TASK_MEMORY)
    assert all(r.metadata.get('source_type') != 'web_evidence' for r in results_task)
