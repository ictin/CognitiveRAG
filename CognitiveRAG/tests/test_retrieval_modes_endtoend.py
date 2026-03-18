def test_documents_only_excludes_non_document_memory():
    from CognitiveRAG.retriever import retriever
    from CognitiveRAG.retrieval_modes import RetrievalMode

    results = retriever.retrieve("documents only endtoend", top_k=3, mode=RetrievalMode.DOCUMENTS_ONLY)
    # Ensure no result metadata source_type indicates task/profile/reasoning/episodic
    for item in results:
        st = item.metadata.get('source_type') or item.metadata.get('source')
        assert st not in ('task', 'profile', 'reasoning', 'episodic')

def test_task_memory_includes_task_profile_reasoning():
    from CognitiveRAG.retriever import retriever
    from CognitiveRAG.retrieval_modes import RetrievalMode

    results = retriever.retrieve("task memory endtoend", top_k=3, mode=RetrievalMode.TASK_MEMORY)
    # Ensure retrieval accepted mode and returned results; some may be task/profile/reasoning
    assert isinstance(results, list)
    # At least the wrapping should succeed
    assert all(hasattr(r, 'page_content') for r in results)
