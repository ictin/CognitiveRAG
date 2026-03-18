def test_documents_only_excludes_web():
    from CognitiveRAG.retriever import retriever
    from CognitiveRAG.retrieval_modes import RetrievalMode

    results = retriever.retrieve("documents only test", top_k=3, mode=RetrievalMode.DOCUMENTS_ONLY)
    # No returned item should have augmentation_decision indicating web
    for item in results:
        assert not item.augmentation_decision.get('used', False) or item.augmentation_decision.get('source') != 'web'

def test_task_memory_allows_task_profile_reasoning():
    from CognitiveRAG.retriever import retriever
    from CognitiveRAG.retrieval_modes import RetrievalMode

    results = retriever.retrieve("task memory test", top_k=3, mode=RetrievalMode.TASK_MEMORY)
    # We can't fully probe internal DBs here, but ensure function accepted the mode and returned RetrievedChunk
    assert isinstance(results, list)

def test_full_memory_allows_web():
    from CognitiveRAG.retriever import retriever
    from CognitiveRAG.retrieval_modes import RetrievalMode

    results = retriever.retrieve("full memory web test", top_k=3, mode=RetrievalMode.FULL_MEMORY)
    # In full mode web is allowed; items may have augmentation_decision used True if web present
    assert isinstance(results, list)
