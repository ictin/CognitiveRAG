def test_retriever_uses_injected_store():
    from CognitiveRAG.retriever import set_memory_stores, _injected_task_store
    from CognitiveRAG.retrieval_modes import RetrievalMode

    class DummyTaskStore:
        def query(self, query, top_k=5):
            return [{
                'chunk_id': 'dummy1',
                'text': 'dummy task result',
                'source_type': 'task',
                'metadata': {'injected': True}
            }]

    dummy = DummyTaskStore()
    # inject and run
    set_memory_stores(task_store=dummy)

    from CognitiveRAG.retriever import retriever
    results = retriever.retrieve('injection test', top_k=1, mode=RetrievalMode.TASK_MEMORY)
    # expect at least one returned item carrying injected metadata
    assert any(getattr(r, 'metadata', {}).get('injected') for r in results)

    # cleanup: clear injection
    set_memory_stores(task_store=None)
