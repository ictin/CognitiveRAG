def test_web_promoted_store_and_persistence(tmp_path):
    from CognitiveRAG.memory.web_promoted_store import WebPromotedStore
    from CognitiveRAG.retriever import retriever, set_memory_stores
    from CognitiveRAG.retrieval_modes import RetrievalMode

    db = tmp_path / "web_promoted.sqlite3"
    store = WebPromotedStore(db)

    # create a fake RetrievedChunk-like object
    class RC:
        def __init__(self):
            self.page_content = 'hello web'
            self.metadata = {'source': 'https://example.com', 'source_type': 'web_evidence', 'title': 'Ex'}
            self.rank = 1

    rc = RC()
    # promote via retriever helper (attach helper by calling retrieve to ensure self assigned)
    retriever.retrieve('no-op', top_k=1, mode=RetrievalMode.FULL_MEMORY)
    rid = getattr(retriever, 'promote_web_promoted')(rc, record_id='wp_test1', store=store)
    assert rid == 'wp_test1'

    # ensure stored record exists
    rec = store.get('wp_test1')
    assert rec is not None
    assert rec['source_url'] == 'https://example.com'
    assert 'hello web' in rec['page_content']
    assert rec['metadata'].get('source') == 'https://example.com'
