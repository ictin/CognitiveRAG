import os, json
from pathlib import Path
from CognitiveRAG.memory.reasoning_store import ReasoningStore
from CognitiveRAG.retriever import HybridRetriever
from CognitiveRAG.retrieval_modes import RetrievalMode
from CognitiveRAG.core.settings import settings
from fastapi.testclient import TestClient


def test_promoted_influences_live_query(tmp_path):
    session_id = 'live_sess'
    data_dir = Path(os.getcwd()) / 'data' / 'session_memory'
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        {'chunk_index': 0, 'summary': 'Influence summary about Y that should be retrieved.'},
    ]
    sum_file = data_dir / f'summaries_{session_id}.json'
    with open(sum_file, 'w', encoding='utf-8') as f:
        json.dump(summaries, f)

    # Call promote_session endpoint to promote into durable reasoning DB (settings.store.reasoning_db_path)
    from CognitiveRAG.main_server import app
    client = TestClient(app)
    resp = client.post('/promote_session', json={'session_id': session_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body['promoted_count'] >= 1

    # Now construct a retriever that points at the same reasoning DB and run a retrieval in TASK_MEMORY mode
    rs = ReasoningStore(settings.store.reasoning_db_path)

    class DummyKB:
        bm25_index = None
        doc_store = {}
        vector_store = None

    retriever = HybridRetriever(DummyKB(), reasoning_store=rs)
    docs = retriever.retrieve('summary about Y', top_k=5, mode=RetrievalMode.TASK_MEMORY)

    found = False
    for d in docs:
        meta = getattr(d, 'metadata', {}) or {}
        if meta.get('source_type') == 'reasoning' or meta.get('source_type') == 'reasoning':
            found = True
            break
    assert found, 'Promoted reasoning item not found in live retrieval results'

    # cleanup
    sum_file.unlink()
