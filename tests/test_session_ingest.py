from fastapi.testclient import TestClient
from pathlib import Path


def test_session_append_message_endpoint_uses_durable_store(tmp_path, monkeypatch):
    monkeypatch.setenv('COGNITIVERAG_SKIP_KB', '1')
    db_path = tmp_path / 'conversations.sqlite3'

    import CognitiveRAG.session_memory.conversation_store as cs_mod
    real_store_cls = cs_mod.ConversationStore
    monkeypatch.setattr(cs_mod, 'ConversationStore', lambda: real_store_cls(db_path=str(db_path)))

    from CognitiveRAG.main_server import app
    client = TestClient(app)

    payload = {
        'session_id': 'ingest_sess',
        'message_id': 'm1',
        'sender': 'user',
        'text': 'Hello world',
    }
    fallback_file = Path.cwd() / 'data' / 'session_memory' / 'raw_ingest_sess.json'
    if fallback_file.exists():
        fallback_file.unlink()

    resp = client.post('/session_append_message', json=payload)
    assert resp.status_code == 200
    assert resp.json()['status'] in ('inserted', 'updated')

    store = real_store_cls(db_path=str(db_path))
    messages = store.get_messages('ingest_sess')
    assert messages == [
        {
            'message_id': 'm1',
            'sender': 'user',
            'text': 'Hello world',
            'created_at': None,
        }
    ]
    assert not fallback_file.exists()
