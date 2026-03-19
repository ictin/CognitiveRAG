from fastapi.testclient import TestClient
from pathlib import Path

from CognitiveRAG import main_server

client = TestClient(main_server.app)


def test_message_part_ingest_uses_durable_store(tmp_path, monkeypatch):
    monkeypatch.setenv('COGNITIVERAG_SKIP_KB', '1')
    db_path = tmp_path / 'message_parts.sqlite3'

    import CognitiveRAG.session_memory.message_parts_store as mp_mod
    real_store_cls = mp_mod.MessagePartsStore
    monkeypatch.setattr(mp_mod, 'MessagePartsStore', lambda: real_store_cls(db_path=str(db_path)))

    body = {
        'session_id': 'sess-123',
        'message_id': 'msg-1',
        'part_index': 0,
        'text': 'hello part',
    }

    resp = client.post('/session_append_message_part', json=body)
    assert resp.status_code == 200
    assert resp.json()['status'] in ('inserted', 'updated')

    store = real_store_cls(db_path=str(db_path))
    parts = store.get_parts('sess-123', 'msg-1')
    assert parts == [{'part_index': 0, 'text': 'hello part', 'meta_json': None}]
    assert not (Path.cwd() / 'data' / 'session_memory' / 'parts_sess-123_msg-1.json').exists()
