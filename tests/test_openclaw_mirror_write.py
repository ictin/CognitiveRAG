import json
from fastapi.testclient import TestClient
from CognitiveRAG import main_server

client = TestClient(main_server.app)

def test_openclaw_mirror_write_endtoend(tmp_path):
    session_id = 'sess-mw-1'
    message_id = 'm-1'

    # append message
    r = client.post('/session_append_message', json={
        'session_id': session_id,
        'message_id': message_id,
        'sender': 'user',
        'text': 'Hello world'
    })
    assert r.status_code == 200

    # append part
    r2 = client.post('/session_append_message_part', json={
        'session_id': session_id,
        'message_id': message_id,
        'part_index': 0,
        'text': 'part text'
    })
    assert r2.status_code == 200

    # upsert context item
    item_id = 'itm-1'
    r3 = client.post('/session_upsert_context_item', json={
        'item_id': item_id,
        'session_id': session_id,
        'type': 'note',
        'payload_json': {'k':'v'}
    })
    assert r3.status_code == 200

    # basic checks on responses
    assert r.json().get('status') in ('inserted','updated','ok',None)
    assert r2.json().get('status') in ('inserted','updated','ok',None)
    assert r3.json().get('status') in ('inserted','updated')
