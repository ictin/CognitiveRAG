from fastapi.testclient import TestClient
import os, json
from pathlib import Path


def test_context_item_upsert_endpoint_and_client(tmp_path):
    from CognitiveRAG.main_server import app
    client = TestClient(app)

    payload = {
        'item_id': 'item1',
        'session_id': 'sessA',
        'type': 'note',
        'payload_json': {'k':'v'},
    }

    # call endpoint
    resp = client.post('/session_upsert_context_item', json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body['status'] in ('inserted','updated')

    # call again (idempotent upsert)
    resp2 = client.post('/session_upsert_context_item', json=payload)
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2['status'] in ('inserted','updated')

    # client helper test via monkeypatch
    import requests
    def fake_post(url, json=None, timeout=None):
        path = url.split('://')[-1].split('/',1)[-1]
        if not path.startswith('/'):
            path = '/' + path
        r = client.post(path, json=json)
        class R:
            status_code = r.status_code
            def raise_for_status(self):
                if self.status_code>=400:
                    raise requests.HTTPError(r.text)
            def json(self):
                return r.json()
        return R()

    import CognitiveRAG.client as cl
    cl_requests = __import__('requests')
    old_post = cl_requests.post
    cl_requests.post = fake_post
    try:
        res = cl.upsert_context_item('item2', 'sessB', 'tag', {'a':1})
        assert isinstance(res, dict)
        assert res['status'] in ('inserted','updated')
    finally:
        cl_requests.post = old_post

    # cleanup fallback file
    data_dir = Path(os.getcwd()) / 'data' / 'session_memory'
    raw_path = data_dir / f'context_item_item1.json'
    if raw_path.exists():
        raw_path.unlink()
