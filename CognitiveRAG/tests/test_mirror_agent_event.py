from fastapi.testclient import TestClient
from CognitiveRAG import main_server
import CognitiveRAG.client as cl

client = TestClient(main_server.app)


def test_mirror_agent_event(monkeypatch):
    import requests

    def fake_post(url, json=None, timeout=None):
        path = url.split('://')[-1].split('/',1)[-1]
        if not path.startswith('/'):
            path = '/' + path
        r = client.post(path, json=json)
        class R:
            status_code = r.status_code
            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(r.text)
            def json(self):
                return r.json()
        return R()

    monkeypatch.setattr('requests.post', fake_post)

    event = {
        'session_id': 's-evt-1',
        'turn_id': 'te1',
        'user_text': 'Please store this',
        'assistant_text': 'Stored',
        'part_text': 'p0',
        'context_item_id': 'ctx-te1',
        'context_payload': {'meta':'evt'}
    }

    res = cl.mirror_agent_event(event)
    assert 'user' in res and 'assistant' in res
    assert 'part' in res and 'context' in res
    assert res['context'].get('status') in ('inserted','updated')
