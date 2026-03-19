from fastapi.testclient import TestClient
from CognitiveRAG import main_server
import CognitiveRAG.client as cl

client = TestClient(main_server.app)


def test_mirror_turn(monkeypatch):
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

    res = cl.mirror_turn(
        session_id='s-turn-1',
        turn_id='t1',
        user_text='Hello assistant',
        assistant_text='Hello user',
        part_text='assistant-part-0',
        context_item_id='ctx-t1',
        context_payload={'note': 'from mirror_turn'},
    )

    assert 'user' in res and 'assistant' in res
    assert 'part' in res and 'context' in res
    assert res['context'].get('status') in ('inserted','updated')
