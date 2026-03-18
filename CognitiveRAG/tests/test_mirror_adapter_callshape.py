from fastapi.testclient import TestClient
from CognitiveRAG import main_server
import CognitiveRAG.client as cl

client = TestClient(main_server.app)


def test_mirror_write_callshape(monkeypatch):
    # realistic OpenClaw-like parameters
    session_id = 'oc-s-1'
    message_id = 'oc-msg-1'
    sender = 'openclaw-bot'
    text = 'Action: store this'
    part_text = 'chunk-0'
    context_item_id = 'oc-ctx-1'
    context_payload = {'source':'openclaw','importance':5}

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

    monkeypatch.setattr('requests.post', fake_post)

    res = cl.mirror_write_interaction(session_id, message_id, sender, text, part_text, context_item_id, context_payload)
    assert isinstance(res, dict)
    assert set(res.keys()) == {'message','part','context'}
    assert res['context'].get('status') in ('inserted','updated')
