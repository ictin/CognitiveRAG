import requests
from tools.cognitiverag_shim import emit_cognitiverag_event


def test_emit_cognitiverag_event(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        class R:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {'ok': True, 'url': url, 'json': json}
        return R()

    monkeypatch.setattr('requests.post', fake_post)

    event = {
        'session_id': 's-openclaw',
        'turn_id': 't-open',
        'user_text': 'User asks X',
        'assistant_text': 'Assistant answers',
        'part_text': 'part-0',
        'context_item_id': 'ctx-open',
        'context_payload': {'meta': 'v'}
    }

    out = emit_cognitiverag_event(event, base_url='http://fake')
    assert 'user' in out and 'assistant' in out and 'part' in out and 'context' in out
    # ensure calls recorded in expected order
    assert calls[0][0].endswith('/session_append_message')
    assert calls[1][0].endswith('/session_append_message')
    assert calls[2][0].endswith('/session_append_message_part')
    assert calls[3][0].endswith('/session_upsert_context_item')
