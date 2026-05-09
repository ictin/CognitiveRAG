import json
import requests
from CognitiveRAG.llm.clients_ollama import OllamaClient


def test_ollama_client_sanitizes_metadata(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        # capture payload for assertions
        captured['url'] = url
        captured['payload'] = json
        # assert that metadata keys are not present in any message content
        payload_str = json and json.get('messages')
        assert payload_str is not None
        # inspect user message content
        user_msg = None
        for m in json.get('messages', []):
            if m.get('role') == 'user':
                user_msg = m.get('content', '')
        assert user_msg is not None
        # The user content should still contain the human instruction
        assert 'yes, tell me a short story' in user_msg
        # It should not contain metadata labels
        assert 'message_id' not in user_msg
        assert 'sender_id' not in user_msg
        assert 'chat_id' not in user_msg
        assert 'Gemma 4' not in user_msg

        class FauxResp:
            def raise_for_status(self):
                return None

            def json(self):
                return {'message': {'content': 'Once upon a time...'}}

        return FauxResp()

    monkeypatch.setattr(requests, 'post', fake_post)

    client = OllamaClient('http://localhost:8000', api_path='/api', api_key=None)

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Sender (untrusted metadata):\n{"message_id":"5379","sender":"Gemma 4","sender_id":"7694802153","chat_id":"telegram:7694802153"}\n\nyes, tell me a short story'}
    ]

    resp = client.chat('gemma3', messages, max_tokens=128, stream=False)
    assert isinstance(resp, dict)
    assert resp.get('message', {}).get('content', '').startswith('Once upon')
