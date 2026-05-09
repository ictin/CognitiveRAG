from fastapi.testclient import TestClient
import os, json
from pathlib import Path
from types import SimpleNamespace


def test_promoted_influences_synthesizer_prompt(monkeypatch, tmp_path):
    session_id = 'answer_sess'
    data_dir = Path(os.getcwd()) / 'data' / 'session_memory'
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        {'chunk_index': 0, 'summary': 'Crucial answer detail: Zed.'},
    ]
    sum_file = data_dir / f'summaries_{session_id}.json'
    with open(sum_file, 'w', encoding='utf-8') as f:
        json.dump(summaries, f)

    # Promote via endpoint
    from CognitiveRAG.main_server import app
    import CognitiveRAG.config as config
    import CognitiveRAG.llm_provider as llm_provider

    captured = {}

    class FakeOllamaModel:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt, *args, **kwargs):
            captured['prompt'] = prompt
            return SimpleNamespace(content='influenced-answer')

    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama", raising=False)
    monkeypatch.setattr(llm_provider, "ChatOllama", FakeOllamaModel, raising=True)

    client = TestClient(app)
    resp = client.post('/promote_session', json={'session_id': session_id})
    assert resp.status_code == 200

    resp2 = client.post('/query', json={'query': 'Tell me about Zed', 'session_id': session_id})
    assert resp2.status_code == 200
    body = resp2.json()
    assert body['answer'] == 'influenced-answer'
    # ensure the real provider-bound prompt saw the promoted session summary
    assert captured.get('prompt') is not None
    assert 'Crucial answer detail' in captured['prompt']
    assert 'Tell me about Zed' in captured['prompt']

    # cleanup
    sum_file.unlink(missing_ok=True)
