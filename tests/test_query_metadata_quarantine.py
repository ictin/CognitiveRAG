from types import SimpleNamespace

from fastapi.testclient import TestClient


METADATA_RICH = """Sender (untrusted metadata):
Gemma 4
{
  "chat_id": "telegram:7694802153",
  "message_id": "5487",
  "sender_id": "7694802153",
  "sender": "Gogu",
  "timestamp": "2026-05-09T10:00:00Z"
}

Tell me a short story
"""


def _build_query_orchestrator_for_provider():
    class _Orchestrator:
        def run(self, query: str, session_id=None):
            import CognitiveRAG.llm_provider as llm_provider

            llm = llm_provider.get_llm("provider-model")
            msg = llm.invoke(query)
            answer = msg.content if hasattr(msg, "content") else str(msg)
            return {"answer": answer}

    return _Orchestrator()


def test_query_ollama_path_captures_sanitized_provider_payload(monkeypatch):
    import CognitiveRAG.config as config
    import CognitiveRAG.llm_provider as llm_provider
    import CognitiveRAG.main_server as main_server

    captured = {}

    class FakeOllamaModel:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt, *args, **kwargs):
            captured["provider"] = "ollama"
            captured["prompt"] = prompt
            return SimpleNamespace(content="story via ollama")

    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama", raising=False)
    monkeypatch.setattr(llm_provider, "ChatOllama", FakeOllamaModel, raising=True)
    monkeypatch.setattr(main_server, "_build_simple_query_orchestrator", _build_query_orchestrator_for_provider, raising=True)

    client = TestClient(main_server.app)
    resp = client.post("/query", json={"query": METADATA_RICH, "session_id": "s-ollama"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "story via ollama"
    assert "Tell me a short story" in captured["prompt"]
    assert "sender_id" not in captured["prompt"]
    assert "message_id" not in captured["prompt"]
    assert "chat_id" not in captured["prompt"]
    assert "timestamp" not in captured["prompt"]
    assert "Gemma 4" not in captured["prompt"]


def test_query_openai_compatible_path_captures_sanitized_provider_payload(monkeypatch):
    import CognitiveRAG.config as config
    import CognitiveRAG.llm_provider as llm_provider
    import CognitiveRAG.main_server as main_server

    captured = {}

    class FakeOpenAIModel:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt, *args, **kwargs):
            captured["provider"] = "openai-compatible"
            captured["prompt"] = prompt
            return SimpleNamespace(content="story via gpt-5-mini path")

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(llm_provider, "ChatOpenAI", FakeOpenAIModel, raising=True)
    monkeypatch.setattr(main_server, "_build_simple_query_orchestrator", _build_query_orchestrator_for_provider, raising=True)

    client = TestClient(main_server.app)
    resp = client.post("/query", json={"query": METADATA_RICH, "session_id": "s-openai"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "story via gpt-5-mini path"
    assert "Tell me a short story" in captured["prompt"]
    assert "sender_id" not in captured["prompt"]
    assert "message_id" not in captured["prompt"]
    assert "chat_id" not in captured["prompt"]
    assert "timestamp" not in captured["prompt"]
    assert "Gemma 4" not in captured["prompt"]


def test_start_new_conversation_command_routes_without_llm(monkeypatch):
    import CognitiveRAG.main_server as main_server

    called = {"orchestrator_run": False}

    class FailingOrchestrator:
        def run(self, query: str, session_id=None):
            called["orchestrator_run"] = True
            raise AssertionError("LLM path should not be called for start a new conversation")

    monkeypatch.setattr(main_server, "_build_simple_query_orchestrator", lambda: FailingOrchestrator(), raising=True)
    client = TestClient(main_server.app)
    resp = client.post("/query", json={"query": "start a new conversation", "session_id": "s-cmd"})
    assert resp.status_code == 200
    assert "Started a new conversation" in resp.json()["answer"]
    assert called["orchestrator_run"] is False
