from types import SimpleNamespace


def test_get_llm_openai_sanitizes_but_preserves_contract(monkeypatch):
    import CognitiveRAG.config as config
    import CognitiveRAG.llm_provider as llm_provider

    captured = {}

    class FakeOpenAIModel:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt, *args, **kwargs):
            captured["prompt"] = prompt
            return SimpleNamespace(content="ok-openai")

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(llm_provider, "ChatOpenAI", FakeOpenAIModel, raising=True)

    llm = llm_provider.get_llm("gpt-5-mini")
    raw = """Sender (untrusted metadata):
Gemma 4
{"message_id":"1","sender_id":"2","chat_id":"x","timestamp":"2026-05-09T10:00:00Z"}

Tell me a short story
"""
    out = llm.invoke(raw)
    assert out.content == "ok-openai"
    assert "Tell me a short story" in captured["prompt"]
    assert "sender_id" not in captured["prompt"]
    assert "message_id" not in captured["prompt"]
    assert "chat_id" not in captured["prompt"]


def test_get_llm_ollama_sanitizes_prompt(monkeypatch):
    import CognitiveRAG.config as config
    import CognitiveRAG.llm_provider as llm_provider

    captured = {}

    class FakeOllamaModel:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt, *args, **kwargs):
            captured["prompt"] = prompt
            return SimpleNamespace(content="ok-ollama")

    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama", raising=False)
    monkeypatch.setattr(llm_provider, "ChatOllama", FakeOllamaModel, raising=True)

    llm = llm_provider.get_llm("gemma3")
    out = llm.invoke('{"sender_id":"abc","chat_id":"c"}\n\nTell me a short story')
    assert out.content == "ok-ollama"
    assert "Tell me a short story" in captured["prompt"]
    assert "sender_id" not in captured["prompt"]
