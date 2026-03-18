from fastapi.testclient import TestClient
import os, json, types, sys
from pathlib import Path


def test_promoted_influences_synthesizer_prompt(tmp_path):
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
    client = TestClient(app)
    resp = client.post('/promote_session', json={'session_id': session_id})
    assert resp.status_code == 200

    # Prepare fake LLM clients to capture synthesizer user prompt
    recorded = {}

    class FakeSynthClient:
        async def ainvoke_text(self, system_prompt=None, user_prompt=None):
            recorded['system'] = system_prompt
            recorded['user'] = user_prompt
            return 'influenced-answer'

    class FakePlannerClient:
        async def ainvoke(self, *a, **k):
            return {'objective': 'noop', 'steps': []}

    class FakeCriticClient:
        async def ainvoke(self, *a, **k):
            return {'approved': True, 'issues': [], 'follow_up_actions': []}

    class LLMCs:
        def __init__(self):
            self.planner = FakePlannerClient()
            self.synthesizer = FakeSynthClient()
            self.critic = FakeCriticClient()

    # Create factory returning real Orchestrator initialized with fake LLMs
    def OrchestratorFactory():
        from CognitiveRAG.agents.orchestrator import Orchestrator as RealOrch
        class SimpleRouter:
            def route(self, q):
                return types.SimpleNamespace(use_episodic=False)
        class SimpleRetriever:
            async def retrieve(self, q, plan, policy=None):
                return types.SimpleNamespace(chunks=[])
        orch = RealOrch(settings=None, llm_clients=LLMCs(), router=SimpleRouter(), retriever=SimpleRetriever(), episodic_store=types.SimpleNamespace(upsert=lambda *a, **k: None), task_store=None, reasoning_store=None)
        return orch

    # Inject factory into package
    try:
        import importlib
        agents_pkg = importlib.import_module('CognitiveRAG.agents')
        setattr(agents_pkg, 'Orchestrator', OrchestratorFactory)
        injected = True
    except Exception:
        agents_mod = types.ModuleType('CognitiveRAG.agents')
        agents_mod.Orchestrator = OrchestratorFactory
        sys.modules['CognitiveRAG.agents'] = agents_mod
        injected = False

    try:
        # Call /query which should construct Orchestrator via our injected factory and invoke synthesizer
        resp2 = client.post('/query', json={'query': 'Tell me about Zed', 'session_id': session_id})
        assert resp2.status_code == 200
        body = resp2.json()
        assert 'answer' in body
        # ensure the real synthesizer client saw session-derived content in user prompt
        assert recorded.get('user') is not None
        assert 'Crucial answer detail' in recorded.get('user')
    finally:
        # restore any module changes
        if injected:
            try:
                delattr(agents_pkg, 'Orchestrator')
            except Exception:
                pass
        else:
            if 'CognitiveRAG.agents' in sys.modules:
                del sys.modules['CognitiveRAG.agents']
        # cleanup
        sum_file.unlink()
