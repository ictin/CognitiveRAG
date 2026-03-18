from unittest.mock import patch
import requests
from fastapi.testclient import TestClient
import types, sys


def test_client_query_and_promote(monkeypatch):
    # prepare TestClient app
    from CognitiveRAG.main_server import app
    client = TestClient(app)

    # monkeypatch requests.post to route to TestClient.post
    def fake_post(url, json=None, timeout=None):
        # strip base url
        path = url.split('://')[-1].split('/', 1)[-1]
        if not path.startswith('/'):
            path = '/' + path
        resp = client.post(path, json=json)
        class R:
            status_code = resp.status_code
            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(resp.text)
            def json(self):
                return resp.json()
        return R()

    monkeypatch.setattr(requests, 'post', fake_post)

    from CognitiveRAG.client import query, promote_session

    # Test promote_session (no summaries exist -> no-op but should return structure)
    res = promote_session('nonexistent_session')
    assert isinstance(res, dict)
    assert 'promoted_count' in res

    # Ensure /query can import Orchestrator by injecting a simple stub
    import importlib, types, sys
    try:
        agents_pkg = importlib.import_module('CognitiveRAG.agents')
        async def _run(self, q, session_id=None):
            return {'answer': 'stub'}
        setattr(agents_pkg, 'Orchestrator', lambda: type('O', (), {'run': _run})())
    except Exception:
        async def _run(self, q, session_id=None):
            return {'answer': 'stub'}
        mod = types.ModuleType('CognitiveRAG.agents')
        mod.Orchestrator = lambda: type('O', (), {'run': _run})()
        sys.modules['CognitiveRAG.agents'] = mod

    # Test query (simple)
    qr = query(None, 'hello')
    assert isinstance(qr, dict)
    assert 'answer' in qr
