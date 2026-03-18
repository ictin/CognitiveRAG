from fastapi.testclient import TestClient
from CognitiveRAG.app import app

def test_app_starts():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
