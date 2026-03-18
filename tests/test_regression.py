import time
from pathlib import Path

from fastapi.testclient import TestClient

from CognitiveRAG.app import app


MARKER = f"test-{int(time.time())}"


def test_lexical_only_returns_docs():
    Path("data/source_documents").mkdir(parents=True, exist_ok=True)
    Path("data/source_documents/smoke.txt").write_text(
        f"Hello world. This is a test document about OpenClaw and CognitiveRAG. Marker: {MARKER}\n",
        encoding="utf-8",
    )
    try:
        Path("data/episodic.sqlite3").unlink()
    except FileNotFoundError:
        pass

    with TestClient(app) as client:
        ingest_body = {
            "path": str(Path("data/source_documents/smoke.txt").resolve()),
            "recursive": False,
        }
        r1 = client.post("/ingest", json=ingest_body)
        assert r1.status_code == 200

        qbody = {"query": "What is this document about?", "lexical_only": True}
        r2 = client.post("/query", json=qbody)
        assert r2.status_code == 200

        resp = r2.json()
        summary = resp.get("trace", {}).get("retrieval_summary", [])
        assert len(summary) > 0
        assert all(x.startswith("doc_") for x in summary)


def test_planner_returns_steps():
    with TestClient(app) as client:
        qbody = {"query": "What is this document about?", "lexical_only": True}
        r = client.post("/query", json=qbody)
        assert r.status_code == 200

        resp = r.json()
        plan = resp.get("trace", {}).get("plan", {})
        assert isinstance(plan.get("objective"), str)
        assert len(plan.get("objective", "")) > 0

        steps = plan.get("steps", [])
        assert isinstance(steps, list)
        assert len(steps) > 0

        for s in steps:
            assert isinstance(s, dict)
            assert s.get("description", "").strip()


def test_critic_schema_keys():
    with TestClient(app) as client:
        qbody = {"query": "What is this document about?", "lexical_only": True}
        r = client.post("/query", json=qbody)
        assert r.status_code == 200

        resp = r.json()
        crit = resp.get("trace", {}).get("critique", {})
        assert "approved" in crit
        assert "issues" in crit
        assert "follow_up_actions" in crit
        assert isinstance(crit.get("issues"), list)
        assert isinstance(crit.get("follow_up_actions"), list)
