from __future__ import annotations

from fastapi.testclient import TestClient

from CognitiveRAG.app import app


def test_documents_only_retrieval_sources_have_ranking_fields():
    with TestClient(app) as client:
        r = client.post(
            "/query",
            json={"query": "What is this document about?", "retrieval_mode": "documents_only"},
        )
        assert r.status_code == 200, r.text
        resp = r.json()

    trace = resp.get("trace", {})
    summary = trace.get("retrieval_summary", [])
    sources = trace.get("retrieval_sources", [])

    assert summary, resp
    assert sources, resp
    assert summary == [x.get("chunk_id") for x in sources], (summary, sources)

    for idx, item in enumerate(sources, start=1):
        assert item.get("rank") == idx, item
        assert "final_score" in item, item
        assert "ranking_reason" in item, item
