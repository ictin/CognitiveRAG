from __future__ import annotations

from fastapi.testclient import TestClient

from CognitiveRAG.app import app


def test_documents_only_trace_includes_retrieval_sources():
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
    assert all(x.startswith("doc_") for x in summary), summary

    for item in sources:
        assert "chunk_id" in item, item
        assert "source_type" in item, item
        assert "score" in item, item


def test_task_memory_trace_sources_are_non_episodic():
    with TestClient(app) as client:
        r = client.post(
            "/query",
            json={"query": "What is this document about?", "retrieval_mode": "task_memory"},
        )
        assert r.status_code == 200, r.text
        resp = r.json()

    trace = resp.get("trace", {})
    summary = trace.get("retrieval_summary", [])
    sources = trace.get("retrieval_sources", [])

    assert summary, resp
    assert sources, resp
    assert all(not x.startswith("evt_") for x in summary), summary

    for item in sources:
        assert "chunk_id" in item, item
        assert "source_type" in item, item
        assert "score" in item, item
        assert item["source_type"] != "episodic", item
