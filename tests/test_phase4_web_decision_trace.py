from __future__ import annotations

from fastapi.testclient import TestClient

from CognitiveRAG.app import app


def _decision_from_trace(trace: dict):
    return (
        trace.get("web_decision")
        or trace.get("retrieval_decision")
        or trace.get("augmentation_decision")
    )


def test_documents_only_has_web_decision_trace():
    with TestClient(app) as client:
        r = client.post(
            "/query",
            json={"query": "What is this document about?", "retrieval_mode": "documents_only"},
        )
        assert r.status_code == 200, r.text
        resp = r.json()

    trace = resp.get("trace", {})
    decision = _decision_from_trace(trace)

    assert trace.get("retrieval_summary"), resp
    assert decision, trace
    assert "considered" in decision, decision
    assert "allowed" in decision, decision
    assert "used" in decision, decision
    assert "reason" in decision, decision
    assert all(x.startswith("doc_") for x in trace.get("retrieval_summary", [])), trace


def test_full_memory_latest_current_news_has_web_decision_trace():
    with TestClient(app) as client:
        r = client.post(
            "/query",
            json={"query": "latest current news", "retrieval_mode": "full_memory"},
        )
        assert r.status_code == 200, r.text
        resp = r.json()

    trace = resp.get("trace", {})
    decision = _decision_from_trace(trace)

    assert decision, trace
    assert "considered" in decision, decision
    assert "allowed" in decision, decision
    assert "used" in decision, decision
    assert "reason" in decision, decision


def test_task_memory_has_web_decision_trace_and_no_episodic_summary():
    with TestClient(app) as client:
        r = client.post(
            "/query",
            json={"query": "What is this document about?", "retrieval_mode": "task_memory"},
        )
        assert r.status_code == 200, r.text
        resp = r.json()

    trace = resp.get("trace", {})
    decision = _decision_from_trace(trace)
    summary = trace.get("retrieval_summary", [])

    assert decision, trace
    assert all(not x.startswith("evt_") for x in summary), summary
    assert "used" in decision, decision
