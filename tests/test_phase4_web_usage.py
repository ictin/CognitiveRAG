from __future__ import annotations

from fastapi.testclient import TestClient

from CognitiveRAG.app import app
from CognitiveRAG.schemas.retrieval import RetrievedChunk


async def _fake_search(query: str, top_k: int = 5):
    return [
        RetrievedChunk(
            chunk_id="web_fake_1",
            document_id=None,
            text="Fake web result for testing",
            source_type="web",
            score=9.9,
            metadata={"origin": "fake-web"},
        )
    ]


def _decision(trace: dict):
    return (
        trace.get("web_decision")
        or trace.get("retrieval_decision")
        or trace.get("augmentation_decision")
    )


def _force_route_use_web(services):
    orig_route = services.router.route

    def force_route(query: str):
        plan = orig_route(query)
        plan.use_web = True
        return plan

    return orig_route, force_route


def test_full_memory_can_use_web_when_allowed():
    with TestClient(app) as client:
        services = client.app.state.services
        orig_route, forced_route = _force_route_use_web(services)
        orig_search = services.retriever.web_search.search
        services.router.route = forced_route
        services.retriever.web_search.search = _fake_search
        try:
            r = client.post(
                "/query",
                json={"query": "latest current news", "retrieval_mode": "full_memory"},
            )
            assert r.status_code == 200, r.text
            resp = r.json()
        finally:
            services.router.route = orig_route
            services.retriever.web_search.search = orig_search

    trace = resp.get("trace", {})
    decision = _decision(trace)
    sources = trace.get("retrieval_sources", [])

    assert decision, trace
    assert decision.get("used") is True, decision
    assert any(s.get("source_type") == "web" for s in sources), sources


def test_documents_only_does_not_use_web_even_if_route_requests_it():
    with TestClient(app) as client:
        services = client.app.state.services
        orig_route, forced_route = _force_route_use_web(services)
        orig_search = services.retriever.web_search.search
        services.router.route = forced_route
        services.retriever.web_search.search = _fake_search
        try:
            r = client.post(
                "/query",
                json={"query": "latest current news", "retrieval_mode": "documents_only"},
            )
            assert r.status_code == 200, r.text
            resp = r.json()
        finally:
            services.router.route = orig_route
            services.retriever.web_search.search = orig_search

    trace = resp.get("trace", {})
    decision = _decision(trace)
    summary = trace.get("retrieval_summary", [])
    sources = trace.get("retrieval_sources", [])

    assert decision, trace
    assert decision.get("used") is False, decision
    assert all(s.get("source_type") != "web" for s in sources), sources
    assert all(x.startswith("doc_") for x in summary), summary


def test_regression_test_does_not_use_web_even_if_route_requests_it():
    with TestClient(app) as client:
        services = client.app.state.services
        orig_route, forced_route = _force_route_use_web(services)
        orig_search = services.retriever.web_search.search
        services.router.route = forced_route
        services.retriever.web_search.search = _fake_search
        try:
            r = client.post(
                "/query",
                json={"query": "latest current news", "retrieval_mode": "regression_test"},
            )
            assert r.status_code == 200, r.text
            resp = r.json()
        finally:
            services.router.route = orig_route
            services.retriever.web_search.search = orig_search

    trace = resp.get("trace", {})
    decision = _decision(trace)
    summary = trace.get("retrieval_summary", [])
    sources = trace.get("retrieval_sources", [])

    assert decision, trace
    assert decision.get("used") is False, decision
    assert all(s.get("source_type") != "web" for s in sources), sources
    assert all(x.startswith("doc_") for x in summary), summary


def test_task_memory_does_not_use_web_even_if_route_requests_it():
    with TestClient(app) as client:
        services = client.app.state.services
        orig_route, forced_route = _force_route_use_web(services)
        orig_search = services.retriever.web_search.search
        services.router.route = forced_route
        services.retriever.web_search.search = _fake_search
        try:
            r = client.post(
                "/query",
                json={"query": "latest current news", "retrieval_mode": "task_memory"},
            )
            assert r.status_code == 200, r.text
            resp = r.json()
        finally:
            services.router.route = orig_route
            services.retriever.web_search.search = orig_search

    trace = resp.get("trace", {})
    decision = _decision(trace)
    summary = trace.get("retrieval_summary", [])
    sources = trace.get("retrieval_sources", [])

    assert decision, trace
    assert decision.get("used") is False, decision
    assert all(s.get("source_type") != "web" for s in sources), sources
    assert all(not x.startswith("evt_") for x in summary), summary
