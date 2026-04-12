from fastapi.testclient import TestClient

from CognitiveRAG.memory.reasoning_store import ReasoningStore
from CognitiveRAG.schemas.memory import ReasoningPattern


def test_promoted_search_and_get_endpoints(tmp_path, monkeypatch):
    from CognitiveRAG import main_server

    rs = ReasoningStore(tmp_path / "reasoning.sqlite3")
    rp = ReasoningPattern(
        pattern_id="prom:proposition:test1",
        item_id="prom:proposition:test1",
        problem_signature="proposition:test-session",
        reasoning_steps=[],
        solution_summary="deployment window is tuesday 14:00 utc",
        confidence=0.91,
        provenance=[
            '{"session_id":"test-session","summary_chunk_index":0,"source_class":"promoted_memory","source_refs":[{"session_id":"test-session","summary_chunk_index":0}]}'
        ],
        memory_subtype="stable_fact",
        normalized_text="proposition:deployment window is tuesday 14:00 utc",
        freshness_state="hot",
    )
    rs.upsert(rp)

    monkeypatch.setattr(main_server, "_resolve_reasoning_store", lambda: rs)
    client = TestClient(main_server.app)

    search = client.post("/promoted_search", json={"query": "deployment window", "top_k": 5})
    assert search.status_code == 200
    body = search.json()
    assert body["count"] >= 1
    first = body["items"][0]
    assert first["source_class"] == "promoted_memory"
    assert first["memory_subtype"] == "stable_fact"
    assert first["provenance"]

    get_item = client.post("/promoted_get", json={"pattern_id": "prom:proposition:test1"})
    assert get_item.status_code == 200
    got = get_item.json()
    assert got["pattern_id"] == "prom:proposition:test1"
    assert got["source_class"] == "promoted_memory"
