from __future__ import annotations

from fastapi.testclient import TestClient

from CognitiveRAG import main_server


client = TestClient(main_server.app)


def test_session_structured_export_and_recall_endpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COGNITIVERAG_SKIP_KB", "1")

    session_id = "phase3-readback-s1"
    message_id = "m1"
    tool_call_id = "tc-42"

    append_msg = {
        "session_id": session_id,
        "message_id": message_id,
        "sender": "assistant",
        "text": "Tool trace seed for runtime readback.",
        "created_at": "2026-04-12T09:45:00Z",
    }
    r = client.post("/session_append_message", json=append_msg)
    assert r.status_code == 200
    assert r.json()["status"] in ("inserted", "updated")

    part0 = {
        "session_id": session_id,
        "message_id": message_id,
        "part_index": 0,
        "text": "tool call started",
        "meta_json": {
            "part_type": "tool_call",
            "status": "started",
            "tool_name": "memory_search",
            "tool_call_id": tool_call_id,
        },
    }
    part1 = {
        "session_id": session_id,
        "message_id": message_id,
        "part_index": 1,
        "text": "tool call finished",
        "meta_json": {
            "part_type": "tool_result",
            "status": "succeeded",
            "tool_name": "memory_search",
            "tool_call_id": tool_call_id,
            "retry_of_part_index": 0,
        },
    }
    for payload in (part0, part1):
        pr = client.post("/session_append_message_part", json=payload)
        assert pr.status_code == 200
        assert pr.json()["status"] in ("inserted", "updated")

    ex = client.post("/session_structured_export", json={"session_id": session_id})
    assert ex.status_code == 200
    body = ex.json()
    assert body["session_id"] == session_id
    assert body["structured_parts"] is True
    assert body["part_stats"]["message_count"] == 1
    assert body["part_stats"]["part_count"] == 2
    parts = body["messages"][0]["parts"]
    assert parts[0]["tool_call_id"] == tool_call_id
    assert parts[0]["tool_name"] == "memory_search"
    assert parts[1]["retry_of_part_index"] == 0
    assert "compaction" in body
    assert "segments" in body["compaction"]

    recall = client.post(
        "/session_recall",
        json={"session_id": session_id, "query": tool_call_id, "top_k": 5},
    )
    assert recall.status_code == 200
    refs = recall.json()["results"]
    assert refs
    mp_refs = [r for r in refs if r.get("item_type") == "message_part"]
    assert mp_refs

    desc = client.post("/session_describe_item", json={"ref": mp_refs[0]})
    assert desc.status_code == 200
    desc_body = desc.json()
    assert desc_body["item_type"] == "message_part"
    assert desc_body["tool_call_id"] == tool_call_id
    assert desc_body["tool_name"] == "memory_search"

    exp = client.post("/session_expand_item", json={"ref": mp_refs[0]})
    assert exp.status_code == 200
    expanded = exp.json()["expanded"]
    assert any(item.get("item_type") == "message_part" for item in expanded)
