from CognitiveRAG.session_memory.message_parts_store import MessagePartsStore


def test_message_parts_store_persists_structured_fields(tmp_path):
    db = tmp_path / "message_parts.sqlite3"
    store = MessagePartsStore(str(db))

    store.upsert_structured_part(
        session_id="sess-a",
        message_id="msg-1",
        part_index=0,
        text="tool call payload",
        part_type="tool_call",
        status="started",
        tool_name="grep",
        tool_call_id="call-1",
        file_refs=[{"path": "/tmp/a.txt"}],
        meta_json={"phase": "exec"},
    )

    parts = store.get_parts("sess-a", "msg-1")
    assert len(parts) == 1
    p = parts[0]
    assert p["part_type"] == "tool_call"
    assert p["status"] == "started"
    assert p["tool_name"] == "grep"
    assert p["tool_call_id"] == "call-1"
    assert p["file_refs"] == [{"path": "/tmp/a.txt"}]
    assert p["meta"]["phase"] == "exec"
