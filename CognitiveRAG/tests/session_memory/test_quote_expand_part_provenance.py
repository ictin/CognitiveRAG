from CognitiveRAG.session_memory import ConversationStore, MessagePartsStore
from CognitiveRAG.session_memory.recall import expand_session_item


def test_expand_message_part_keeps_retry_chain_provenance(tmp_path):
    conv = ConversationStore(str(tmp_path / "conversations.sqlite3"))
    parts = MessagePartsStore(str(tmp_path / "message_parts.sqlite3"))

    conv.append_message("sess-q", "m7", "assistant", "tool trace", "2026-03-29T12:00:00Z")
    parts.upsert_structured_part(
        session_id="sess-q",
        message_id="m7",
        part_index=0,
        text="run tool",
        part_type="tool_call",
        status="started",
        tool_name="fetch",
        tool_call_id="call-77",
    )
    parts.upsert_structured_part(
        session_id="sess-q",
        message_id="m7",
        part_index=1,
        text="timeout",
        part_type="tool_result",
        status="failed",
        tool_name="fetch",
        tool_call_id="call-77",
    )
    parts.upsert_structured_part(
        session_id="sess-q",
        message_id="m7",
        part_index=2,
        text="ok after retry",
        part_type="tool_result",
        status="success",
        retry_of_part_index=1,
        tool_name="fetch",
        tool_call_id="call-77",
    )

    expanded = expand_session_item(
        {"item_type": "message_part", "session_id": "sess-q", "primary_id": "m7", "secondary_id": "1"},
        db_prefix=str(tmp_path),
    )
    related_indexes = sorted(int(x["secondary_id"]) for x in expanded)
    assert related_indexes == [0, 2]
    for item in expanded:
        prov = (item.get("metadata") or {}).get("provenance") or {}
        assert prov["session_id"] == "sess-q"
        assert prov["message_id"] == "m7"
