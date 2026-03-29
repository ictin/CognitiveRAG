from CognitiveRAG.session_memory import ConversationStore, MessagePartsStore
from CognitiveRAG.session_memory.recall import search_session_memory, describe_session_item


def test_part_aware_recall_surfaces_tool_trace_and_provenance(tmp_path):
    conv = ConversationStore(str(tmp_path / "conversations.sqlite3"))
    parts = MessagePartsStore(str(tmp_path / "message_parts.sqlite3"))

    conv.append_message("sess-p", "m1", "assistant", "Running tool fetch", "2026-03-29T10:00:00Z")
    parts.upsert_structured_part(
        session_id="sess-p",
        message_id="m1",
        part_index=0,
        text="fetch(url=https://example.com)",
        part_type="tool_call",
        status="started",
        tool_name="fetch",
        tool_call_id="call-22",
    )
    parts.upsert_structured_part(
        session_id="sess-p",
        message_id="m1",
        part_index=1,
        text="HTTP 200",
        part_type="tool_result",
        status="success",
        tool_name="fetch",
        tool_call_id="call-22",
    )

    refs = search_session_memory("sess-p", "fetch", db_prefix=str(tmp_path))
    part_refs = [r for r in refs if r["item_type"] == "message_part"]
    assert part_refs, refs
    assert any((r.get("metadata") or {}).get("part_type") == "tool_call" for r in part_refs)

    desc = describe_session_item(part_refs[0], db_prefix=str(tmp_path))
    assert desc["item_type"] == "message_part"
    assert "provenance" in desc
    assert desc["provenance"]["session_id"] == "sess-p"
