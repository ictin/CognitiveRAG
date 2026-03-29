from CognitiveRAG.session_memory import ConversationStore, MessagePartsStore
from CognitiveRAG.session_memory.export import export_session_with_parts


def test_structured_session_export_preserves_parts(tmp_path):
    conv = ConversationStore(str(tmp_path / "conversations.sqlite3"))
    parts = MessagePartsStore(str(tmp_path / "message_parts.sqlite3"))

    conv.append_message("sess-e", "m1", "user", "run checks", "2026-03-29T11:00:00Z")
    conv.append_message("sess-e", "m2", "assistant", "running now", "2026-03-29T11:00:01Z")
    parts.upsert_structured_part(
        session_id="sess-e",
        message_id="m2",
        part_index=0,
        text="pytest -q",
        part_type="tool_call",
        tool_name="shell",
        tool_call_id="tool-1",
    )
    parts.upsert_structured_part(
        session_id="sess-e",
        message_id="m2",
        part_index=1,
        text="passed",
        part_type="tool_result",
        status="success",
        tool_name="shell",
        tool_call_id="tool-1",
        file_refs=[{"path": "/tmp/report.txt"}],
    )

    exported = export_session_with_parts("sess-e", db_prefix=str(tmp_path))
    assert exported["structured_parts"] is True
    assert exported["part_stats"]["message_count"] == 2
    assert exported["part_stats"]["part_count"] == 2

    assistant = next(m for m in exported["messages"] if m["message_id"] == "m2")
    assert len(assistant["parts"]) == 2
    assert assistant["parts"][0]["part_type"] == "tool_call"
    assert assistant["parts"][1]["part_type"] == "tool_result"
