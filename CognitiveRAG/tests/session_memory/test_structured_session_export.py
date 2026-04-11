from CognitiveRAG.session_memory import ConversationStore, MessagePartsStore
from CognitiveRAG.session_memory.compaction import SessionCompactionStore
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
    assert exported["compaction"]["segment_count"] == 0
    assert exported["compaction"]["quarantined_count"] == 0


def test_structured_session_export_includes_compaction_lineage_and_snapshot(tmp_path):
    conv = ConversationStore(str(tmp_path / "conversations.sqlite3"))
    parts = MessagePartsStore(str(tmp_path / "message_parts.sqlite3"))
    compaction = SessionCompactionStore(str(tmp_path / "compaction.sqlite3"))

    conv.append_message("sess-c", "m1", "user", "alpha", "2026-03-29T11:00:00Z")
    conv.append_message("sess-c", "m2", "assistant", "beta", "2026-03-29T11:00:01Z")
    parts.upsert_structured_part(
        session_id="sess-c",
        message_id="m2",
        part_index=0,
        text="tool ok",
        part_type="tool_result",
        status="success",
    )
    compaction.upsert_segment(
        session_id="sess-c",
        segment_id="compact:test",
        chunk_index=0,
        start_index=0,
        end_index=1,
        summary="summary text",
        source_count=2,
        policy_reason="age_based_chunk_compaction",
        status="compacted",
        lineage=[
            {"message_key": "message_id:m1", "message_id": "m1", "index": 0},
            {"message_key": "message_id:m2", "message_id": "m2", "index": 1},
        ],
        raw_snapshot=[
            {"message_id": "m1", "index": 0, "sender": "user", "text": "alpha"},
            {"message_id": "m2", "index": 1, "sender": "assistant", "text": "beta"},
        ],
        metadata={"created_by": "test"},
    )
    compaction.upsert_quarantined(
        session_id="sess-c",
        message_key="message_id:m0",
        msg_index=-1,
        reason="low_value_quarantine",
        metadata={"source": "test"},
    )

    exported = export_session_with_parts("sess-c", db_prefix=str(tmp_path))
    comp = exported["compaction"]
    assert comp["segment_count"] == 1
    assert comp["quarantined_count"] == 1
    assert len(comp["segments"]) == 1
    seg = comp["segments"][0]
    assert seg["segment_id"] == "compact:test"
    assert seg["lineage"][0]["message_id"] == "m1"
    assert seg["raw_snapshot"][1]["message_id"] == "m2"
    assert seg["metadata"]["created_by"] == "test"
