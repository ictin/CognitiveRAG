from CognitiveRAG.session_memory.message_parts_store import MessagePartsStore


def test_failed_call_and_retry_remain_distinct(tmp_path):
    db = tmp_path / "message_parts.sqlite3"
    store = MessagePartsStore(str(db))

    store.upsert_structured_part(
        session_id="sess-r",
        message_id="msg-tool",
        part_index=1,
        text="Tool failed: timeout",
        part_type="tool_result",
        status="failed",
        tool_name="fetch",
        tool_call_id="call-1",
    )
    store.upsert_structured_part(
        session_id="sess-r",
        message_id="msg-tool",
        part_index=2,
        text="Retry success",
        part_type="tool_result",
        status="success",
        retry_of_part_index=1,
        tool_name="fetch",
        tool_call_id="call-1",
    )

    parts = store.get_parts("sess-r", "msg-tool")
    assert len(parts) == 2
    failed = next(p for p in parts if p["part_index"] == 1)
    retried = next(p for p in parts if p["part_index"] == 2)
    assert failed["status"] == "failed"
    assert retried["status"] == "success"
    assert retried["retry_of_part_index"] == 1
