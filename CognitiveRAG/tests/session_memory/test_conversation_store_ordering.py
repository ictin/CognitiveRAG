from CognitiveRAG.session_memory.conversation_store import ConversationStore


def test_conversation_store_preserves_append_order_for_same_timestamp(tmp_path):
    store = ConversationStore(str(tmp_path / "conversations.sqlite3"))
    ts = "2026-04-10T10:10:10Z"
    store.append_message("sess-o", "m1", "user", "first", ts)
    store.append_message("sess-o", "m2", "assistant", "second", ts)
    store.append_message("sess-o", "m3", "user", "third", ts)

    msgs = store.get_messages("sess-o")
    assert [m["message_id"] for m in msgs] == ["m1", "m2", "m3"]
    assert [m["text"] for m in msgs] == ["first", "second", "third"]
