from CognitiveRAG.llm.sanitizer import sanitize_messages, sanitize_text


METADATA_RICH = """Sender (untrusted metadata):
Gemma 4
{
  "chat_id": "telegram:7694802153",
  "message_id": "5487",
  "sender_id": "7694802153",
  "sender": "Gogu",
  "timestamp": "2026-05-09T10:00:00Z"
}

Tell me a short story
"""


def test_sanitize_text_removes_metadata_block_but_keeps_user_text():
    cleaned = sanitize_text(METADATA_RICH)
    assert "Tell me a short story" in cleaned
    assert "sender_id" not in cleaned
    assert "message_id" not in cleaned
    assert "chat_id" not in cleaned
    assert "timestamp" not in cleaned
    # "Gemma 4" was metadata label in this block and must be removed with the block.
    assert "Gemma 4" not in cleaned


def test_sanitize_text_preserves_legitimate_technical_terms():
    text = "Explain ID tokens, create a label, what is a timestamp, compare Gemma 4 and gpt-5-mini."
    cleaned = sanitize_text(text)
    assert "ID tokens" in cleaned
    assert "label" in cleaned
    assert "timestamp" in cleaned
    assert "Gemma 4" in cleaned


def test_sanitize_messages_handles_list_content():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [{"type": "text", "text": METADATA_RICH}],
        },
    ]
    cleaned = sanitize_messages(messages)
    assert len(cleaned) == 2
    assert cleaned[1]["role"] == "user"
    assert "Tell me a short story" in cleaned[1]["content"]
    assert "sender_id" not in cleaned[1]["content"]
