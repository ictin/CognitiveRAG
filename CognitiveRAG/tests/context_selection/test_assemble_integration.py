import json
import os
import shutil

from CognitiveRAG.session_memory.context_window import WORKDIR, assemble_context, compact_session


def test_assemble_context_uses_selector_and_emits_explanation(tmp_path):
    session_id = "selector-integ"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)

    raw_path = os.path.join(WORKDIR, f"raw_{session_id}.json")
    raw = [{"index": i, "text": f"message {i}", "message_id": f"m{i}"} for i in range(25)]
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    compact_session(session_id, older_than_index=15)
    out = assemble_context(session_id, fresh_tail_count=4, budget=600, query="what do you remember?")

    assert "fresh_tail" in out and "summaries" in out
    assert "explanation" in out and isinstance(out["explanation"], dict)
    assert out["explanation"]["intent_family"] == "memory_summary"
    assert len(out["fresh_tail"]) >= 1
    assert "selected_blocks" in out
