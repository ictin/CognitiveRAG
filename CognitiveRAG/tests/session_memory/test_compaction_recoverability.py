import json
import os
import shutil
from pathlib import Path

from CognitiveRAG.session_memory.compaction import SessionCompactionStore, summarize_compaction_state
from CognitiveRAG.session_memory.context_window import WORKDIR, assemble_context, compact_session
from CognitiveRAG.session_memory.recall import describe_session_item, expand_session_item, search_session_memory


def _write_raw(session_id: str, rows: list[dict]) -> None:
    os.makedirs(WORKDIR, exist_ok=True)
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def _seed_messages(session_id: str, count: int = 30) -> list[dict]:
    rows: list[dict] = []
    for i in range(count):
        rows.append(
            {
                "message_id": f"m{i}",
                "index": i,
                "sender": "assistant" if i % 3 == 0 else "user",
                "text": f"important payload message {i} for compaction and recall",
                "created_at": f"2026-04-03T10:{i:02d}:00Z",
            }
        )
    return rows


def test_compaction_policy_creates_lineage_and_preserves_raw(tmp_path: Path):
    session_id = "cmp1"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    rows = _seed_messages(session_id, count=28)
    _write_raw(session_id, rows)

    created = compact_session(session_id, older_than_index=20)
    assert created

    # raw truth must remain unchanged
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded == rows

    store = SessionCompactionStore(os.path.join(WORKDIR, "compaction.sqlite3"))
    state = summarize_compaction_state(session_id, store)
    assert int(state["stats"]["compacted_segments"]) >= 1
    first = store.list_segments(session_id)[0]
    assert first["lineage"]
    assert first["raw_snapshot"]
    assert first["status"] == "compacted"


def test_compaction_is_deterministic_and_idempotent(tmp_path: Path):
    session_id = "cmp2"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    _write_raw(session_id, _seed_messages(session_id, count=26))

    first = compact_session(session_id, older_than_index=18)
    second = compact_session(session_id, older_than_index=18)
    assert first
    assert second == []

    store = SessionCompactionStore(os.path.join(WORKDIR, "compaction.sqlite3"))
    segs = store.list_segments(session_id)
    assert len({s["segment_id"] for s in segs}) == len(segs)


def test_quarantine_is_explicit_and_safe(tmp_path: Path):
    session_id = "cmp3"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    rows = _seed_messages(session_id, count=18)
    rows[1]["text"] = "ok"  # low-value message
    rows[2]["text"] = ""  # low-value message
    _write_raw(session_id, rows)

    compact_session(session_id, older_than_index=10)
    store = SessionCompactionStore(os.path.join(WORKDIR, "compaction.sqlite3"))
    quarantined = store.list_quarantined(session_id)
    assert quarantined
    reasons = {q["reason"] for q in quarantined}
    assert "low_value_quarantine" in reasons


def test_compacted_summary_search_describe_expand_recovery(tmp_path: Path):
    session_id = "cmp4"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    rows = _seed_messages(session_id, count=22)
    # specific searchable text
    rows[5]["text"] = "deterministic compact recall anchor phrase"
    _write_raw(session_id, rows)
    compact_session(session_id, older_than_index=15)

    refs = search_session_memory(session_id, "anchor phrase", db_prefix=os.path.join(os.getcwd(), "data", "session_memory"), top_k=10)
    compact_refs = [r for r in refs if r["item_type"] == "compacted_summary"]
    assert compact_refs

    desc = describe_session_item(compact_refs[0], db_prefix=os.path.join(os.getcwd(), "data", "session_memory"))
    assert desc["item_type"] == "compacted_summary"
    assert desc["recoverability"] == "raw_or_snapshot"

    expanded = expand_session_item(compact_refs[0], db_prefix=os.path.join(os.getcwd(), "data", "session_memory"))
    assert expanded
    assert all(row["item_type"] == "message" for row in expanded)
    assert any("from_compaction_segment" in (row.get("metadata") or {}) for row in expanded)


def test_expand_compacted_summary_falls_back_to_snapshot_when_raw_missing(tmp_path: Path):
    session_id = "cmp5"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    rows = _seed_messages(session_id, count=20)
    rows[4]["text"] = "snapshot fallback proof phrase"
    _write_raw(session_id, rows)
    compact_session(session_id, older_than_index=12)

    refs = search_session_memory(session_id, "fallback proof phrase", db_prefix=os.path.join(os.getcwd(), "data", "session_memory"), top_k=10)
    compact_ref = next(r for r in refs if r["item_type"] == "compacted_summary")

    # Remove raw file to force snapshot fallback during expansion.
    os.remove(os.path.join(WORKDIR, f"raw_{session_id}.json"))
    expanded = expand_session_item(compact_ref, db_prefix=os.path.join(os.getcwd(), "data", "session_memory"))
    assert expanded
    assert any((row.get("metadata") or {}).get("recovered_from") == "compaction_snapshot" for row in expanded)


def test_assemble_context_surfaces_compaction_metadata_and_exact_recall_survives(tmp_path: Path):
    session_id = "cmp6"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    rows = _seed_messages(session_id, count=24)
    rows[2]["text"] = "exact recall survives compaction marker"
    _write_raw(session_id, rows)
    compact_session(session_id, older_than_index=16)

    assembled = assemble_context(
        session_id=session_id,
        fresh_tail_count=6,
        budget=1200,
        query="what did we say earlier about marker",
    )
    assert "compaction" in assembled
    assert "recoverability" in assembled
    refs = search_session_memory(session_id, "marker", db_prefix=os.path.join(os.getcwd(), "data", "session_memory"), top_k=10)
    assert any(r["item_type"] in {"message", "compacted_summary"} for r in refs)
