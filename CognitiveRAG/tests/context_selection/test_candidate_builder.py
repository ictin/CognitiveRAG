import json
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.context_selection.candidate_builder import build_candidates


def test_candidate_builder_normalizes_sources_and_preserves_provenance(tmp_path: Path):
    workdir = tmp_path

    # context_items corpus source
    cdb = sqlite3.connect(workdir / "context_items.sqlite3")
    cdb.execute("CREATE TABLE context_items (item_id TEXT, session_id TEXT, type TEXT, payload_json TEXT, created_at TEXT)")
    cdb.execute(
        "INSERT INTO context_items VALUES (?, ?, ?, ?, ?)",
        ("i1", "sess", "corpus_chunk", json.dumps({"text": "youtube secrets synopsis text", "file_path": "book.md"}), "2026-01-01"),
    )
    cdb.commit()
    cdb.close()

    fresh_tail = [{"index": 3, "text": "recent tail"}]
    older_raw = [{"index": 1, "text": "older message"}]
    summaries = [{"chunk_index": 0, "summary": "session summary"}]

    out = build_candidates(
        session_id="sess",
        query="youtube secrets",
        fresh_tail=fresh_tail,
        older_raw=older_raw,
        summaries=summaries,
        workdir=str(workdir),
        intent_family=IntentFamily.CORPUS_OVERVIEW,
    )

    ids = {c.id for c in out}
    assert any(i.startswith("fresh:") for i in ids)
    assert any(i.startswith("summary:") for i in ids)
    corpus = [c for c in out if c.lane.value == "corpus"]
    assert corpus, "Expected corpus candidates from context_items"
    assert "payload" in corpus[0].provenance
    assert corpus[0].tokens > 0
