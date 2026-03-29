import json
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.retrieval.corpus_lane import retrieve


def test_corpus_lane_reads_corpus_items_and_preserves_provenance(tmp_path: Path):
    db = sqlite3.connect(tmp_path / "context_items.sqlite3")
    db.execute("CREATE TABLE context_items (item_id TEXT, session_id TEXT, type TEXT, payload_json TEXT, created_at TEXT)")
    payload = {"summary": "YouTube Secrets discusses channel positioning", "file_path": "books/youtube_secrets.md"}
    db.execute("INSERT INTO context_items VALUES (?, ?, ?, ?, ?)", ("c1", "sess", "corpus_chunk", json.dumps(payload), "2026-01-01"))
    db.commit()
    db.close()

    hits = retrieve(workdir=str(tmp_path), query="youtube secrets", top_k=5)
    assert hits
    assert hits[0].lane.value == "corpus"
    assert hits[0].provenance["payload"]["file_path"].endswith("youtube_secrets.md")
