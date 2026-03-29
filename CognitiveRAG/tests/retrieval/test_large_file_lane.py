import json
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.retrieval.large_file_lane import retrieve


def test_large_file_lane_returns_excerpt_hits(tmp_path: Path):
    db = sqlite3.connect(tmp_path / "large_files.sqlite3")
    db.execute("CREATE TABLE large_files (record_id TEXT, file_path TEXT, metadata_json TEXT, created_at TEXT)")
    db.execute(
        "INSERT INTO large_files VALUES (?, ?, ?, ?)",
        ("lf1", "/mnt/g/@Cursuri/book1.md", json.dumps({"excerpt": "The synopsis explains audience growth loops."}), "2026-01-01"),
    )
    db.commit()
    db.close()

    hits = retrieve(workdir=str(tmp_path), query="synopsis audience growth", top_k=5)
    assert hits
    assert hits[0].lane.value == "large_file"
    assert "file_path" in hits[0].provenance
