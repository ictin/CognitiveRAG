import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.promoted_lane import retrieve


def test_promoted_lane_reads_reasoning_store(tmp_path: Path):
    db = sqlite3.connect(tmp_path / "reasoning.sqlite3")
    db.execute(
        "CREATE TABLE reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, solution_summary TEXT, confidence REAL, provenance_json TEXT)"
    )
    db.execute(
        "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?)",
        ("p1", "sig", "[]", "user prefers concise summaries", 0.9, "[]"),
    )
    db.commit()
    db.close()

    hits = retrieve(workdir=str(tmp_path), intent_family=IntentFamily.MEMORY_SUMMARY, query="what do you know about me", top_k=5)
    assert hits
    assert hits[0].lane.value == "promoted"
    assert hits[0].trust_score > 0
