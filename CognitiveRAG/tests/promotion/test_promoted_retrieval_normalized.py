import json
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.promoted_lane import retrieve


def test_promoted_retrieval_surfaces_normalized_units(tmp_path: Path):
    db_path = tmp_path / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT)"
        )
        db.execute(
            "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "prom:proposition:abc",
                "proposition:s1",
                "[]",
                "user prefers concise technical answers",
                0.9,
                json.dumps([{"source": "test"}]),
                "profile_preference",
                "proposition:user prefers concise technical answers",
                "current",
            ),
        )
        db.commit()
    hits = retrieve(workdir=str(tmp_path), intent_family=IntentFamily.MEMORY_SUMMARY, query="what do you know about me", top_k=3)
    assert hits, "promoted lane should return hits"
    top = hits[0]
    assert top.lane.value == "promoted"
    assert "profile_preference" in (top.text or "")
    assert top.provenance.get("memory_subtype") == "profile_preference"
    assert top.provenance.get("normalized_text")

