import json
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.promoted_lane import retrieve


def test_promoted_retrieval_surfaces_normalized_units(tmp_path: Path, monkeypatch):
    from CognitiveRAG.core.settings import settings

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
    monkeypatch.setattr(settings.store, "reasoning_db_path", str(db_path))
    hits = retrieve(workdir=str(tmp_path), intent_family=IntentFamily.MEMORY_SUMMARY, query="what do you know about me", top_k=3)
    assert hits, "promoted lane should return hits"
    top = hits[0]
    assert top.lane.value == "promoted"
    assert "profile_preference" in (top.text or "")
    assert top.provenance.get("memory_subtype") == "profile_preference"
    assert top.provenance.get("normalized_text")
    assert top.provenance.get("source_class") == "promoted_memory"
    lifecycle = dict(top.provenance.get("lifecycle") or {})
    assert lifecycle.get("lifecycle_state") in {"approved", "revalidated", "stale"}
    assert lifecycle.get("approval_state") == "approved"


def test_promoted_retrieval_prefers_settings_reasoning_db(tmp_path: Path, monkeypatch):
    from CognitiveRAG.core.settings import settings

    db_path = tmp_path / "settings_reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT)"
        )
        db.execute(
            "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "prom:proposition:settings-db",
                "proposition:s2",
                "[]",
                "deployment window is tuesday 14:00 utc",
                0.9,
                json.dumps([{"source": "settings_db"}]),
                "stable_fact",
                "proposition:deployment window is tuesday 14:00 utc",
                "warm",
            ),
        )
        db.commit()

    monkeypatch.setattr(settings.store, "reasoning_db_path", str(db_path))
    hits = retrieve(workdir=str(tmp_path / "empty"), intent_family=IntentFamily.MEMORY_SUMMARY, query="deployment window", top_k=3)
    assert hits
    assert any("deployment window" in (h.text or "").lower() for h in hits)
