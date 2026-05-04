from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.router import clear_routing_caches, route_and_retrieve


def _seed_reasoning_db(tmp_path: Path) -> None:
    db_path = tmp_path / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT)"
        )
        db.execute(
            "INSERT OR REPLACE INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "p_topic",
                "postgres migration rollback timeout",
                "[]",
                "Use staged rollback with timeout fallback checks.",
                0.81,
                json.dumps([{"source": "doc://topic"}]),
                "workflow_pattern",
                "postgres migration rollback timeout",
                "current",
            ),
        )
        db.commit()


def _seed_corpus_db(tmp_path: Path) -> None:
    db_path = tmp_path / "context_items.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS context_items (item_id TEXT, session_id TEXT, type TEXT, payload_json TEXT, created_at TEXT)"
        )
        payload = {
            "summary": "Postgres migration rollback checklist and timeout fallback strategy.",
            "file_path": "books/postgres_ops.md",
        }
        db.execute(
            "INSERT INTO context_items VALUES (?, ?, ?, ?, ?)",
            ("topic-corpus-1", "topic-session", "corpus_chunk", json.dumps(payload), "2026-05-04T12:00:00Z"),
        )
        db.commit()


def test_topic_graph_helper_is_visible_but_selector_remains_authoritative(tmp_path: Path, monkeypatch):
    _seed_reasoning_db(tmp_path)
    _seed_corpus_db(tmp_path)
    clear_routing_caches()
    monkeypatch.delenv("CRAG_DISABLE_TOPIC_GRAPH", raising=False)
    monkeypatch.delenv("CRAG_DISABLE_CATEGORY_GRAPH", raising=False)

    plan_enabled, hits_enabled = route_and_retrieve(
        query="postgres migration rollback timeout",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="topic-helper-enabled",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )

    clear_routing_caches()
    monkeypatch.setenv("CRAG_DISABLE_TOPIC_GRAPH", "1")
    plan_disabled, hits_disabled = route_and_retrieve(
        query="postgres migration rollback timeout",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="topic-helper-disabled",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )

    assert [l.value for l in plan_enabled.lanes] == [l.value for l in plan_disabled.lanes]
    topic_meta_enabled = dict((plan_enabled.metadata or {}).get("topic_graph_bridge") or {})
    topic_meta_disabled = dict((plan_disabled.metadata or {}).get("topic_graph_bridge") or {})
    assert topic_meta_enabled.get("helper_enabled") is True
    assert isinstance(topic_meta_enabled.get("hinted_topics"), list)
    assert topic_meta_disabled.get("helper_enabled") is False

    assert any((h.provenance or {}).get("topic_graph", {}).get("topic_count", 0) >= 1 for h in hits_enabled)
    assert all((h.provenance or {}).get("topic_graph") in (None, {}) for h in hits_disabled)
