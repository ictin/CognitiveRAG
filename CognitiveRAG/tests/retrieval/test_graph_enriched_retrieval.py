import json
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType
from CognitiveRAG.crag.graph_memory.relations import (
    record_problem_signature_resolved_by,
    record_reasoning_pattern_supported_by,
    record_web_promoted_derived_from,
)
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.crag.retrieval import web_lane
from CognitiveRAG.crag.retrieval.promoted_lane import retrieve as retrieve_promoted
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore
from CognitiveRAG.schemas.memory import ReasoningPattern


def _seed_reasoning_db(tmp_path: Path) -> None:
    db_path = tmp_path / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT)"
        )
        db.execute(
            "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "p_a",
                "debug timeout handling",
                "[]",
                "Apply bounded diagnosis and verify fallback.",
                0.7,
                json.dumps([{"source": "doc://timeout"}]),
                "workflow_pattern",
                "workflow:bounded diagnosis",
                "current",
            ),
        )
        db.execute(
            "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "p_b",
                "generic issue handling",
                "[]",
                "Apply bounded diagnosis and verify fallback.",
                0.7,
                json.dumps([{"source": "doc://generic"}]),
                "workflow_pattern",
                "workflow:bounded diagnosis",
                "current",
            ),
        )
        db.commit()


def _pattern(pattern_id: str, signature: str) -> ReasoningPattern:
    return ReasoningPattern(
        pattern_id=pattern_id,
        problem_signature=signature,
        reasoning_steps=["step"],
        solution_summary="Apply bounded diagnosis and verify fallback.",
        confidence=0.8,
        provenance=["doc://p"],
    )


def test_promoted_lane_surfaces_graph_support_links_and_problem_signature_reuse(tmp_path: Path):
    _seed_reasoning_db(tmp_path)
    store = GraphMemoryStore(tmp_path / "graph.sqlite3")
    record_reasoning_pattern_supported_by(
        store,
        pattern=_pattern("p_a", "debug timeout handling"),
        source={"source_url": "https://docs.example/debug-timeout", "source": "docs"},
        provenance={"test_case": "f1"},
    )
    record_problem_signature_resolved_by(
        store,
        problem_signature="debug timeout handling",
        pattern=_pattern("p_a", "debug timeout handling"),
        provenance={"test_case": "f3"},
    )

    hits = retrieve_promoted(
        workdir=str(tmp_path),
        intent_family=IntentFamily.INVESTIGATIVE,
        query="debug timeout handling",
        top_k=2,
    )
    assert [h.id for h in hits] == ["promoted:p_a", "promoted:p_b"]
    top = hits[0]
    assert top.provenance.get("graph_support_count") == 1
    assert top.provenance["graph_support_links"][0]["source_key"] == "https://docs.example/debug-timeout"
    assert top.provenance.get("graph_problem_signature_matches")
    assert top.provenance["graph_problem_signature_matches"][0]["problem_signature"] == "debug timeout handling"


def test_promoted_lane_graph_enrichment_fallback_without_graph_db(tmp_path: Path):
    _seed_reasoning_db(tmp_path)
    hits = retrieve_promoted(
        workdir=str(tmp_path),
        intent_family=IntentFamily.INVESTIGATIVE,
        query="debug timeout handling",
        top_k=2,
    )
    assert hits
    assert "graph_support_links" not in hits[0].provenance
    assert "graph_problem_signature_matches" not in hits[0].provenance


def test_web_lane_surfaces_graph_source_origin_for_promoted_web(tmp_path: Path, monkeypatch):
    promoted_store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    promoted_store.upsert_fact(
        promoted_id="wp_btc",
        canonical_fact="Bitcoin traded near 100k in latest update.",
        evidence_ids=["ev1"],
        confidence=0.8,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/btc"},
        now_iso="2026-04-01T00:00:00Z",
    )
    store = GraphMemoryStore(tmp_path / "graph.sqlite3")
    record_web_promoted_derived_from(
        store,
        promoted_id="wp_btc",
        source_url="https://example.com/btc",
        provenance={"test_case": "f2"},
    )

    monkeypatch.setattr(web_lane.WebFetcher, "fetch_plan", lambda self, plan, need, min_cache_hits=2: [])
    hits = web_lane.retrieve(
        workdir=str(tmp_path),
        query="Bitcoin traded",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=3,
    )
    promoted_hits = [h for h in hits if h.memory_type == MemoryType.WEB_PROMOTED_FACT]
    assert promoted_hits
    top = promoted_hits[0]
    assert top.provenance.get("graph_source_origin_count") == 1
    assert top.provenance["graph_source_origins"][0]["source_url"] == "https://example.com/btc"
    assert top.provenance.get("source_url") == "https://example.com/btc"


def test_web_lane_graph_origin_fallback_without_graph_db(tmp_path: Path, monkeypatch):
    promoted_store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    promoted_store.upsert_fact(
        promoted_id="wp_eth",
        canonical_fact="Ethereum volatility increased this week.",
        evidence_ids=["ev2"],
        confidence=0.6,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/eth"},
        now_iso="2026-04-01T00:00:00Z",
    )
    monkeypatch.setattr(web_lane.WebFetcher, "fetch_plan", lambda self, plan, need, min_cache_hits=2: [])
    hits = web_lane.retrieve(
        workdir=str(tmp_path),
        query="Ethereum volatility",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=3,
    )
    promoted_hits = [h for h in hits if h.memory_type == MemoryType.WEB_PROMOTED_FACT]
    assert promoted_hits
    assert "graph_source_origins" not in promoted_hits[0].provenance
    assert promoted_hits[0].provenance.get("promoted_id") == "wp_eth"
