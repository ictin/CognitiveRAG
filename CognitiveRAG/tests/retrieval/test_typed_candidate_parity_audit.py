import json
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.context_selection.candidate_builder import build_candidates_with_route
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.retrieval.router import clear_hot_cache, route_and_retrieve
from CognitiveRAG.crag.web_memory.evidence_store import WebEvidenceStore
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def _seed_reasoning(workdir: Path) -> None:
    db = sqlite3.connect(workdir / "reasoning.sqlite3")
    db.execute(
        "CREATE TABLE reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT)"
    )
    db.execute(
        "INSERT INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "p1",
            "sig",
            "[]",
            "Use retention loops when drafting copy.",
            0.88,
            "[]",
            "workflow",
            "retention loop drafting workflow",
            "warm",
        ),
    )
    db.commit()
    db.close()


def _seed_corpus(workdir: Path) -> None:
    db = sqlite3.connect(workdir / "context_items.sqlite3")
    db.execute("CREATE TABLE context_items (item_id TEXT, session_id TEXT, type TEXT, payload_json TEXT, created_at TEXT)")
    db.execute(
        "INSERT INTO context_items VALUES (?, ?, ?, ?, ?)",
        (
            "c1",
            "sess",
            "corpus_chunk",
            json.dumps({"summary": "Copywriting retention system with clear hook structure", "file_path": "books/copywriting.md"}),
            "2026-01-01",
        ),
    )
    db.commit()
    db.close()


def _seed_large_files(workdir: Path) -> None:
    db = sqlite3.connect(workdir / "large_files.sqlite3")
    db.execute("CREATE TABLE large_files (record_id TEXT, file_path TEXT, metadata_json TEXT, created_at TEXT)")
    db.execute(
        "INSERT INTO large_files VALUES (?, ?, ?, ?)",
        (
            "lf1",
            "books/long_copywriting_notes.md",
            json.dumps({"excerpt": "Long-form note on retention loops and hook cadence."}),
            "2026-01-02",
        ),
    )
    db.commit()
    db.close()


def _seed_web(workdir: Path) -> None:
    evidence_store = WebEvidenceStore(workdir / "web_evidence.sqlite3")
    evidence_store.upsert_evidence(
        {
            "query": "latest copywriting retention update",
            "query_variant": "latest copywriting retention update",
            "source_id": "https://example.com/source-a",
            "url": "https://example.com/source-a",
            "title": "Retention Update A",
            "snippet": "snippet a",
            "extracted_text": "Latest retention update A",
            "fetched_at": "2026-03-29T12:00:00Z",
            "published_at": None,
            "updated_at": None,
            "trust_score": 0.72,
            "freshness_class": "hot",
            "content_hash": "hash-a",
            "raw": {},
        }
    )
    evidence_store.upsert_evidence(
        {
            "query": "latest copywriting retention update",
            "query_variant": "latest copywriting retention update",
            "source_id": "https://example.com/source-b",
            "url": "https://example.com/source-b",
            "title": "Retention Update B",
            "snippet": "snippet b",
            "extracted_text": "Latest retention update B",
            "fetched_at": "2026-03-29T12:01:00Z",
            "published_at": None,
            "updated_at": None,
            "trust_score": 0.75,
            "freshness_class": "warm",
            "content_hash": "hash-b",
            "raw": {},
        }
    )

    promoted = WebPromotedMemoryStore(workdir / "web_promoted_memory.sqlite3")
    promoted.upsert_fact(
        promoted_id="wp1",
        canonical_fact="Latest copywriting retention update promoted fact.",
        evidence_ids=["we1"],
        confidence=0.8,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/source-a"},
        now_iso="2026-03-29T12:02:00Z",
    )


def _seed_fast_lane_promoted_tiers(workdir: Path) -> None:
    promoted = WebPromotedMemoryStore(workdir / "web_promoted_memory.sqlite3")
    promoted.upsert_fact(
        promoted_id="fast_local",
        canonical_fact="Retention loop local tier validated fact",
        evidence_ids=["we_local"],
        confidence=0.81,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/local"},
        now_iso="2026-03-29T12:03:00Z",
        promotion_tier=WebPromotedMemoryStore.TIER_LOCAL,
        origin_tier=WebPromotedMemoryStore.TIER_LOCAL,
    )
    promoted.upsert_fact(
        promoted_id="fast_workspace",
        canonical_fact="Retention loop workspace tier validated fact",
        evidence_ids=["we_workspace"],
        confidence=0.83,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/workspace"},
        now_iso="2026-03-29T12:03:10Z",
        promotion_tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        origin_tier=WebPromotedMemoryStore.TIER_LOCAL,
    )
    promoted.upsert_fact(
        promoted_id="fast_global",
        canonical_fact="Retention loop global tier validated fact",
        evidence_ids=["we_global"],
        confidence=0.85,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/global"},
        now_iso="2026-03-29T12:03:20Z",
        promotion_tier=WebPromotedMemoryStore.TIER_GLOBAL,
        origin_tier=WebPromotedMemoryStore.TIER_LOCAL,
    )


def _assert_typed_lane_hit(hit: LaneHit) -> None:
    assert isinstance(hit.id, str) and hit.id
    assert isinstance(hit.lane, RetrievalLane)
    assert isinstance(hit.memory_type, MemoryType)
    assert isinstance(hit.text, str) and hit.text
    assert isinstance(hit.provenance, dict)
    assert hit.tokens > 0

    assert isinstance(hit.lexical_score, float)
    assert isinstance(hit.semantic_score, float)
    assert isinstance(hit.recency_score, float)
    assert isinstance(hit.freshness_score, float)
    assert isinstance(hit.trust_score, float)
    assert isinstance(hit.novelty_score, float)
    assert isinstance(hit.contradiction_risk, float)
    assert isinstance(hit.must_include, bool)
    assert isinstance(hit.compressible, bool)


def test_route_and_retrieve_emits_typed_candidates_across_active_lanes(tmp_path: Path):
    clear_hot_cache()
    _seed_reasoning(tmp_path)
    _seed_corpus(tmp_path)
    _seed_large_files(tmp_path)
    _seed_web(tmp_path)

    plan, hits = route_and_retrieve(
        query="latest copywriting retention update",
        intent_family=IntentFamily.INVESTIGATIVE,
        session_id="sess",
        fresh_tail=[{"index": 2, "text": "fresh retention note", "message_id": "m2"}],
        older_raw=[{"index": 0, "text": "older copywriting memory", "message_id": "m0"}],
        summaries=[{"chunk_index": 0, "summary": "session summary on retention"}],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )

    assert plan.lanes == [
        RetrievalLane.SEMANTIC,
        RetrievalLane.LEXICAL,
        RetrievalLane.EPISODIC,
        RetrievalLane.PROMOTED,
        RetrievalLane.CORPUS,
        RetrievalLane.LARGE_FILE,
        RetrievalLane.WEB,
    ]
    assert hits

    for hit in hits:
        _assert_typed_lane_hit(hit)

    seen = {h.lane for h in hits}
    assert RetrievalLane.SEMANTIC in seen
    assert RetrievalLane.LEXICAL in seen
    # episodic lane emits fresh_tail/episodic/session_summary typed sub-lanes
    assert any(h.lane in {RetrievalLane.FRESH_TAIL, RetrievalLane.EPISODIC, RetrievalLane.SESSION_SUMMARY} for h in hits)
    assert RetrievalLane.PROMOTED in seen
    assert RetrievalLane.CORPUS in seen
    assert RetrievalLane.LARGE_FILE in seen
    assert RetrievalLane.WEB in seen

    web_types = {h.memory_type for h in hits if h.lane == RetrievalLane.WEB}
    assert MemoryType.WEB_EVIDENCE in web_types
    assert MemoryType.WEB_PROMOTED_FACT in web_types


def test_candidate_builder_preserves_lane_hit_fields_for_selector(tmp_path: Path):
    clear_hot_cache()
    _seed_reasoning(tmp_path)
    _seed_corpus(tmp_path)
    _seed_large_files(tmp_path)
    _seed_web(tmp_path)

    route_plan, lane_hits = route_and_retrieve(
        query="latest copywriting retention update",
        intent_family=IntentFamily.INVESTIGATIVE,
        session_id="sess",
        fresh_tail=[{"index": 2, "text": "fresh retention note", "message_id": "m2"}],
        older_raw=[{"index": 0, "text": "older copywriting memory", "message_id": "m0"}],
        summaries=[{"chunk_index": 0, "summary": "session summary on retention"}],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )
    assert route_plan.lanes
    by_id = {h.id: h for h in lane_hits}

    _, candidates = build_candidates_with_route(
        session_id="sess",
        query="latest copywriting retention update",
        fresh_tail=[{"index": 2, "text": "fresh retention note", "message_id": "m2"}],
        older_raw=[{"index": 0, "text": "older copywriting memory", "message_id": "m0"}],
        summaries=[{"chunk_index": 0, "summary": "session summary on retention"}],
        workdir=str(tmp_path),
        intent_family=IntentFamily.INVESTIGATIVE,
    )

    assert candidates
    for cand in candidates:
        hit = by_id.get(cand.id)
        assert hit is not None, f"missing lane hit for candidate {cand.id}"
        assert cand.lane == hit.lane
        assert cand.memory_type == hit.memory_type
        assert cand.text == hit.text
        assert cand.tokens == hit.tokens
        assert cand.provenance == hit.provenance
        assert cand.lexical_score == hit.lexical_score
        assert cand.semantic_score == hit.semantic_score
        assert cand.recency_score == hit.recency_score
        assert cand.freshness_score == hit.freshness_score
        assert cand.trust_score == hit.trust_score
        assert cand.novelty_score == hit.novelty_score
        assert cand.contradiction_risk == hit.contradiction_risk
        assert cand.cluster_id == hit.cluster_id
        assert cand.must_include == hit.must_include
        assert cand.compressible == hit.compressible


def test_route_and_retrieve_emits_typed_fast_lane_candidates(tmp_path: Path):
    clear_hot_cache()
    _seed_fast_lane_promoted_tiers(tmp_path)

    plan, hits = route_and_retrieve(
        query="retention loop",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        session_id="sess",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=8,
    )

    assert plan.lanes == [
        RetrievalLane.PROMOTED,
        RetrievalLane.EPISODIC,
        RetrievalLane.SEMANTIC,
    ]

    fast_hits = [h for h in hits if h.lane in {RetrievalLane.GLOBAL_PROMOTED, RetrievalLane.WORKSPACE_FAST, RetrievalLane.INSTALLATION_FAST}]
    assert fast_hits, "expected fast-lane typed hits for promoted tiers"
    assert {h.lane for h in fast_hits} == {
        RetrievalLane.GLOBAL_PROMOTED,
        RetrievalLane.WORKSPACE_FAST,
        RetrievalLane.INSTALLATION_FAST,
    }
    assert all(h.memory_type == MemoryType.WEB_PROMOTED_FACT for h in fast_hits)
    for hit in fast_hits:
        _assert_typed_lane_hit(hit)


def test_candidate_builder_preserves_fast_lane_fields_for_selector(tmp_path: Path):
    clear_hot_cache()
    _seed_fast_lane_promoted_tiers(tmp_path)

    _, lane_hits = route_and_retrieve(
        query="retention loop",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        session_id="sess",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=8,
    )
    clear_hot_cache()
    _, candidates = build_candidates_with_route(
        session_id="sess",
        query="retention loop",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        intent_family=IntentFamily.MEMORY_SUMMARY,
    )

    fast_lanes = {RetrievalLane.GLOBAL_PROMOTED, RetrievalLane.WORKSPACE_FAST, RetrievalLane.INSTALLATION_FAST}
    lane_hits_by_id = {h.id: h for h in lane_hits if h.lane in fast_lanes}
    candidate_by_id = {c.id: c for c in candidates if c.lane in fast_lanes}
    assert set(candidate_by_id) == set(lane_hits_by_id)
    for cid, hit in lane_hits_by_id.items():
        cand = candidate_by_id[cid]
        assert cand.lane == hit.lane
        assert cand.memory_type == hit.memory_type
        assert cand.text == hit.text
        assert cand.tokens == hit.tokens
        assert cand.provenance == hit.provenance
        assert cand.lexical_score == hit.lexical_score
        assert cand.semantic_score == hit.semantic_score
        assert cand.recency_score == hit.recency_score
        assert cand.freshness_score == hit.freshness_score
        assert cand.trust_score == hit.trust_score
        assert cand.novelty_score == hit.novelty_score
        assert cand.contradiction_risk == hit.contradiction_risk
        assert cand.cluster_id == hit.cluster_id
        assert cand.must_include == hit.must_include
        assert cand.compressible == hit.compressible
