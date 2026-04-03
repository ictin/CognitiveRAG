import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType
from CognitiveRAG.crag.retrieval import web_lane
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def test_staged_and_trusted_states_are_stored_and_read_distinctly(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.stage_fact(
        promoted_id="wp_stage",
        canonical_fact="Staged proposition.",
        evidence_ids=["ev1"],
        confidence=0.65,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/stage"},
        now_iso="2026-04-03T10:00:00Z",
    )
    store.upsert_fact(
        promoted_id="wp_trusted",
        canonical_fact="Trusted proposition.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.85,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/trusted"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        approval_reason="seeded_trusted_test",
        approval_basis={"test": True},
        now_iso="2026-04-03T10:00:00Z",
    )

    staged = store.get("wp_stage")
    trusted = store.get("wp_trusted")
    assert staged is not None and trusted is not None
    assert staged["promotion_state"] == "staged"
    assert trusted["promotion_state"] == "trusted"
    assert staged["approval_reason"] == "staged_pending_approval"
    assert trusted["approval_reason"] == "seeded_trusted_test"


def test_promotion_transition_is_deterministic_and_guarded(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.stage_fact(
        promoted_id="wp_guarded",
        canonical_fact="Needs more proof.",
        evidence_ids=["ev1"],
        confidence=0.9,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/guarded"},
        now_iso="2026-04-03T10:00:00Z",
    )
    first = store.promote_if_eligible("wp_guarded", now_iso="2026-04-03T10:05:00Z")
    second = store.promote_if_eligible("wp_guarded", now_iso="2026-04-03T10:06:00Z")
    assert first is not None and second is not None
    assert first["promotion_state"] == "staged"
    assert second["promotion_state"] == "staged"
    assert first["approval_reason"] == second["approval_reason"] == "insufficient_evidence_or_confidence"

    store.stage_fact(
        promoted_id="wp_good",
        canonical_fact="Enough proof now.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.8,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/good"},
        now_iso="2026-04-03T10:00:00Z",
    )
    promoted = store.promote_if_eligible("wp_good", now_iso="2026-04-03T10:05:00Z")
    assert promoted is not None
    assert promoted["promotion_state"] == "trusted"
    assert promoted["approval_reason"] == "evidence_count_confidence_source_threshold"
    assert promoted["approved_at"] == "2026-04-03T10:05:00Z"


def test_web_lane_surfaces_state_and_prefers_trusted_modestly(tmp_path: Path, monkeypatch):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.stage_fact(
        promoted_id="wp_stage",
        canonical_fact="Bitcoin moved sharply on macro headlines.",
        evidence_ids=["ev1"],
        confidence=0.75,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/staged"},
        now_iso="2026-04-03T10:00:00Z",
    )
    store.upsert_fact(
        promoted_id="wp_trusted",
        canonical_fact="Bitcoin moved sharply on macro headlines.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.75,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/trusted"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        approval_reason="seeded_trusted_test",
        approval_basis={"test": True},
        now_iso="2026-04-03T10:00:00Z",
    )
    monkeypatch.setattr(web_lane.WebFetcher, "fetch_plan", lambda self, plan, need, min_cache_hits=2: [])

    hits = web_lane.retrieve(
        workdir=str(tmp_path),
        query="Bitcoin moved",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=4,
    )
    promoted_hits = [h for h in hits if h.memory_type == MemoryType.WEB_PROMOTED_FACT]
    assert promoted_hits and len(promoted_hits) >= 2
    assert promoted_hits[0].provenance.get("promotion_state") == "trusted"
    assert promoted_hits[1].provenance.get("promotion_state") == "staged"
    delta = promoted_hits[0].semantic_score - promoted_hits[1].semantic_score
    assert 0.0 <= delta <= 0.06


def test_old_records_without_state_fields_migrate_safely(tmp_path: Path):
    db_path = tmp_path / "web_promoted_memory.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE web_promoted_memory (
                promoted_id TEXT PRIMARY KEY,
                canonical_fact TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                freshness_state TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO web_promoted_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "wp_legacy",
                "Legacy promoted fact.",
                "[\"ev1\"]",
                0.7,
                "warm",
                "{\"source_url\":\"https://example.com/legacy\"}",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.commit()

    store = WebPromotedMemoryStore(db_path)
    records = store.search("Legacy", top_k=3)
    assert records
    assert records[0]["promoted_id"] == "wp_legacy"
    # Migration default keeps old trusted behavior unless explicitly staged.
    assert records[0]["promotion_state"] == "trusted"

