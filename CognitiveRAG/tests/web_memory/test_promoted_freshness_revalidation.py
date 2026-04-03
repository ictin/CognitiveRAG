import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType
from CognitiveRAG.crag.retrieval import web_lane
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def test_trusted_item_stays_fresh_within_ttl_window(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_fresh",
        canonical_fact="Fresh trusted fact.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.8,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/fresh", "freshness_ttl_hours": 72},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        now_iso="2026-04-03T10:00:00Z",
    )
    record = store.evaluate_freshness("wp_fresh", now_iso="2026-04-05T09:00:00Z")
    assert record is not None
    assert record["freshness_lifecycle_state"] == "fresh"
    assert record["freshness_reason"] == "within_ttl_window"


def test_trusted_item_becomes_stale_after_ttl_and_can_request_revalidation(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_stale",
        canonical_fact="Aging trusted fact.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.8,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/stale", "freshness_ttl_hours": 24},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        now_iso="2026-04-03T10:00:00Z",
    )
    stale = store.evaluate_freshness("wp_stale", now_iso="2026-04-05T10:00:00Z")
    assert stale is not None
    assert stale["freshness_lifecycle_state"] == "stale"
    assert "ttl_expired" in stale["freshness_reason"]

    pending = store.request_revalidation(
        "wp_stale",
        reason="ttl_expired_requires_revalidation",
        now_iso="2026-04-05T10:01:00Z",
    )
    assert pending is not None
    assert pending["freshness_lifecycle_state"] == "revalidation_pending"
    assert pending["revalidation_requested_at"] == "2026-04-05T10:01:00Z"
    assert pending["freshness_reason"] == "ttl_expired_requires_revalidation"


def test_missing_validation_basis_is_conservative(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_missing_basis",
        canonical_fact="No validation basis fact.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.9,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/no-basis"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        now_iso="2026-04-03T10:00:00Z",
        last_validated_at=None,
    )
    # Force missing validation basis by removing temporal fields.
    with sqlite3.connect(tmp_path / "web_promoted_memory.sqlite3") as conn:
        conn.execute(
            "UPDATE web_promoted_memory SET last_validated_at=NULL, approved_at=NULL, state_updated_at=NULL WHERE promoted_id='wp_missing_basis'"
        )
        conn.commit()
    # Re-open store to trigger conservative migration/update guardrail.
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    record = store.get("wp_missing_basis")
    assert record is not None
    assert record["freshness_lifecycle_state"] == "stale"
    assert record["freshness_reason"] == "missing_validation_basis"


def test_web_lane_surfaces_fresh_vs_stale_vs_revalidation_pending(tmp_path: Path, monkeypatch):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_fresh",
        canonical_fact="Bitcoin moved on macro trends.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.78,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/fresh", "freshness_ttl_hours": 120},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        now_iso="2026-04-03T10:00:00Z",
    )
    store.upsert_fact(
        promoted_id="wp_old",
        canonical_fact="Bitcoin moved on macro trends.",
        evidence_ids=["ev3", "ev4"],
        confidence=0.78,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/old", "freshness_ttl_hours": 12},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        now_iso="2026-04-01T10:00:00Z",
    )
    store.evaluate_freshness("wp_old", now_iso="2026-04-03T10:00:00Z")
    store.request_revalidation("wp_old", reason="ttl_expired_requires_revalidation", now_iso="2026-04-03T10:01:00Z")

    monkeypatch.setattr(web_lane.WebFetcher, "fetch_plan", lambda self, plan, need, min_cache_hits=2: [])
    hits = web_lane.retrieve(
        workdir=str(tmp_path),
        query="Bitcoin moved",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=4,
    )
    promoted = [h for h in hits if h.memory_type == MemoryType.WEB_PROMOTED_FACT]
    assert promoted and len(promoted) >= 2
    states = [h.provenance.get("freshness_lifecycle_state") for h in promoted[:2]]
    assert states[0] == "fresh"
    assert states[1] in {"revalidation_pending", "stale"}
    assert promoted[0].trust_score >= promoted[1].trust_score
    assert "freshness_reason" in promoted[0].provenance
    assert "revalidation_requested_at" in promoted[1].provenance

