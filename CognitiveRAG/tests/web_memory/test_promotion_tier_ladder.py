import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType
from CognitiveRAG.crag.retrieval import web_lane
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def test_local_to_workspace_promotion_preserves_lineage_and_is_idempotent(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_local",
        canonical_fact="Postgres extension X is stable.",
        evidence_ids=["ev1"],
        confidence=0.72,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/local"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        promotion_tier=WebPromotedMemoryStore.TIER_LOCAL,
        now_iso="2026-04-03T10:00:00Z",
    )

    promoted = store.promote_local_to_workspace("wp_local", now_iso="2026-04-03T10:05:00Z")
    assert promoted is not None
    assert promoted["promotion_tier"] == WebPromotedMemoryStore.TIER_WORKSPACE
    assert "wp_local" in promoted["promoted_from_ids"]
    assert promoted["promotion_basis"]["rule"] == "local_to_workspace_threshold"
    assert promoted["promotion_history"][-1]["from_tier"] == WebPromotedMemoryStore.TIER_LOCAL
    assert promoted["promotion_history"][-1]["to_tier"] == WebPromotedMemoryStore.TIER_WORKSPACE

    again = store.promote_local_to_workspace("wp_local", now_iso="2026-04-03T10:06:00Z")
    assert again is not None
    assert again["promotion_tier"] == WebPromotedMemoryStore.TIER_WORKSPACE
    assert len(again["promotion_history"]) == len(promoted["promotion_history"])


def test_workspace_to_global_promotion_is_stricter_and_deterministic(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_workspace",
        canonical_fact="Protocol v2 requires signed headers.",
        evidence_ids=["ev1", "ev2", "ev3"],
        confidence=0.87,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/workspace"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        promotion_tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_FRESH,
        now_iso="2026-04-03T10:00:00Z",
    )
    first = store.promote_workspace_to_global("wp_workspace", now_iso="2026-04-03T10:07:00Z")
    second = store.promote_workspace_to_global("wp_workspace", now_iso="2026-04-03T10:08:00Z")
    assert first is not None and second is not None
    assert first["promotion_tier"] == WebPromotedMemoryStore.TIER_GLOBAL
    assert second["promotion_tier"] == WebPromotedMemoryStore.TIER_GLOBAL
    assert first["promotion_history"][-1]["reason"] == "workspace_to_global_strict_threshold"
    assert len(second["promotion_history"]) == len(first["promotion_history"])


def test_workspace_to_global_guardrails_block_stale_or_contradicted_records(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    # Stale workspace record should not auto-promote globally.
    store.upsert_fact(
        promoted_id="wp_stale",
        canonical_fact="Service is available in EU only.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.91,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/stale"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        promotion_tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_STALE,
        now_iso="2026-04-03T10:00:00Z",
    )
    stale_out = store.promote_workspace_to_global("wp_stale", now_iso="2026-04-03T10:10:00Z")
    assert stale_out is not None
    assert stale_out["promotion_tier"] == WebPromotedMemoryStore.TIER_WORKSPACE

    # Contradicted workspace record should not auto-promote globally.
    store.upsert_fact(
        promoted_id="wp_a",
        canonical_fact="Flag alpha is enabled.",
        evidence_ids=["ev1", "ev2"],
        confidence=0.9,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/a", "claim_key": "feature.alpha.enabled", "claim_value": "true"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        promotion_tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_FRESH,
        now_iso="2026-04-03T10:00:00Z",
    )
    store.upsert_fact(
        promoted_id="wp_b",
        canonical_fact="Flag alpha is disabled.",
        evidence_ids=["ev3", "ev4"],
        confidence=0.9,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/b", "claim_key": "feature.alpha.enabled", "claim_value": "false"},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        promotion_tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_FRESH,
        now_iso="2026-04-03T10:01:00Z",
    )
    contradicted = store.promote_workspace_to_global("wp_b", now_iso="2026-04-03T10:11:00Z")
    assert contradicted is not None
    assert contradicted["promotion_tier"] == WebPromotedMemoryStore.TIER_WORKSPACE
    assert store.get_contradiction_summary("wp_b")["open_contradiction_count"] >= 1


def test_web_lane_surfaces_and_orders_promotion_tiers(tmp_path: Path, monkeypatch):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    for promoted_id, tier in (
        ("wp_global", WebPromotedMemoryStore.TIER_GLOBAL),
        ("wp_workspace", WebPromotedMemoryStore.TIER_WORKSPACE),
        ("wp_local", WebPromotedMemoryStore.TIER_LOCAL),
    ):
        store.upsert_fact(
            promoted_id=promoted_id,
            canonical_fact="Routing policy supports bounded retries.",
            evidence_ids=["ev1", "ev2"],
            confidence=0.8,
            freshness_state="warm",
            metadata={"source_url": f"https://example.com/{promoted_id}"},
            promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
            promotion_tier=tier,
            freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_FRESH,
            now_iso="2026-04-03T10:00:00Z",
        )

    monkeypatch.setattr(web_lane.WebFetcher, "fetch_plan", lambda self, plan, need, min_cache_hits=2: [])
    hits = web_lane.retrieve(
        workdir=str(tmp_path),
        query="Routing",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=6,
    )
    promoted = [h for h in hits if h.memory_type == MemoryType.WEB_PROMOTED_FACT]
    assert promoted
    tiers = [str(h.provenance.get("promotion_tier") or "") for h in promoted[:3]]
    assert tiers == ["global", "workspace", "local"]
    assert "promotion_history" in promoted[0].provenance
    assert "promoted_from_ids" in promoted[0].provenance


def test_legacy_rows_without_tier_columns_load_safely(tmp_path: Path):
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
            """
            INSERT INTO web_promoted_memory(promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state, metadata_json, created_at, updated_at)
            VALUES ('legacy_tier', 'Legacy row', '[]', 0.6, 'warm', '{}', '2026-03-01T10:00:00Z', '2026-03-01T10:00:00Z')
            """
        )
        conn.commit()

    store = WebPromotedMemoryStore(db_path)
    row = store.get("legacy_tier")
    assert row is not None
    assert row["promotion_tier"] in {"local", "workspace", "global"}
