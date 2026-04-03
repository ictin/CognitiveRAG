import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType
from CognitiveRAG.crag.retrieval import web_lane
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def _upsert_claim(
    store: WebPromotedMemoryStore,
    *,
    promoted_id: str,
    fact: str,
    claim_key: str | None,
    claim_value: str | None,
    source_class: str,
    now_iso: str,
) -> None:
    metadata = {"source_class": source_class, "source_url": f"https://example.com/{promoted_id}"}
    if claim_key is not None:
        metadata["claim_key"] = claim_key
    if claim_value is not None:
        metadata["claim_value"] = claim_value
    store.upsert_fact(
        promoted_id=promoted_id,
        canonical_fact=fact,
        evidence_ids=[f"{promoted_id}-e1", f"{promoted_id}-e2"],
        confidence=0.85,
        freshness_state="warm",
        metadata=metadata,
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        now_iso=now_iso,
    )


def test_conflicting_claim_creates_single_persisted_contradiction_and_preserves_both_claims(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    _upsert_claim(
        store,
        promoted_id="local_weather",
        fact="Tomorrow SF temperature is 20C.",
        claim_key="weather.sf.tomorrow.temp_c",
        claim_value="20",
        source_class="local_durable",
        now_iso="2026-04-03T10:00:00Z",
    )
    _upsert_claim(
        store,
        promoted_id="web_weather",
        fact="Tomorrow SF temperature is 30C.",
        claim_key="weather.sf.tomorrow.temp_c",
        claim_value="30",
        source_class="web_promoted",
        now_iso="2026-04-03T10:01:00Z",
    )

    first = store.get_contradictions_for("web_weather")
    assert len(first) == 1
    assert first[0]["detection_rule"] == "claim_key_value_mismatch"
    assert first[0]["source_basis"]["claim_key"] == "weather.sf.tomorrow.temp_c"
    assert first[0]["source_basis"]["claim_a_source_class"] in {"local_durable", "web_promoted"}
    assert first[0]["source_basis"]["claim_b_source_class"] in {"local_durable", "web_promoted"}

    # Repeating the write should not create duplicate contradiction rows.
    _upsert_claim(
        store,
        promoted_id="web_weather",
        fact="Tomorrow SF temperature is 30C.",
        claim_key="weather.sf.tomorrow.temp_c",
        claim_value="30",
        source_class="web_promoted",
        now_iso="2026-04-03T10:02:00Z",
    )
    second = store.get_contradictions_for("web_weather")
    assert len(second) == 1
    assert second[0]["contradiction_id"] == first[0]["contradiction_id"]

    # Conflict registration must preserve both claims and avoid overwrite.
    local = store.get("local_weather")
    web = store.get("web_weather")
    assert local is not None and web is not None
    assert local["canonical_fact"] == "Tomorrow SF temperature is 20C."
    assert web["canonical_fact"] == "Tomorrow SF temperature is 30C."


def test_non_conflicting_or_ambiguous_claims_do_not_create_false_contradictions(tmp_path: Path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    _upsert_claim(
        store,
        promoted_id="a",
        fact="Claim A",
        claim_key="facts.key",
        claim_value="same",
        source_class="local_durable",
        now_iso="2026-04-03T10:00:00Z",
    )
    _upsert_claim(
        store,
        promoted_id="b",
        fact="Claim B",
        claim_key="facts.key",
        claim_value="same",
        source_class="web_promoted",
        now_iso="2026-04-03T10:01:00Z",
    )
    _upsert_claim(
        store,
        promoted_id="c",
        fact="Claim C",
        claim_key=None,
        claim_value=None,
        source_class="web_promoted",
        now_iso="2026-04-03T10:02:00Z",
    )
    assert store.get_contradictions_for("a") == []
    assert store.get_contradictions_for("b") == []
    assert store.get_contradictions_for("c") == []


def test_web_lane_surfaces_contradiction_with_freshness_context(tmp_path: Path, monkeypatch):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    _upsert_claim(
        store,
        promoted_id="local_truth",
        fact="API v1 is deprecated.",
        claim_key="api.v1.deprecated",
        claim_value="true",
        source_class="local_durable",
        now_iso="2026-04-03T10:00:00Z",
    )
    _upsert_claim(
        store,
        promoted_id="web_counter",
        fact="API v1 is not deprecated.",
        claim_key="api.v1.deprecated",
        claim_value="false",
        source_class="web_promoted",
        now_iso="2026-04-03T10:05:00Z",
    )
    # Make one side stale to ensure freshness context remains visible.
    store.evaluate_freshness("local_truth", now_iso="2026-04-09T10:00:00Z")

    monkeypatch.setattr(web_lane.WebFetcher, "fetch_plan", lambda self, plan, need, min_cache_hits=2: [])
    hits = web_lane.retrieve(
        workdir=str(tmp_path),
        query="API v1",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=4,
    )
    promoted_hits = [h for h in hits if h.memory_type == MemoryType.WEB_PROMOTED_FACT]
    assert promoted_hits
    by_id = {h.id: h for h in promoted_hits}
    assert "webpromoted:web_counter" in by_id
    web_hit = by_id["webpromoted:web_counter"]
    assert web_hit.provenance["contradiction"]["has_contradiction"] is True
    assert web_hit.provenance["contradiction"]["open_contradiction_count"] >= 1
    assert web_hit.contradiction_risk >= 0.65
    assert "freshness_lifecycle_state" in web_hit.provenance
    contradiction_rows = web_hit.provenance["contradiction"]["contradictions"]
    assert contradiction_rows
    assert contradiction_rows[0]["other_claim_freshness_lifecycle_state"] in {
        WebPromotedMemoryStore.FRESHNESS_FRESH,
        WebPromotedMemoryStore.FRESHNESS_STALE,
        WebPromotedMemoryStore.FRESHNESS_REVALIDATION_PENDING,
    }


def test_legacy_schema_loads_safely_with_no_contradiction_fields(tmp_path: Path):
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
            VALUES ('legacy', 'Legacy fact', '[]', 0.6, 'warm', '{}', '2026-03-01T10:00:00Z', '2026-03-01T10:00:00Z')
            """
        )
        conn.commit()

    store = WebPromotedMemoryStore(db_path)
    rec = store.get("legacy")
    assert rec is not None
    summary = store.get_contradiction_summary("legacy")
    assert summary["has_contradiction"] is False
    assert summary["open_contradiction_count"] == 0
