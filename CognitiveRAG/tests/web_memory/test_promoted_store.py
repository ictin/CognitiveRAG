from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def test_promoted_store_keeps_source_bundle_and_freshness(tmp_path):
    store = WebPromotedMemoryStore(tmp_path / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp:1",
        canonical_fact="Postgres 18 released in 2026.",
        evidence_ids=["e1", "e2"],
        confidence=0.9,
        freshness_state="warm",
        metadata={"source_bundle": ["https://example.com/postgres"]},
        now_iso="2026-03-29T12:02:00Z",
    )

    hits = store.search("Postgres", top_k=5)
    assert hits
    assert hits[0]["evidence_ids"] == ["e1", "e2"]
    assert hits[0]["freshness_state"] == "warm"
