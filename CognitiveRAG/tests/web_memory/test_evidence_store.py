from CognitiveRAG.crag.web_memory.evidence_store import WebEvidenceStore


def test_evidence_store_upsert_and_search(tmp_path):
    store = WebEvidenceStore(tmp_path / "web_evidence.sqlite3")
    evidence = {
        "query": "postgres release",
        "query_variant": "postgres release latest",
        "source_id": "https://example.com/postgres",
        "url": "https://example.com/postgres",
        "title": "Postgres Release",
        "snippet": "Release notes summary",
        "extracted_text": "Release notes summary and details",
        "fetched_at": "2026-03-29T12:01:00Z",
        "published_at": "2026-03-20",
        "updated_at": None,
        "trust_score": 0.8,
        "freshness_class": "hot",
        "content_hash": "abc123",
        "raw": {"x": 1},
    }
    evidence_id = store.upsert_evidence(evidence)
    assert evidence_id.startswith("https://example.com/postgres::")

    hits = store.search("release", top_k=3)
    assert hits
    assert hits[0]["url"] == "https://example.com/postgres"
