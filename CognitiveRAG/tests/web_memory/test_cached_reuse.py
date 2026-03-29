from CognitiveRAG.crag.web_memory.evidence_store import WebEvidenceStore
from CognitiveRAG.crag.web_memory.fetch import WebFetcher
from CognitiveRAG.crag.web_memory.fetch_log import WebFetchLogStore
from CognitiveRAG.crag.web_memory.query_planner import WebNeedDecision, WebQueryPlan


def test_cached_reuse_skips_refetch(tmp_path):
    evidence_store = WebEvidenceStore(tmp_path / "web_evidence.sqlite3")
    fetch_log = WebFetchLogStore(tmp_path / "web_fetch_log.sqlite3")

    evidence_store.upsert_evidence(
        {
            "query": "latest postgres",
            "query_variant": "latest postgres",
            "source_id": "https://example.com/postgres",
            "url": "https://example.com/postgres",
            "title": "Pg",
            "snippet": "cached",
            "extracted_text": "cached evidence",
            "fetched_at": "2026-03-29T12:03:00Z",
            "published_at": None,
            "updated_at": None,
            "trust_score": 0.7,
            "freshness_class": "hot",
            "content_hash": "hash1",
            "raw": {},
        }
    )

    calls = {"n": 0}

    def provider(_query, _k):
        calls["n"] += 1
        return [{"title": "New", "href": "https://example.com/new", "body": "new evidence"}]

    fetcher = WebFetcher(evidence_store=evidence_store, fetch_log=fetch_log, provider=provider)
    plan = WebQueryPlan(
        query="latest postgres",
        variants=["latest postgres"],
        source_hints=[],
        freshness_sensitive=True,
        max_results=3,
    )
    need = WebNeedDecision(needed=True, reason="fresh", freshness_sensitive=True, volatility="high")
    out = fetcher.fetch_plan(plan=plan, need=need, min_cache_hits=1)
    assert out
    assert calls["n"] == 0
    logs = fetch_log.list_recent(3)
    assert logs and logs[0]["status"] == "cache_hit"
