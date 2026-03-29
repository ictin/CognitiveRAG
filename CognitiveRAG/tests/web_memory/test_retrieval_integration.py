from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane
from CognitiveRAG.crag.retrieval.router import LaneRouter, route_and_retrieve
from CognitiveRAG.crag.web_memory.evidence_store import WebEvidenceStore


def test_router_adds_web_lane_for_freshness_queries():
    plan = LaneRouter().route(
        query="latest status of postgres release",
        intent_family=IntentFamily.INVESTIGATIVE,
    )
    assert RetrievalLane.WEB in plan.lanes


def test_route_and_retrieve_surfaces_web_hits_from_cache(tmp_path):
    evidence_store = WebEvidenceStore(tmp_path / "web_evidence.sqlite3")
    evidence_store.upsert_evidence(
        {
            "query": "latest postgres release",
            "query_variant": "latest postgres release",
            "source_id": "https://example.com/postgres",
            "url": "https://example.com/postgres",
            "title": "Postgres Release",
            "snippet": "release summary",
            "extracted_text": "release summary detail",
            "fetched_at": "2026-03-29T12:04:00Z",
            "published_at": None,
            "updated_at": None,
            "trust_score": 0.75,
            "freshness_class": "hot",
            "content_hash": "hash2",
            "raw": {},
        }
    )

    plan, hits = route_and_retrieve(
        query="latest postgres release",
        intent_family=IntentFamily.INVESTIGATIVE,
        session_id="s1",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=3,
    )
    assert RetrievalLane.WEB in plan.lanes
    assert any(h.lane == RetrievalLane.WEB for h in hits)
