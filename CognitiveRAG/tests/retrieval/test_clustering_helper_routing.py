from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.retrieval.router import clear_routing_caches, route_and_retrieve


def _hit(*, hid: str, lane: RetrievalLane, text: str) -> LaneHit:
    return (
        LaneHit(
            id=hid,
            lane=lane,
            memory_type=MemoryType.CORPUS_CHUNK,
            text=text,
            provenance={
                "source_class": "corpus",
                "lifecycle_state": "approved",
                "topic_graph": {"topics": [{"topic": "database_change_risk", "score": 0.8}]},
                "category_graph": {"categories": [{"category": "engineering_db", "score": 0.9}]},
            },
            lexical_score=0.5,
            semantic_score=0.6,
            recency_score=0.4,
            freshness_score=0.7,
            trust_score=0.8,
            novelty_score=0.2,
            contradiction_risk=0.0,
        ).with_token_estimate()
    )


def test_clustering_helper_is_stable_non_authoritative_and_disableable(tmp_path: Path, monkeypatch):
    from CognitiveRAG.crag.retrieval import router as r

    def fake_fast(**_kw):
        return []

    def fake_corpus(**_kw):
        return [
            _hit(hid="corpus:1", lane=RetrievalLane.CORPUS, text="postgres migration rollback checklist"),
            _hit(hid="corpus:2", lane=RetrievalLane.CORPUS, text="postgres timeout fallback strategy"),
        ]

    monkeypatch.setattr(r.fast_lanes, "retrieve_fast_lanes", fake_fast)
    monkeypatch.setitem(r.LANE_HANDLERS, RetrievalLane.CORPUS, fake_corpus)
    monkeypatch.delenv("CRAG_DISABLE_CLUSTERING_HELPER", raising=False)
    clear_routing_caches()

    plan_enabled_a, hits_enabled_a = route_and_retrieve(
        query="postgres migration rollback timeout",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="cluster-enabled-a",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )
    clear_routing_caches()
    plan_enabled_b, hits_enabled_b = route_and_retrieve(
        query="postgres migration rollback timeout",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="cluster-enabled-b",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )

    monkeypatch.setenv("CRAG_DISABLE_CLUSTERING_HELPER", "1")
    clear_routing_caches()
    plan_disabled, hits_disabled = route_and_retrieve(
        query="postgres migration rollback timeout",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="cluster-disabled",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )

    assert [l.value for l in plan_enabled_a.lanes] == [l.value for l in plan_disabled.lanes]
    helper_meta_enabled = dict((plan_enabled_a.metadata or {}).get("clustering_helper") or {})
    helper_meta_disabled = dict((plan_disabled.metadata or {}).get("clustering_helper") or {})
    assert helper_meta_enabled.get("helper_enabled") is True
    assert helper_meta_enabled.get("authoritative") is False
    assert int(helper_meta_enabled.get("cluster_count") or 0) >= 1
    assert helper_meta_disabled.get("helper_enabled") is False

    enabled_ids_a = [h.cluster_id for h in hits_enabled_a]
    enabled_ids_b = [h.cluster_id for h in hits_enabled_b]
    assert all(enabled_ids_a)
    assert enabled_ids_a == enabled_ids_b

    for h in hits_enabled_a:
        prov = dict(h.provenance or {})
        assert dict(prov.get("clustering_helper") or {}).get("helper_only") is True
        assert prov.get("source_class") == "corpus"
        assert prov.get("lifecycle_state") == "approved"
        assert "topic_graph" in prov
        assert "category_graph" in prov

    assert all(h.cluster_id is None for h in hits_disabled)
    assert all("clustering_helper" not in (h.provenance or {}) for h in hits_disabled)
