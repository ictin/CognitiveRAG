import time
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.retrieval.router import (
    RouteDecisionCache,
    TopicShortlistCache,
    clear_routing_caches,
    route_and_retrieve,
)


def _hit(*, hid: str, lane: RetrievalLane, text: str, provenance: dict | None = None) -> LaneHit:
    memory_type = {
        RetrievalLane.CORPUS: MemoryType.CORPUS_CHUNK,
        RetrievalLane.WEB: MemoryType.WEB_EVIDENCE,
        RetrievalLane.LARGE_FILE: MemoryType.LARGE_FILE_EXCERPT,
        RetrievalLane.PROMOTED: MemoryType.PROMOTED_FACT,
    }.get(lane, MemoryType.CORPUS_CHUNK)
    return (
        LaneHit(
            id=hid,
            lane=lane,
            memory_type=memory_type,
            text=text,
            provenance=dict(provenance or {}),
            lexical_score=0.5,
            semantic_score=0.6,
            recency_score=0.4,
            freshness_score=0.7,
            trust_score=0.8,
            novelty_score=0.2,
            contradiction_risk=0.0,
        ).with_token_estimate()
    )


def test_shortlist_and_route_cache_hit_on_near_repeated_safe_query(tmp_path: Path, monkeypatch):
    clear_routing_caches()
    from CognitiveRAG.crag.retrieval import router as r

    def fake_fast(**_kw):
        return []

    def fake_corpus(**_kw):
        return [_hit(hid="corpus:c1", lane=RetrievalLane.CORPUS, text="postgres migration rollback strategy")]

    monkeypatch.setattr(r.fast_lanes, "retrieve_fast_lanes", fake_fast)
    monkeypatch.setitem(r.LANE_HANDLERS, RetrievalLane.CORPUS, fake_corpus)

    plan1, _hits1 = route_and_retrieve(
        query="postgres migration schema rollback strategy",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="cache-near-repeat",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=8,
    )
    assert plan1.metadata["route_cache"]["hit"] is False
    assert plan1.metadata["category_routing"]["shortlist_cache"]["hit"] is False

    plan2, _hits2 = route_and_retrieve(
        query="strategy for postgres rollback schema migration",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="cache-near-repeat",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=8,
    )
    assert plan2.metadata["route_cache"]["hit"] is True
    assert plan2.metadata["category_routing"]["shortlist_cache"]["hit"] is True


def test_route_cache_miss_when_route_relevant_intent_differs(tmp_path: Path):
    clear_routing_caches()
    plan1, _ = route_and_retrieve(
        query="postgres migration schema rollback strategy",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="cache-intent",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )
    assert plan1.metadata["route_cache"]["hit"] is False
    plan2, _ = route_and_retrieve(
        query="postgres migration schema rollback strategy",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        session_id="cache-intent",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=6,
    )
    assert plan2.metadata["route_cache"]["hit"] is False


def test_shortlist_cache_and_route_cache_bounded_expiry():
    shortlist = TopicShortlistCache(max_entries=2, ttl_seconds=0.01)
    shortlist.set(("a",), hinted_categories=("x",), strong_signal=True, score=0.8, reason="r")
    shortlist.set(("b",), hinted_categories=("x",), strong_signal=False, score=0.2, reason="r")
    shortlist.set(("c",), hinted_categories=("x",), strong_signal=False, score=0.2, reason="r")
    assert int(shortlist.stats()["entries"]) == 2
    assert int(shortlist.stats()["evictions"]) >= 1
    time.sleep(0.02)
    assert shortlist.get(("b",)) is None

    route = RouteDecisionCache(max_entries=1, ttl_seconds=0.01)
    from CognitiveRAG.crag.retrieval.router import RoutePlan

    p = RoutePlan(intent_family=IntentFamily.MEMORY_SUMMARY, lanes=[RetrievalLane.PROMOTED], reason="x")
    route.set(("k1",), p)
    route.set(("k2",), p)
    assert int(route.stats()["entries"]) == 1
    assert int(route.stats()["evictions"]) >= 1
    time.sleep(0.02)
    assert route.get(("k2",)) is None


def test_weak_shortlist_signal_falls_back_without_aggressive_prune(tmp_path: Path, monkeypatch):
    clear_routing_caches()
    from CognitiveRAG.crag.retrieval import router as r

    def fake_fast(**_kw):
        return []

    def fake_corpus(**_kw):
        return [
            _hit(hid="corpus:a", lane=RetrievalLane.CORPUS, text="backend api service"),
            _hit(hid="corpus:b", lane=RetrievalLane.CORPUS, text="youtube retention hooks"),
        ]

    monkeypatch.setattr(r.fast_lanes, "retrieve_fast_lanes", fake_fast)
    monkeypatch.setitem(r.LANE_HANDLERS, RetrievalLane.CORPUS, fake_corpus)

    plan, hits = route_and_retrieve(
        query="hello maybe useful",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="weak-fallback",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=8,
    )
    route_meta = plan.metadata["category_routing"]
    assert route_meta["strong_signal"] is False
    assert int(route_meta["pruned_hit_count"]) == 0
    assert {h.id for h in hits} >= {"corpus:a", "corpus:b"}


def test_cached_routing_keeps_provenance_freshness_and_contradiction_truth(tmp_path: Path, monkeypatch):
    clear_routing_caches()
    from CognitiveRAG.crag.retrieval import router as r

    def fake_fast(**_kw):
        return [
            _hit(
                hid="global_promoted:wp_cache_truth_a",
                lane=RetrievalLane.GLOBAL_PROMOTED,
                text="Feature flag status conflict.",
                provenance={
                    "promotion_tier": "global",
                    "source_class": "web_promoted",
                    "freshness_lifecycle_state": "stale",
                    "contradiction": {"has_contradiction": True},
                },
            )
        ]

    monkeypatch.setattr(r.fast_lanes, "retrieve_fast_lanes", fake_fast)

    _plan1, hits1 = route_and_retrieve(
        query="feature flag status",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        session_id="truth-cache",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=10,
    )
    plan2, hits2 = route_and_retrieve(
        query="status of feature flag",
        intent_family=IntentFamily.MEMORY_SUMMARY,
        session_id="truth-cache",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=10,
    )

    assert plan2.metadata["route_cache"]["hit"] is True
    assert plan2.metadata["category_routing"]["shortlist_cache"]["hit"] is True
    assert [h.id for h in hits1] == [h.id for h in hits2]
    probe = next(h for h in hits2 if h.id.startswith("global_promoted:"))
    assert probe.provenance.get("freshness_lifecycle_state") == "stale"
    assert bool(probe.provenance.get("contradiction", {}).get("has_contradiction")) is True
