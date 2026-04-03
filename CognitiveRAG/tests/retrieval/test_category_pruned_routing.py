from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.retrieval.router import clear_hot_cache, route_and_retrieve


def _hit(*, hid: str, lane: RetrievalLane, text: str) -> LaneHit:
    memory_type = {
        RetrievalLane.CORPUS: MemoryType.CORPUS_CHUNK,
        RetrievalLane.LARGE_FILE: MemoryType.LARGE_FILE_EXCERPT,
        RetrievalLane.WEB: MemoryType.WEB_EVIDENCE,
        RetrievalLane.PROMOTED: MemoryType.PROMOTED_FACT,
    }.get(lane, MemoryType.CORPUS_CHUNK)
    return (
        LaneHit(
            id=hid,
            lane=lane,
            memory_type=memory_type,
            text=text,
            provenance={"source_url": f"https://example.com/{hid.replace(':', '_')}"},
            lexical_score=0.5,
            semantic_score=0.6,
            recency_score=0.4,
            freshness_score=0.5,
            trust_score=0.7,
            novelty_score=0.3,
            contradiction_risk=0.0,
        ).with_token_estimate()
    )


def test_category_pruned_routing_strong_signal_prunes_expensive_lanes(tmp_path: Path, monkeypatch):
    clear_hot_cache()

    from CognitiveRAG.crag.retrieval import router as r

    topk_by_lane: dict[RetrievalLane, int] = {}

    def fake_fast(**_kw):
        return []

    def fake_corpus(**kw):
        topk_by_lane[RetrievalLane.CORPUS] = int(kw["top_k"])
        return [
            _hit(hid="corpus:db_good", lane=RetrievalLane.CORPUS, text="postgres migration rollback runbook"),
            _hit(hid="corpus:mkt_bad", lane=RetrievalLane.CORPUS, text="youtube retention cta hooks"),
        ]

    def fake_large_file(**kw):
        topk_by_lane[RetrievalLane.LARGE_FILE] = int(kw["top_k"])
        return [
            _hit(hid="large_file:db_doc", lane=RetrievalLane.LARGE_FILE, text="schema migration and postgres index maintenance")
        ]

    monkeypatch.setattr(r.fast_lanes, "retrieve_fast_lanes", fake_fast)
    monkeypatch.setitem(r.LANE_HANDLERS, RetrievalLane.CORPUS, fake_corpus)
    monkeypatch.setitem(r.LANE_HANDLERS, RetrievalLane.LARGE_FILE, fake_large_file)

    plan, hits = route_and_retrieve(
        query="postgres migration schema rollback",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="cat-strong",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=8,
    )

    route_meta = dict((plan.metadata or {}).get("category_routing") or {})
    assert route_meta["strong_signal"] is True
    assert RetrievalLane.CORPUS.value in set(route_meta["pruned_lanes"])
    assert int(route_meta["pruned_hit_count"]) >= 1
    assert topk_by_lane[RetrievalLane.CORPUS] == 4
    assert topk_by_lane[RetrievalLane.LARGE_FILE] == 4

    hit_ids = {h.id for h in hits}
    assert "corpus:db_good" in hit_ids
    assert "corpus:mkt_bad" not in hit_ids

    matched = next(h for h in hits if h.id == "corpus:db_good")
    cg = dict((matched.provenance or {}).get("category_graph") or {})
    assert int(cg.get("category_count") or 0) >= 1


def test_category_pruned_routing_weak_signal_falls_back_without_prune(tmp_path: Path, monkeypatch):
    clear_hot_cache()

    from CognitiveRAG.crag.retrieval import router as r

    observed_topk: list[int] = []

    def fake_fast(**_kw):
        return []

    def fake_corpus(**kw):
        observed_topk.append(int(kw["top_k"]))
        return [
            _hit(hid="corpus:a", lane=RetrievalLane.CORPUS, text="backend api service controller"),
            _hit(hid="corpus:b", lane=RetrievalLane.CORPUS, text="retention marketing hooks"),
        ]

    monkeypatch.setattr(r.fast_lanes, "retrieve_fast_lanes", fake_fast)
    monkeypatch.setitem(r.LANE_HANDLERS, RetrievalLane.CORPUS, fake_corpus)

    plan, hits = route_and_retrieve(
        query="hello generic question",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="cat-weak",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=8,
    )

    route_meta = dict((plan.metadata or {}).get("category_routing") or {})
    assert route_meta["strong_signal"] is False
    assert observed_topk[0] == 8
    assert list(route_meta["pruned_lanes"]) == []
    assert int(route_meta["pruned_hit_count"]) == 0
    assert {h.id for h in hits} >= {"corpus:a", "corpus:b"}


def test_category_pruned_routing_strong_signal_fallback_when_no_matches(tmp_path: Path, monkeypatch):
    clear_hot_cache()

    from CognitiveRAG.crag.retrieval import router as r

    def fake_fast(**_kw):
        return []

    def fake_corpus(**_kw):
        return [
            _hit(hid="corpus:only_marketing", lane=RetrievalLane.CORPUS, text="youtube retention hooks cta"),
        ]

    monkeypatch.setattr(r.fast_lanes, "retrieve_fast_lanes", fake_fast)
    monkeypatch.setitem(r.LANE_HANDLERS, RetrievalLane.CORPUS, fake_corpus)

    plan, hits = route_and_retrieve(
        query="postgres migration schema rollback",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="cat-fallback",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=str(tmp_path),
        top_k_per_lane=8,
    )

    route_meta = dict((plan.metadata or {}).get("category_routing") or {})
    assert route_meta["strong_signal"] is True
    assert RetrievalLane.CORPUS.value in set(route_meta["fallback_lanes"])
    assert "corpus:only_marketing" in {h.id for h in hits}
