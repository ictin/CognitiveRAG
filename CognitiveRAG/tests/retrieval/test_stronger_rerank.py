from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.retrieval.rerank import rerank_hits


def _hit(
    *,
    hit_id: str,
    text: str,
    semantic: float,
    lexical: float,
    trust: float,
    provenance: dict,
    contradiction_risk: float = 0.0,
) -> LaneHit:
    return LaneHit(
        id=hit_id,
        lane=RetrievalLane.PROMOTED,
        memory_type=MemoryType.PROMOTED_FACT,
        text=text,
        provenance=dict(provenance),
        semantic_score=float(semantic),
        lexical_score=float(lexical),
        trust_score=float(trust),
        freshness_score=0.6,
        recency_score=0.4,
        novelty_score=0.2,
        contradiction_risk=float(contradiction_risk),
    ).with_token_estimate()


def test_stronger_rerank_reorders_when_caution_signals_present():
    bad = _hit(
        hit_id="a_bad",
        text="cache route rules for rerank",
        semantic=0.92,
        lexical=0.86,
        trust=0.8,
        provenance={
            "source_class": "web_promoted",
            "promotion_tier": "global",
            "freshness_lifecycle_state": "stale",
            "contradiction": {"has_contradiction": True},
        },
        contradiction_risk=0.9,
    )
    good = _hit(
        hit_id="b_good",
        text="cache route rules for rerank",
        semantic=0.74,
        lexical=0.72,
        trust=0.82,
        provenance={
            "source_class": "local_durable",
            "promotion_tier": "workspace",
            "freshness_lifecycle_state": "fresh",
            "reuse_count": 3,
            "success_confidence": 0.8,
        },
    )
    out = rerank_hits(
        query="cache route rules rerank",
        hits=[bad, good],
        plan_lanes=[RetrievalLane.PROMOTED],
        hinted_categories=[],
        category_strong=False,
    )
    assert out.metadata["applied"] is True
    assert out.metadata["moved_count"] >= 1
    assert [h.id for h in out.hits][:2] == ["b_good", "a_bad"]
    # Source truth is preserved; rerank is additive metadata only.
    assert out.hits[0].provenance.get("source_class") == "local_durable"
    assert out.hits[1].provenance.get("source_class") == "web_promoted"
    assert "rerank" in out.hits[0].provenance
    assert "rerank" in out.hits[1].provenance


def test_stronger_rerank_falls_back_on_empty_query_terms():
    first = _hit(
        hit_id="x1",
        text="first row",
        semantic=0.4,
        lexical=0.4,
        trust=0.4,
        provenance={"source_class": "local_durable"},
    )
    second = _hit(
        hit_id="x2",
        text="second row",
        semantic=0.41,
        lexical=0.39,
        trust=0.4,
        provenance={"source_class": "local_durable"},
    )
    out = rerank_hits(
        query="",
        hits=[first, second],
        plan_lanes=[RetrievalLane.PROMOTED],
        hinted_categories=[],
        category_strong=False,
    )
    assert out.metadata["applied"] is False
    assert out.metadata["reason"] == "empty_query_terms"
    assert [h.id for h in out.hits] == ["x1", "x2"]


def test_stronger_rerank_is_deterministic():
    a = _hit(
        hit_id="d1",
        text="routing cache controls",
        semantic=0.62,
        lexical=0.61,
        trust=0.7,
        provenance={"source_class": "workspace_promoted", "reuse_count": 2},
    )
    b = _hit(
        hit_id="d2",
        text="routing cache controls",
        semantic=0.62,
        lexical=0.61,
        trust=0.7,
        provenance={"source_class": "workspace_promoted", "reuse_count": 2},
    )
    run1 = rerank_hits(
        query="routing cache controls",
        hits=[a, b],
        plan_lanes=[RetrievalLane.PROMOTED],
        hinted_categories=[],
        category_strong=False,
    )
    run2 = rerank_hits(
        query="routing cache controls",
        hits=[a, b],
        plan_lanes=[RetrievalLane.PROMOTED],
        hinted_categories=[],
        category_strong=False,
    )
    assert [h.id for h in run1.hits] == [h.id for h in run2.hits]
    assert run1.metadata["strategy"] == run2.metadata["strategy"]

