from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.graph_memory.schemas import GraphRelationType, stable_node_id
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.crag.graph_memory.topic_graph import (
    decide_query_topic_bridge,
    infer_topic_hints,
    read_topics_for_node,
    record_topic_relations_for_hits,
)
from CognitiveRAG.crag.retrieval.models import LaneHit


def test_infer_topic_hints_is_deterministic_and_bounded():
    text = "postgres migration rollback api timeout fallback"
    first = infer_topic_hints(text)
    second = infer_topic_hints(text)
    assert [(h.topic, h.score, h.matched_keywords) for h in first] == [(h.topic, h.score, h.matched_keywords) for h in second]
    assert len(first) <= 3
    assert first[0].score >= first[-1].score


def test_decide_query_topic_bridge_marks_strong_vs_weak():
    strong = decide_query_topic_bridge("postgres migration rollback timeout")
    weak = decide_query_topic_bridge("hello there maybe things")
    assert strong.strong_signal is True
    assert strong.hinted_topics
    assert weak.strong_signal is False


def test_record_topic_relations_roundtrip_with_provenance(tmp_path):
    store = GraphMemoryStore(tmp_path / "graph.sqlite3")
    hit = LaneHit(
        id="webpromoted:wp_pg",
        lane=RetrievalLane.WEB,
        memory_type=MemoryType.WEB_PROMOTED_FACT,
        text="Postgres migration rollback checklist for backend service timeout handling.",
        provenance={"promoted_id": "wp_pg", "source_url": "https://example.com/pg"},
        lexical_score=0.5,
        semantic_score=0.6,
        recency_score=0.4,
        freshness_score=0.7,
        trust_score=0.8,
        novelty_score=0.2,
        contradiction_risk=0.0,
    ).with_token_estimate()

    out = record_topic_relations_for_hits(store, hits=[hit], now_iso="2026-05-04T10:00:00Z")
    assert "webpromoted:wp_pg" in out
    assert out["webpromoted:wp_pg"]

    rows = read_topics_for_node(store, node_type="web_promoted", node_key="wp_pg")
    assert rows
    assert rows[0]["topic"]

    node_id = stable_node_id("web_promoted", "wp_pg")
    edges = store.get_edges_for_node(node_id, direction="outgoing")
    assert edges
    assert all(e.relation_type == GraphRelationType.BELONGS_TO_TOPIC for e in edges)
    assert all(e.provenance.get("reason") == "deterministic_keyword_topic_inference" for e in edges)


def test_topic_relation_writes_are_additive_and_deterministic(tmp_path):
    store = GraphMemoryStore(tmp_path / "graph.sqlite3")
    hit = LaneHit(
        id="corpus:c1",
        lane=RetrievalLane.CORPUS,
        memory_type=MemoryType.CORPUS_CHUNK,
        text="API service controller backend endpoint migration rollback timeout steps",
        provenance={},
        lexical_score=0.3,
        semantic_score=0.4,
        recency_score=0.2,
        freshness_score=0.2,
        trust_score=0.5,
        novelty_score=0.2,
        contradiction_risk=0.0,
    ).with_token_estimate()

    record_topic_relations_for_hits(store, hits=[hit], now_iso="2026-05-04T10:00:00Z")
    record_topic_relations_for_hits(store, hits=[hit], now_iso="2026-05-04T10:00:00Z")

    node_id = stable_node_id("corpus_chunk", "c1")
    edges = store.get_edges_for_node(node_id, direction="outgoing")
    assert edges
    ids = [e.edge_id for e in edges]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
