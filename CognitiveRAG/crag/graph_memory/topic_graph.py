from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

from CognitiveRAG.crag.contracts.enums import RetrievalLane

from .schemas import GraphEdge, GraphNode, GraphRelationType, stable_edge_id, stable_node_id
from .store import GraphMemoryStore

if TYPE_CHECKING:
    from CognitiveRAG.crag.retrieval.models import LaneHit


TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "migration_rollout_safety": ("migration", "rollback", "rollout", "guardrail", "staging"),
    "database_change_risk": ("postgres", "mysql", "schema", "index", "query", "locking"),
    "runtime_reliability": ("incident", "outage", "latency", "timeout", "retry", "fallback"),
    "memory_retrieval_design": ("rag", "retrieval", "selector", "context", "token", "provenance"),
    "execution_quality_loop": ("execution", "evaluation", "rubric", "improvement", "quality"),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@dataclass(frozen=True)
class TopicHint:
    topic: str
    score: float
    matched_keywords: Tuple[str, ...]


@dataclass(frozen=True)
class TopicBridgeDecision:
    hinted_topics: Tuple[str, ...]
    strong_signal: bool
    score: float
    reason: str


def infer_topic_hints(text: str, *, max_topics: int = 3) -> List[TopicHint]:
    normalized = _normalize(text)
    if not normalized:
        return []

    hints: List[TopicHint] = []
    for topic, words in TOPIC_KEYWORDS.items():
        matched = sorted({word for word in words if word in normalized})
        if not matched:
            continue
        score = min(1.0, 0.3 + (0.2 * len(matched)))
        hints.append(TopicHint(topic=topic, score=score, matched_keywords=tuple(matched)))

    hints.sort(key=lambda h: (-h.score, h.topic))
    return hints[: max(1, int(max_topics))]


def decide_query_topic_bridge(query: str) -> TopicBridgeDecision:
    hints = infer_topic_hints(query, max_topics=2)
    if not hints:
        return TopicBridgeDecision(hinted_topics=(), strong_signal=False, score=0.0, reason="no_topic_signal")

    top = hints[0]
    strong = top.score >= 0.7
    return TopicBridgeDecision(
        hinted_topics=tuple(h.topic for h in hints),
        strong_signal=strong,
        score=float(top.score),
        reason=("strong_topic_signal" if strong else "weak_topic_signal"),
    )


def _topic_node(topic: str, *, now_iso: str) -> GraphNode:
    return GraphNode(
        node_id=stable_node_id("topic", topic),
        node_type="topic",
        label=topic,
        properties={"topic": topic},
        provenance={"source": "topic_graph", "kind": "topic_seed"},
        created_at=now_iso,
        updated_at=now_iso,
    )


def _source_node_for_hit(hit: LaneHit) -> tuple[str, str] | None:
    hid = str(hit.id or "")
    prov = dict(hit.provenance or {})

    if hit.lane == RetrievalLane.PROMOTED and hid.startswith("promoted:"):
        pattern_id = hid.split(":", 1)[1]
        return ("reasoning_pattern", pattern_id)

    if hit.memory_type.value == "web_promoted_fact":
        promoted_id = str(prov.get("promoted_id") or "")
        if not promoted_id and ":" in hid:
            promoted_id = hid.split(":", 1)[1]
        if promoted_id:
            return ("web_promoted", promoted_id)

    if hit.lane == RetrievalLane.CORPUS and hid.startswith("corpus:"):
        return ("corpus_chunk", hid.split(":", 1)[1])

    if hit.lane == RetrievalLane.LARGE_FILE and hid.startswith("large_file:"):
        return ("large_file_chunk", hid.split(":", 1)[1])

    if hit.memory_type.value == "web_evidence":
        evidence_id = str(prov.get("evidence_id") or "")
        if not evidence_id and hid.startswith("webevidence:"):
            evidence_id = hid.split(":", 1)[1]
        if evidence_id:
            return ("web_evidence", evidence_id)

    return None


def record_topic_relations_for_hits(
    store: GraphMemoryStore,
    *,
    hits: Iterable[LaneHit],
    now_iso: str | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    now = now_iso or _now_iso()
    out: Dict[str, List[Dict[str, Any]]] = {}

    for hit in hits:
        source = _source_node_for_hit(hit)
        if not source:
            continue
        node_type, node_key = source
        source_node_id = stable_node_id(node_type, node_key)
        src_node = GraphNode(
            node_id=source_node_id,
            node_type=node_type,
            label=(node_key[:160] if node_key else None),
            properties={"source_key": node_key, "lane": hit.lane.value, "memory_type": hit.memory_type.value},
            provenance={"source_class": "retrieval_hit", "hit_id": hit.id},
            created_at=now,
            updated_at=now,
        )

        hints = infer_topic_hints(f"{hit.text} {hit.provenance}", max_topics=3)
        if not hints:
            continue

        created: List[Dict[str, Any]] = []
        nodes = [src_node]
        edges: List[GraphEdge] = []
        for hint in hints:
            topic = hint.topic
            topic_node = _topic_node(topic, now_iso=now)
            nodes.append(topic_node)
            edge = GraphEdge(
                edge_id=stable_edge_id(src_node.node_id, GraphRelationType.BELONGS_TO_TOPIC, topic_node.node_id),
                source_node_id=src_node.node_id,
                relation_type=GraphRelationType.BELONGS_TO_TOPIC,
                target_node_id=topic_node.node_id,
                properties={
                    "topic": topic,
                    "score": float(hint.score),
                    "matched_keywords": list(hint.matched_keywords),
                    "lane": hit.lane.value,
                },
                provenance={
                    "reason": "deterministic_keyword_topic_inference",
                    "source_hit_id": hit.id,
                    "source_class": "retrieval_hit",
                },
                created_at=now,
                updated_at=now,
            )
            edges.append(edge)
            created.append({"topic": topic, "score": float(hint.score), "matched_keywords": list(hint.matched_keywords)})

        store.upsert_nodes(nodes)
        store.upsert_edges(edges)
        out[hit.id] = created

    return out


def read_topics_for_node(store: GraphMemoryStore, *, node_type: str, node_key: str) -> List[Dict[str, Any]]:
    node_id = stable_node_id(node_type, node_key)
    edges = store.get_edges_for_node(node_id, direction="outgoing")
    edges = [e for e in edges if e.relation_type == GraphRelationType.BELONGS_TO_TOPIC]
    rows: List[Dict[str, Any]] = []
    for edge in sorted(edges, key=lambda e: (-float(e.properties.get("score") or 0.0), e.edge_id)):
        topic_node = store.get_node(edge.target_node_id)
        rows.append(
            {
                "topic": (topic_node.properties.get("topic") if topic_node else edge.properties.get("topic")),
                "score": float(edge.properties.get("score") or 0.0),
                "matched_keywords": list(edge.properties.get("matched_keywords") or []),
                "edge_id": edge.edge_id,
                "node_id": node_id,
            }
        )
    return rows


def topics_for_hit_from_graph(store: GraphMemoryStore, hit: LaneHit) -> List[Dict[str, Any]]:
    source = _source_node_for_hit(hit)
    if not source:
        return []
    node_type, node_key = source
    return read_topics_for_node(store, node_type=node_type, node_key=node_key)
