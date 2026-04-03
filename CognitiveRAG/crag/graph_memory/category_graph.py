from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

from CognitiveRAG.crag.contracts.enums import RetrievalLane

from .schemas import GraphEdge, GraphNode, GraphRelationType, stable_edge_id, stable_node_id
from .store import GraphMemoryStore

if TYPE_CHECKING:
    from CognitiveRAG.crag.retrieval.models import LaneHit


CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "engineering_db": ("postgres", "mysql", "sqlite", "schema", "migration", "index", "query"),
    "engineering_backend": ("backend", "api", "service", "controller", "endpoint", "runtime"),
    "ai_memory": ("rag", "retrieval", "memory", "context", "selector", "token"),
    "content_marketing": ("copywriting", "retention", "hook", "youtube", "storyboard", "cta"),
    "operations_reliability": ("incident", "rollback", "sre", "outage", "latency", "reliability"),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@dataclass(frozen=True)
class CategoryHint:
    category: str
    score: float
    matched_keywords: Tuple[str, ...]


@dataclass(frozen=True)
class CategoryRoutingDecision:
    hinted_categories: Tuple[str, ...]
    strong_signal: bool
    score: float
    reason: str


def infer_category_hints(text: str, *, max_categories: int = 3) -> List[CategoryHint]:
    normalized = _normalize(text)
    if not normalized:
        return []

    hints: List[CategoryHint] = []
    for category, words in CATEGORY_KEYWORDS.items():
        matched = sorted({word for word in words if word in normalized})
        if not matched:
            continue
        score = min(1.0, 0.3 + (0.2 * len(matched)))
        hints.append(CategoryHint(category=category, score=score, matched_keywords=tuple(matched)))

    hints.sort(key=lambda h: (-h.score, h.category))
    return hints[: max(1, int(max_categories))]


def decide_query_category_routing(query: str) -> CategoryRoutingDecision:
    hints = infer_category_hints(query, max_categories=2)
    if not hints:
        return CategoryRoutingDecision(hinted_categories=(), strong_signal=False, score=0.0, reason="no_category_signal")

    top = hints[0]
    strong = top.score >= 0.7
    return CategoryRoutingDecision(
        hinted_categories=tuple(h.category for h in hints),
        strong_signal=strong,
        score=float(top.score),
        reason=("strong_category_signal" if strong else "weak_category_signal"),
    )


def _category_node(category: str, *, now_iso: str) -> GraphNode:
    return GraphNode(
        node_id=stable_node_id("category", category),
        node_type="category",
        label=category,
        properties={"category": category},
        provenance={"source": "category_graph", "kind": "taxonomy_seed"},
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


def record_category_relations_for_hits(
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

        hints = infer_category_hints(f"{hit.text} {hit.provenance}", max_categories=3)
        if not hints:
            continue

        created: List[Dict[str, Any]] = []
        nodes = [src_node]
        edges: List[GraphEdge] = []
        for hint in hints:
            cat = hint.category
            cat_node = _category_node(cat, now_iso=now)
            nodes.append(cat_node)
            edge = GraphEdge(
                edge_id=stable_edge_id(src_node.node_id, GraphRelationType.BELONGS_TO_CATEGORY, cat_node.node_id),
                source_node_id=src_node.node_id,
                relation_type=GraphRelationType.BELONGS_TO_CATEGORY,
                target_node_id=cat_node.node_id,
                properties={
                    "category": cat,
                    "score": float(hint.score),
                    "matched_keywords": list(hint.matched_keywords),
                    "lane": hit.lane.value,
                },
                provenance={
                    "reason": "deterministic_keyword_category_inference",
                    "source_hit_id": hit.id,
                    "source_class": "retrieval_hit",
                },
                created_at=now,
                updated_at=now,
            )
            edges.append(edge)
            created.append({"category": cat, "score": float(hint.score), "matched_keywords": list(hint.matched_keywords)})

        store.upsert_nodes(nodes)
        store.upsert_edges(edges)
        out[hit.id] = created

    return out


def read_categories_for_node(store: GraphMemoryStore, *, node_type: str, node_key: str) -> List[Dict[str, Any]]:
    node_id = stable_node_id(node_type, node_key)
    edges = store.get_edges_for_node(node_id, direction="outgoing")
    edges = [e for e in edges if e.relation_type == GraphRelationType.BELONGS_TO_CATEGORY]
    rows: List[Dict[str, Any]] = []
    for edge in sorted(edges, key=lambda e: (-float(e.properties.get("score") or 0.0), e.edge_id)):
        cat_node = store.get_node(edge.target_node_id)
        rows.append(
            {
                "category": (cat_node.properties.get("category") if cat_node else edge.properties.get("category")),
                "score": float(edge.properties.get("score") or 0.0),
                "matched_keywords": list(edge.properties.get("matched_keywords") or []),
                "edge_id": edge.edge_id,
                "node_id": node_id,
            }
        )
    return rows


def categories_for_hit_from_graph(store: GraphMemoryStore, hit: LaneHit) -> List[Dict[str, Any]]:
    source = _source_node_for_hit(hit)
    if not source:
        return []
    node_type, node_key = source
    return read_categories_for_node(store, node_type=node_type, node_key=node_key)
