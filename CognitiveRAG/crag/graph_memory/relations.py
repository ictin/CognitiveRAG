from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from CognitiveRAG.schemas.memory import ReasoningPattern

from .schemas import GraphEdge, GraphNode, GraphRelationType, stable_edge_id, stable_node_id
from .store import GraphMemoryStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick_source_key(source: dict[str, Any]) -> str:
    for key in ('source_url', 'url', 'source', 'id', 'name', 'ref'):
        value = source.get(key)
        if value:
            return str(value)
    return str(source)


def _reasoning_pattern_node(pattern: ReasoningPattern, *, now_iso: str | None = None) -> GraphNode:
    now = now_iso or _now_iso()
    pattern_id = (pattern.pattern_id or pattern.item_id or '').strip()
    node_id = stable_node_id('reasoning_pattern', pattern_id)
    return GraphNode(
        node_id=node_id,
        node_type='reasoning_pattern',
        label=pattern.solution_summary[:120] if pattern.solution_summary else pattern.problem_signature,
        properties={
            'pattern_id': pattern.pattern_id,
            'item_id': pattern.item_id,
            'problem_signature': pattern.problem_signature,
            'confidence': pattern.confidence,
            'memory_subtype': pattern.memory_subtype,
        },
        provenance={
            'reasoning_provenance': list(pattern.provenance or []),
        },
        created_at=now,
        updated_at=now,
    )


def _problem_signature_node(problem_signature: str, *, now_iso: str | None = None) -> GraphNode:
    now = now_iso or _now_iso()
    clean = (problem_signature or '').strip()
    return GraphNode(
        node_id=stable_node_id('problem_signature', clean),
        node_type='problem_signature',
        label=clean[:160] if clean else None,
        properties={'problem_signature': clean},
        provenance={},
        created_at=now,
        updated_at=now,
    )


def _source_node(source: Dict[str, Any], *, node_type: str = 'provenance_source', now_iso: str | None = None) -> GraphNode:
    now = now_iso or _now_iso()
    key = _pick_source_key(source)
    return GraphNode(
        node_id=stable_node_id(node_type, key),
        node_type=node_type,
        label=key,
        properties={k: v for k, v in dict(source or {}).items()},
        provenance={},
        created_at=now,
        updated_at=now,
    )


def record_reasoning_pattern_supported_by(
    store: GraphMemoryStore,
    *,
    pattern: ReasoningPattern,
    source: Dict[str, Any],
    provenance: Dict[str, Any] | None = None,
) -> GraphEdge:
    now = _now_iso()
    rp_node = _reasoning_pattern_node(pattern, now_iso=now)
    src_node = _source_node(source, node_type='provenance_source', now_iso=now)
    edge = GraphEdge(
        edge_id=stable_edge_id(rp_node.node_id, GraphRelationType.SUPPORTED_BY, src_node.node_id),
        source_node_id=rp_node.node_id,
        relation_type=GraphRelationType.SUPPORTED_BY,
        target_node_id=src_node.node_id,
        properties={
            'pattern_id': pattern.pattern_id,
            'source_key': _pick_source_key(source),
        },
        provenance={
            'reason': 'reasoning pattern supported by provenance/source',
            **dict(provenance or {}),
        },
        created_at=now,
        updated_at=now,
    )
    store.upsert_nodes([rp_node, src_node])
    store.upsert_edge(edge)
    return edge


def record_web_promoted_derived_from(
    store: GraphMemoryStore,
    *,
    promoted_id: str,
    source_url: str,
    metadata: Dict[str, Any] | None = None,
    provenance: Dict[str, Any] | None = None,
) -> GraphEdge:
    now = _now_iso()
    promoted_node = GraphNode(
        node_id=stable_node_id('web_promoted', promoted_id),
        node_type='web_promoted',
        label=promoted_id,
        properties={
            'promoted_id': promoted_id,
            **dict(metadata or {}),
        },
        provenance={},
        created_at=now,
        updated_at=now,
    )
    source_node = GraphNode(
        node_id=stable_node_id('source_url', source_url),
        node_type='source_url',
        label=source_url,
        properties={'source_url': source_url},
        provenance={},
        created_at=now,
        updated_at=now,
    )
    edge = GraphEdge(
        edge_id=stable_edge_id(promoted_node.node_id, GraphRelationType.DERIVED_FROM, source_node.node_id),
        source_node_id=promoted_node.node_id,
        relation_type=GraphRelationType.DERIVED_FROM,
        target_node_id=source_node.node_id,
        properties={
            'promoted_id': promoted_id,
            'source_url': source_url,
        },
        provenance={
            'reason': 'web promoted fact derived from source URL',
            **dict(provenance or {}),
        },
        created_at=now,
        updated_at=now,
    )
    store.upsert_nodes([promoted_node, source_node])
    store.upsert_edge(edge)
    return edge


def record_problem_signature_resolved_by(
    store: GraphMemoryStore,
    *,
    problem_signature: str,
    pattern: ReasoningPattern,
    provenance: Dict[str, Any] | None = None,
) -> GraphEdge:
    now = _now_iso()
    ps_node = _problem_signature_node(problem_signature, now_iso=now)
    rp_node = _reasoning_pattern_node(pattern, now_iso=now)
    edge = GraphEdge(
        edge_id=stable_edge_id(ps_node.node_id, GraphRelationType.RESOLVED_BY, rp_node.node_id),
        source_node_id=ps_node.node_id,
        relation_type=GraphRelationType.RESOLVED_BY,
        target_node_id=rp_node.node_id,
        properties={
            'problem_signature': (problem_signature or '').strip(),
            'pattern_id': pattern.pattern_id,
        },
        provenance={
            'reason': 'problem signature resolved by reasoning pattern',
            **dict(provenance or {}),
        },
        created_at=now,
        updated_at=now,
    )
    store.upsert_nodes([ps_node, rp_node])
    store.upsert_edge(edge)
    return edge
