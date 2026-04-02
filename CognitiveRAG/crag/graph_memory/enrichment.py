from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from .schemas import GraphRelationType, stable_node_id
from .store import GraphMemoryStore


def _tokens(text: str) -> set[str]:
    return {tok for tok in (text or "").lower().replace(":", " ").replace("-", " ").split() if tok}


@dataclass(frozen=True)
class GraphResolvedByMatch:
    problem_signature: str
    pattern_id: str
    overlap: int
    query_token_count: int
    score: float


class GraphRetrievalEnricher:
    """Optional, deterministic graph enrichment helper for retrieval lanes."""

    def __init__(self, workdir: str):
        self.workdir = workdir
        self.db_path = os.path.join(workdir, "graph.sqlite3")
        self._store = None

    def _store_or_none(self) -> GraphMemoryStore | None:
        if self._store is not None:
            return self._store
        if not os.path.exists(self.db_path):
            return None
        self._store = GraphMemoryStore(self.db_path)
        return self._store

    def get_reasoning_support_links(self, *, pattern_id: str) -> List[Dict]:
        store = self._store_or_none()
        if not store or not pattern_id:
            return []
        pattern_node = stable_node_id("reasoning_pattern", pattern_id)
        edges = store.get_edges_for_node(pattern_node, direction="outgoing")
        supported = [e for e in edges if e.relation_type == GraphRelationType.SUPPORTED_BY]

        links: List[Dict] = []
        for edge in sorted(supported, key=lambda e: (e.target_node_id, e.edge_id)):
            target = store.get_node(edge.target_node_id)
            links.append(
                {
                    "edge_id": edge.edge_id,
                    "relation_type": edge.relation_type,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "source_key": edge.properties.get("source_key"),
                    "target_label": target.label if target else None,
                    "target_properties": (target.properties if target else {}),
                    "edge_provenance": edge.provenance,
                }
            )
        return links

    def get_web_promoted_origins(self, *, promoted_id: str) -> List[Dict]:
        store = self._store_or_none()
        if not store or not promoted_id:
            return []
        promoted_node = stable_node_id("web_promoted", promoted_id)
        edges = store.get_edges_for_node(promoted_node, direction="outgoing")
        derived = [e for e in edges if e.relation_type == GraphRelationType.DERIVED_FROM]

        origins: List[Dict] = []
        for edge in sorted(derived, key=lambda e: (e.target_node_id, e.edge_id)):
            target = store.get_node(edge.target_node_id)
            source_url = edge.properties.get("source_url")
            if not source_url and target:
                source_url = target.properties.get("source_url")
            origins.append(
                {
                    "edge_id": edge.edge_id,
                    "relation_type": edge.relation_type,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "source_url": source_url,
                    "edge_provenance": edge.provenance,
                }
            )
        return origins

    def find_problem_signature_matches(self, *, query: str, max_matches: int = 3) -> List[GraphResolvedByMatch]:
        store = self._store_or_none()
        if not store:
            return []
        q_tokens = _tokens(query)
        if not q_tokens:
            return []

        edges = store.get_edges_by_relation(GraphRelationType.RESOLVED_BY)
        matches: list[GraphResolvedByMatch] = []
        for edge in edges:
            src = store.get_node(edge.source_node_id)
            tgt = store.get_node(edge.target_node_id)
            if not src or not tgt:
                continue

            signature = str(src.properties.get("problem_signature") or src.label or "").strip()
            pattern_id = str(tgt.properties.get("pattern_id") or "").strip()
            if not signature or not pattern_id:
                continue

            s_tokens = _tokens(signature)
            overlap = len(q_tokens & s_tokens)
            if overlap <= 0:
                continue
            score = overlap / max(1, len(q_tokens))
            matches.append(
                GraphResolvedByMatch(
                    problem_signature=signature,
                    pattern_id=pattern_id,
                    overlap=overlap,
                    query_token_count=len(q_tokens),
                    score=score,
                )
            )

        matches.sort(key=lambda m: (-m.score, -m.overlap, m.problem_signature, m.pattern_id))
        return matches[: max(1, int(max_matches))]

