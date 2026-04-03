from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane
from CognitiveRAG.crag.graph_memory.category_graph import (
    categories_for_hit_from_graph,
    decide_query_category_routing,
    record_category_relations_for_hits,
)
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.crag.retrieval import (
    corpus_lane,
    episodic_lane,
    fast_lanes,
    large_file_lane,
    lexical_lane,
    promoted_lane,
    semantic_lane,
    web_lane,
)
from CognitiveRAG.crag.retrieval.models import LaneHit


@dataclass
class RoutePlan:
    intent_family: IntentFamily
    lanes: list[RetrievalLane]
    reason: str
    metadata: dict = field(default_factory=dict)


class LaneRouter:
    """Backend-owned retrieval lane router.

    This is the canonical lane choice owner for selector-bound retrieval.
    """

    def route(self, *, query: str, intent_family: IntentFamily) -> RoutePlan:
        q = (query or "").lower()
        web_sensitive = any(
            token in q for token in ("latest", "current", "today", "yesterday", "recent", "news", "update", "verify")
        )

        # Intent-first routing with query refinements.
        if intent_family == IntentFamily.EXACT_RECALL:
            lanes = [RetrievalLane.EPISODIC, RetrievalLane.LEXICAL, RetrievalLane.SEMANTIC]
            return RoutePlan(intent_family=intent_family, lanes=lanes, reason="exact_recall_prefers_episodic")

        if intent_family == IntentFamily.MEMORY_SUMMARY:
            lanes = [RetrievalLane.PROMOTED, RetrievalLane.EPISODIC, RetrievalLane.SEMANTIC]
            if "about me" in q:
                lanes.insert(0, RetrievalLane.PROMOTED)
            return RoutePlan(intent_family=intent_family, lanes=lanes, reason="memory_summary_prefers_promoted")

        if intent_family == IntentFamily.ARCHITECTURE_EXPLANATION:
            lanes = [RetrievalLane.PROMOTED, RetrievalLane.EPISODIC]
            return RoutePlan(intent_family=intent_family, lanes=lanes, reason="architecture_prefers_layered_memory_artifacts")

        if intent_family == IntentFamily.CORPUS_OVERVIEW:
            lanes = [RetrievalLane.CORPUS, RetrievalLane.LARGE_FILE, RetrievalLane.LEXICAL, RetrievalLane.EPISODIC]
            if web_sensitive:
                lanes.append(RetrievalLane.WEB)
            return RoutePlan(intent_family=intent_family, lanes=lanes, reason="corpus_overview_prefers_corpus_lanes")

        if intent_family == IntentFamily.PLANNING:
            lanes = [RetrievalLane.PROMOTED, RetrievalLane.EPISODIC, RetrievalLane.SEMANTIC]
            if web_sensitive:
                lanes.append(RetrievalLane.WEB)
            return RoutePlan(intent_family=intent_family, lanes=lanes, reason="planning_prefers_promoted_and_recent")

        # INVESTIGATIVE mixed mode.
        lanes = [
            RetrievalLane.SEMANTIC,
            RetrievalLane.LEXICAL,
            RetrievalLane.EPISODIC,
            RetrievalLane.PROMOTED,
            RetrievalLane.CORPUS,
            RetrievalLane.LARGE_FILE,
        ]
        if web_sensitive:
            lanes.append(RetrievalLane.WEB)
        return RoutePlan(intent_family=intent_family, lanes=lanes, reason="investigative_mixed_lane_pool")


LANE_HANDLERS = {
    RetrievalLane.LEXICAL: lambda **kw: lexical_lane.retrieve(
        query=kw["query"],
        session_id=kw["session_id"],
        fresh_tail=kw["fresh_tail"],
        older_raw=kw["older_raw"],
        summaries=kw["summaries"],
        top_k=kw.get("top_k", 8),
    ),
    RetrievalLane.SEMANTIC: lambda **kw: semantic_lane.retrieve(
        query=kw["query"],
        session_id=kw["session_id"],
        fresh_tail=kw["fresh_tail"],
        older_raw=kw["older_raw"],
        summaries=kw["summaries"],
        top_k=kw.get("top_k", 8),
    ),
    RetrievalLane.EPISODIC: lambda **kw: episodic_lane.retrieve(
        session_id=kw["session_id"],
        fresh_tail=kw["fresh_tail"],
        older_raw=kw["older_raw"],
        summaries=kw["summaries"],
        top_k=kw.get("top_k", 20),
    ),
    RetrievalLane.PROMOTED: lambda **kw: promoted_lane.retrieve(
        workdir=kw["workdir"],
        intent_family=kw["intent_family"],
        query=kw["query"],
        top_k=kw.get("top_k", 6),
    ),
    RetrievalLane.CORPUS: lambda **kw: corpus_lane.retrieve(
        workdir=kw["workdir"],
        query=kw["query"],
        top_k=kw.get("top_k", 8),
    ),
    RetrievalLane.LARGE_FILE: lambda **kw: large_file_lane.retrieve(
        workdir=kw["workdir"],
        query=kw["query"],
        top_k=kw.get("top_k", 6),
    ),
    RetrievalLane.WEB: lambda **kw: web_lane.retrieve(
        workdir=kw["workdir"],
        query=kw["query"],
        intent_family=kw["intent_family"],
        top_k=kw.get("top_k", 6),
    ),
}


@dataclass
class _HotCacheEntry:
    plan: RoutePlan
    hits: list[LaneHit]
    expires_at: float


class AgentHotCache:
    def __init__(self, *, max_entries: int = 64, ttl_seconds: float = 45.0):
        self.max_entries = int(max(1, max_entries))
        self.ttl_seconds = float(max(0.001, ttl_seconds))
        self._entries: "OrderedDict[Tuple[str, ...], _HotCacheEntry]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._entries.items() if v.expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)
            self.evictions += 1

    def get(self, key: Tuple[str, ...]) -> _HotCacheEntry | None:
        self._prune()
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry

    def set(self, key: Tuple[str, ...], plan: RoutePlan, hits: list[LaneHit]) -> None:
        self._prune()
        self._entries[key] = _HotCacheEntry(
            plan=copy.deepcopy(plan),
            hits=copy.deepcopy(hits),
            expires_at=(time.monotonic() + self.ttl_seconds),
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def stats(self) -> Dict[str, float | int]:
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
        }


_HOT_CACHE = AgentHotCache()


def get_hot_cache_stats() -> Dict[str, float | int]:
    return _HOT_CACHE.stats()


def clear_hot_cache() -> None:
    _HOT_CACHE.clear()


def _normalized_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _cache_key(
    *,
    query: str,
    intent_family: IntentFamily,
    session_id: str,
    workdir: str,
    top_k_per_lane: int,
) -> Tuple[str, ...]:
    return (
        _normalized_query(query),
        intent_family.value,
        str(session_id or ""),
        str(top_k_per_lane),
        # Guardrails are TTL-bounded and session-scoped; avoid over-invalidation
        # from read-time freshness updates that can legitimately touch sqlite mtime.
        str(workdir),
    )


def _annotate_hot_cache(hits: list[LaneHit], *, hit: bool) -> list[LaneHit]:
    out: list[LaneHit] = []
    for row in hits:
        clone = row.model_copy(deep=True)
        prov = dict(clone.provenance or {})
        fast_path = dict(prov.get("fast_path") or {})
        fast_path["agent_hot_cache_hit"] = bool(hit)
        prov["fast_path"] = fast_path
        clone.provenance = prov
        out.append(clone)
    return out


def _graph_store(workdir: str) -> GraphMemoryStore:
    from os import path

    return GraphMemoryStore(path.join(workdir, "graph_memory.sqlite3"))


def _attach_category_graph_metadata(*, workdir: str, hits: list[LaneHit]) -> list[LaneHit]:
    if not hits:
        return []
    store = _graph_store(workdir)
    # Persist deterministic category edges for reusable routing/filtering.
    record_category_relations_for_hits(store, hits=hits)

    out: list[LaneHit] = []
    for hit in hits:
        clone = hit.model_copy(deep=True)
        categories = categories_for_hit_from_graph(store, clone)
        prov = dict(clone.provenance or {})
        prov["category_graph"] = {
            "categories": categories,
            "category_count": len(categories),
        }
        clone.provenance = prov
        out.append(clone)
    return out


def _category_ids(hit: LaneHit) -> set[str]:
    cg = dict((hit.provenance or {}).get("category_graph") or {})
    rows = list(cg.get("categories") or [])
    return {str(r.get("category") or "") for r in rows if str(r.get("category") or "")}


def route_and_retrieve(
    *,
    query: str,
    intent_family: IntentFamily,
    session_id: str,
    fresh_tail: Iterable[Dict],
    older_raw: Iterable[Dict],
    summaries: Iterable[Dict],
    workdir: str,
    top_k_per_lane: int = 8,
) -> tuple[RoutePlan, list[LaneHit]]:
    key = _cache_key(
        query=query,
        intent_family=intent_family,
        session_id=session_id,
        workdir=workdir,
        top_k_per_lane=top_k_per_lane,
    )
    cached = _HOT_CACHE.get(key)
    if cached is not None:
        plan = copy.deepcopy(cached.plan)
        plan.reason = f"{plan.reason}|agent_hot_cache_hit"
        return plan, _annotate_hot_cache(copy.deepcopy(cached.hits), hit=True)

    router = LaneRouter()
    plan = router.route(query=query, intent_family=intent_family)
    category_decision = decide_query_category_routing(query)
    plan.metadata = {
        **dict(plan.metadata or {}),
        "category_routing": {
            "hinted_categories": list(category_decision.hinted_categories),
            "strong_signal": bool(category_decision.strong_signal),
            "score": float(category_decision.score),
            "reason": category_decision.reason,
            "pruned_lanes": [],
            "fallback_lanes": [],
            "pruned_hit_count": 0,
        },
    }

    hits: list[LaneHit] = []

    # Fast lanes are additive and bounded. They never replace canonical lanes.
    fast_hits = fast_lanes.retrieve_fast_lanes(
        workdir=workdir,
        query=query,
        intent_family=intent_family,
        top_k=top_k_per_lane,
    )
    hits.extend(_attach_category_graph_metadata(workdir=workdir, hits=fast_hits))

    expensive_lanes = {RetrievalLane.CORPUS, RetrievalLane.LARGE_FILE, RetrievalLane.WEB}
    for lane in plan.lanes:
        handler = LANE_HANDLERS.get(lane)
        if handler is None:
            continue
        lane_top_k = int(top_k_per_lane)
        if category_decision.strong_signal and lane in expensive_lanes:
            lane_top_k = max(2, int(top_k_per_lane // 2))
        lane_hits = handler(
            query=query,
            intent_family=intent_family,
            session_id=session_id,
            fresh_tail=fresh_tail,
            older_raw=older_raw,
            summaries=summaries,
            workdir=workdir,
            top_k=lane_top_k,
        )
        lane_hits = _attach_category_graph_metadata(workdir=workdir, hits=list(lane_hits or []))

        if category_decision.strong_signal and lane in expensive_lanes and lane_hits:
            hinted = set(category_decision.hinted_categories)
            matched = [h for h in lane_hits if (_category_ids(h) & hinted)]
            if matched:
                pruned = max(0, len(lane_hits) - len(matched))
                lane_hits = matched
                route_meta = dict(plan.metadata.get("category_routing") or {})
                route_meta["pruned_lanes"] = list(dict.fromkeys([*list(route_meta.get("pruned_lanes") or []), lane.value]))
                route_meta["pruned_hit_count"] = int(route_meta.get("pruned_hit_count") or 0) + int(pruned)
                plan.metadata["category_routing"] = route_meta
            else:
                route_meta = dict(plan.metadata.get("category_routing") or {})
                route_meta["fallback_lanes"] = list(dict.fromkeys([*list(route_meta.get("fallback_lanes") or []), lane.value]))
                plan.metadata["category_routing"] = route_meta

        hits.extend(lane_hits)

    fast_lane_order = {
        RetrievalLane.GLOBAL_PROMOTED: -3,
        RetrievalLane.WORKSPACE_FAST: -2,
        RetrievalLane.INSTALLATION_FAST: -1,
    }
    hits.sort(
        key=lambda h: (
            fast_lane_order.get(h.lane, plan.lanes.index(h.lane) if h.lane in plan.lanes else 999),
            h.id,
        )
    )

    uncached_hits = _annotate_hot_cache(hits, hit=False)
    _HOT_CACHE.set(key, plan=plan, hits=uncached_hits)
    return plan, uncached_hits
