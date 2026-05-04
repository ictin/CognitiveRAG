from __future__ import annotations

import copy
import os
import re
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
from CognitiveRAG.crag.retrieval.rerank import rerank_hits


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
            lanes = [RetrievalLane.EPISODIC, RetrievalLane.LEXICAL, RetrievalLane.PROMOTED, RetrievalLane.SEMANTIC]
            if web_sensitive:
                lanes.append(RetrievalLane.WEB)
            return RoutePlan(intent_family=intent_family, lanes=lanes, reason="planning_prefers_session_grounded_state")

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
        query=kw["query"],
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


@dataclass
class _RouteCacheEntry:
    plan: RoutePlan
    expires_at: float


class RouteDecisionCache:
    def __init__(self, *, max_entries: int = 128, ttl_seconds: float = 90.0):
        self.max_entries = int(max(1, max_entries))
        self.ttl_seconds = float(max(0.001, ttl_seconds))
        self._entries: "OrderedDict[Tuple[str, ...], _RouteCacheEntry]" = OrderedDict()
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

    def get(self, key: Tuple[str, ...]) -> _RouteCacheEntry | None:
        self._prune()
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry

    def set(self, key: Tuple[str, ...], plan: RoutePlan) -> None:
        self._prune()
        self._entries[key] = _RouteCacheEntry(
            plan=copy.deepcopy(plan),
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


@dataclass
class _ShortlistCacheEntry:
    hinted_categories: tuple[str, ...]
    strong_signal: bool
    score: float
    reason: str
    expires_at: float


class TopicShortlistCache:
    def __init__(self, *, max_entries: int = 128, ttl_seconds: float = 90.0):
        self.max_entries = int(max(1, max_entries))
        self.ttl_seconds = float(max(0.001, ttl_seconds))
        self._entries: "OrderedDict[Tuple[str, ...], _ShortlistCacheEntry]" = OrderedDict()
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

    def get(self, key: Tuple[str, ...]) -> _ShortlistCacheEntry | None:
        self._prune()
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry

    def set(self, key: Tuple[str, ...], *, hinted_categories: tuple[str, ...], strong_signal: bool, score: float, reason: str) -> None:
        self._prune()
        self._entries[key] = _ShortlistCacheEntry(
            hinted_categories=tuple(hinted_categories),
            strong_signal=bool(strong_signal),
            score=float(score),
            reason=str(reason),
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


_ROUTE_CACHE = RouteDecisionCache()
_TOPIC_SHORTLIST_CACHE = TopicShortlistCache()


def get_hot_cache_stats() -> Dict[str, float | int]:
    return _HOT_CACHE.stats()


def clear_hot_cache() -> None:
    _HOT_CACHE.clear()


def get_route_cache_stats() -> Dict[str, float | int]:
    return _ROUTE_CACHE.stats()


def clear_route_cache() -> None:
    _ROUTE_CACHE.clear()


def get_topic_shortlist_cache_stats() -> Dict[str, float | int]:
    return _TOPIC_SHORTLIST_CACHE.stats()


def clear_topic_shortlist_cache() -> None:
    _TOPIC_SHORTLIST_CACHE.clear()


def clear_routing_caches() -> None:
    clear_hot_cache()
    clear_route_cache()
    clear_topic_shortlist_cache()


_WORD_RE = re.compile(r"\W+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}


def _normalized_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _query_signature(query: str, *, max_terms: int = 12) -> tuple[str, ...]:
    normalized = _normalized_query(query)
    terms = [t for t in _WORD_RE.split(normalized) if t and t not in _STOPWORDS]
    # near-repeated queries share sorted term signatures
    return tuple(sorted(dict.fromkeys(terms))[: max(1, int(max_terms))])


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


def _route_cache_key(*, query: str, intent_family: IntentFamily) -> tuple[str, ...]:
    return (intent_family.value, *_query_signature(query))


def _shortlist_cache_key(*, query: str) -> tuple[str, ...]:
    return _query_signature(query)


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
    if os.getenv("CRAG_DISABLE_CATEGORY_GRAPH", "").strip() == "1":
        return [hit.model_copy(deep=True) for hit in hits]
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

    shortlist_key = _shortlist_cache_key(query=query)
    shortlist_cached = _TOPIC_SHORTLIST_CACHE.get(shortlist_key)
    if shortlist_cached is not None:
        category_hints = tuple(shortlist_cached.hinted_categories)
        category_strong = bool(shortlist_cached.strong_signal)
        category_score = float(shortlist_cached.score)
        category_reason = str(shortlist_cached.reason)
        shortlist_hit = True
    else:
        category_decision = decide_query_category_routing(query)
        category_hints = tuple(category_decision.hinted_categories)
        category_strong = bool(category_decision.strong_signal)
        category_score = float(category_decision.score)
        category_reason = str(category_decision.reason)
        _TOPIC_SHORTLIST_CACHE.set(
            shortlist_key,
            hinted_categories=category_hints,
            strong_signal=category_strong,
            score=category_score,
            reason=category_reason,
        )
        shortlist_hit = False

    route_key = _route_cache_key(query=query, intent_family=intent_family)
    route_cached = _ROUTE_CACHE.get(route_key)
    if route_cached is not None:
        plan = copy.deepcopy(route_cached.plan)
        plan.reason = f"{plan.reason}|route_cache_hit"
        route_hit = True
    else:
        router = LaneRouter()
        plan = router.route(query=query, intent_family=intent_family)
        _ROUTE_CACHE.set(
            route_key,
            RoutePlan(
                intent_family=plan.intent_family,
                lanes=list(plan.lanes),
                reason=plan.reason,
                metadata={},
            ),
        )
        route_hit = False

    plan.metadata = {
        **dict(plan.metadata or {}),
        "category_routing": {
            "helper_enabled": os.getenv("CRAG_DISABLE_CATEGORY_GRAPH", "").strip() != "1",
            "hinted_categories": list(category_hints),
            "strong_signal": bool(category_strong),
            "score": float(category_score),
            "reason": category_reason,
            "pruned_lanes": [],
            "fallback_lanes": [],
            "pruned_hit_count": 0,
            "shortlist_cache": {
                "hit": bool(shortlist_hit),
                "key_terms": list(shortlist_key),
                "reason": "cache_hit" if shortlist_hit else "cache_miss_computed",
            },
        },
        "route_cache": {
            "hit": bool(route_hit),
            "key_terms": list(route_key),
            "reason": "cache_hit" if route_hit else "cache_miss_computed",
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
        if category_strong and lane in expensive_lanes:
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

        if category_strong and lane in expensive_lanes and lane_hits:
            hinted = set(category_hints)
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
    # F-017: hot-path safe optimization.
    # Preserve deterministic semantics while avoiding repeated O(n) lane index lookups.
    if os.getenv("CRAG_F017_LEGACY_SORT", "").strip() == "1":
        hits.sort(
            key=lambda h: (
                fast_lane_order.get(h.lane, plan.lanes.index(h.lane) if h.lane in plan.lanes else 999),
                h.id,
            )
        )
    else:
        lane_rank = {lane: idx for idx, lane in enumerate(plan.lanes)}
        hits.sort(
            key=lambda h: (
                fast_lane_order.get(h.lane, lane_rank.get(h.lane, 999)),
                h.id,
            )
        )

    rerank = rerank_hits(
        query=query,
        hits=hits,
        plan_lanes=plan.lanes,
        hinted_categories=category_hints,
        category_strong=bool(category_strong),
    )
    hits = list(rerank.hits)
    plan.metadata = {
        **dict(plan.metadata or {}),
        "rerank": dict(rerank.metadata or {}),
    }

    uncached_hits = _annotate_hot_cache(hits, hit=False)
    _HOT_CACHE.set(key, plan=plan, hits=uncached_hits)
    return plan, uncached_hits
