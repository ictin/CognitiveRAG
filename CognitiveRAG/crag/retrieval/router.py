from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.retrieval import corpus_lane, episodic_lane, large_file_lane, lexical_lane, promoted_lane, semantic_lane, web_lane


@dataclass
class RoutePlan:
    intent_family: IntentFamily
    lanes: list[RetrievalLane]
    reason: str


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
    router = LaneRouter()
    plan = router.route(query=query, intent_family=intent_family)

    hits: list[LaneHit] = []
    for lane in plan.lanes:
        handler = LANE_HANDLERS.get(lane)
        if handler is None:
            continue
        lane_hits = handler(
            query=query,
            intent_family=intent_family,
            session_id=session_id,
            fresh_tail=fresh_tail,
            older_raw=older_raw,
            summaries=summaries,
            workdir=workdir,
            top_k=top_k_per_lane,
        )
        hits.extend(list(lane_hits or []))

    # Deterministic order before selector scoring.
    hits.sort(key=lambda h: (plan.lanes.index(h.lane) if h.lane in plan.lanes else 999, h.id))
    return plan, hits
