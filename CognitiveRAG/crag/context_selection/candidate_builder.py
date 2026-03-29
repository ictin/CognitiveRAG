from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.retrieval.router import RoutePlan, route_and_retrieve


def _candidate_from_lane_hit(hit: LaneHit) -> ContextCandidate:
    return ContextCandidate(
        id=hit.id,
        lane=hit.lane,
        memory_type=hit.memory_type,
        text=hit.text,
        tokens=hit.tokens,
        provenance=hit.provenance,
        lexical_score=hit.lexical_score,
        semantic_score=hit.semantic_score,
        recency_score=hit.recency_score,
        freshness_score=hit.freshness_score,
        trust_score=hit.trust_score,
        novelty_score=hit.novelty_score,
        contradiction_risk=hit.contradiction_risk,
        cluster_id=hit.cluster_id,
        must_include=hit.must_include,
        compressible=hit.compressible,
    )


def _inject_architecture_candidate(query: str) -> ContextCandidate:
    return ContextCandidate(
        id="architecture:stack",
        lane=RetrievalLane.ARCHITECTURE,
        memory_type=MemoryType.ARCHITECTURE_NOTE,
        text=(
            "Memory stack: active context engine + backend/session memory + "
            "lossless session memory + corpus/large-file retrieval + markdown mirror fallback."
        ),
        provenance={"source": "backend_contract", "query": query},
        lexical_score=0.2,
        semantic_score=0.35,
        recency_score=0.5,
        freshness_score=0.9,
        trust_score=0.95,
        novelty_score=0.45,
        contradiction_risk=0.0,
        cluster_id="architecture",
        must_include=False,
        compressible=False,
    )


def build_candidates_with_route(
    *,
    session_id: str,
    query: str,
    fresh_tail: Iterable[Dict[str, Any]],
    older_raw: Iterable[Dict[str, Any]],
    summaries: Iterable[Dict[str, Any]],
    workdir: str,
    intent_family: IntentFamily,
) -> tuple[RoutePlan, List[ContextCandidate]]:
    plan, lane_hits = route_and_retrieve(
        query=query,
        intent_family=intent_family,
        session_id=session_id,
        fresh_tail=fresh_tail,
        older_raw=older_raw,
        summaries=summaries,
        workdir=workdir,
    )

    candidates = [_candidate_from_lane_hit(hit) for hit in lane_hits]

    if intent_family in {IntentFamily.ARCHITECTURE_EXPLANATION, IntentFamily.MEMORY_SUMMARY}:
        candidates.append(_inject_architecture_candidate(query))

    return plan, candidates


def build_candidates(
    *,
    session_id: str,
    query: str,
    fresh_tail: Iterable[Dict[str, Any]],
    older_raw: Iterable[Dict[str, Any]],
    summaries: Iterable[Dict[str, Any]],
    workdir: str,
    intent_family: IntentFamily,
) -> List[ContextCandidate]:
    _, candidates = build_candidates_with_route(
        session_id=session_id,
        query=query,
        fresh_tail=fresh_tail,
        older_raw=older_raw,
        summaries=summaries,
        workdir=workdir,
        intent_family=intent_family,
    )
    return candidates
