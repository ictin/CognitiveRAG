from __future__ import annotations

from typing import Iterable

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, ContextSelectionPolicy


def _intent_fit(candidate: ContextCandidate, intent: IntentFamily) -> float:
    lane = candidate.lane
    mtype = candidate.memory_type

    if intent == IntentFamily.EXACT_RECALL:
        if lane in {RetrievalLane.EPISODIC, RetrievalLane.FRESH_TAIL}:
            return 1.0
        if mtype == MemoryType.SUMMARY:
            return 0.1
        return 0.2

    if intent == IntentFamily.MEMORY_SUMMARY:
        if mtype in {MemoryType.SUMMARY, MemoryType.PROMOTED_FACT, MemoryType.ARCHITECTURE_NOTE}:
            return 1.0
        if lane == RetrievalLane.EPISODIC:
            return 0.35
        return 0.5

    if intent == IntentFamily.ARCHITECTURE_EXPLANATION:
        return 1.0 if lane == RetrievalLane.ARCHITECTURE else 0.2

    if intent == IntentFamily.CORPUS_OVERVIEW:
        if lane in {RetrievalLane.CORPUS, RetrievalLane.LARGE_FILE}:
            return 1.0
        if lane == RetrievalLane.FRESH_TAIL:
            return 0.35
        return 0.2

    if intent == IntentFamily.PLANNING:
        if lane in {RetrievalLane.FRESH_TAIL, RetrievalLane.EPISODIC}:
            return 1.0
        if lane == RetrievalLane.LEXICAL:
            return 0.8
        if lane == RetrievalLane.PROMOTED:
            return 0.65
        if lane == RetrievalLane.SESSION_SUMMARY:
            return 0.25
        return 0.35

    # investigative
    if lane in {RetrievalLane.CORPUS, RetrievalLane.LARGE_FILE, RetrievalLane.PROMOTED, RetrievalLane.EPISODIC}:
        return 0.8
    return 0.45


def redundancy_penalty(candidate: ContextCandidate, selected: Iterable[ContextCandidate]) -> float:
    c_words = set((candidate.text or "").lower().split())
    if not c_words:
        return 0.0
    max_overlap = 0.0
    for existing in selected:
        e_words = set((existing.text or "").lower().split())
        if not e_words:
            continue
        overlap = float(len(c_words & e_words)) / float(max(1, len(c_words)))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def score_candidate(
    candidate: ContextCandidate,
    selected: Iterable[ContextCandidate],
    policy: ContextSelectionPolicy,
    intent: IntentFamily,
    cluster_unseen_bonus: bool,
) -> float:
    weights = policy.per_intent_weights

    relevance = (candidate.lexical_score + candidate.semantic_score) / 2.0
    provenance = 1.0 if candidate.provenance else 0.2
    recency = candidate.recency_score
    freshness_trust = (candidate.freshness_score + candidate.trust_score) / 2.0
    novelty = candidate.novelty_score
    intent_fit = _intent_fit(candidate, intent)

    redundancy = redundancy_penalty(candidate, selected)
    contradiction = candidate.contradiction_risk

    utility = (
        (weights.relevance * relevance)
        + (weights.provenance * provenance)
        + (weights.recency * recency)
        + (weights.freshness_trust * freshness_trust)
        + (weights.novelty * novelty)
        + (weights.intent_fit * intent_fit)
        - (weights.redundancy_penalty * redundancy)
        - (weights.contradiction_penalty * contradiction)
    )

    if cluster_unseen_bonus and candidate.cluster_id:
        utility += policy.cluster_bonus

    return float(utility)
