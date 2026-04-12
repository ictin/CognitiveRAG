from __future__ import annotations

from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextSelectionPolicy, IntentWeights


def _base_policy(intent: IntentFamily) -> ContextSelectionPolicy:
    return ContextSelectionPolicy(
        intent_family=intent,
        hard_reservation_tokens=256,
        minimal_fresh_tail=3,
        lane_minima={
            RetrievalLane.FRESH_TAIL.value: 2,
        },
        lane_maxima={
            RetrievalLane.LEXICAL.value: 8,
            RetrievalLane.SEMANTIC.value: 8,
            RetrievalLane.FRESH_TAIL.value: 10,
            RetrievalLane.EPISODIC.value: 8,
            RetrievalLane.SESSION_SUMMARY.value: 8,
            RetrievalLane.PROMOTED.value: 4,
            RetrievalLane.CORPUS.value: 8,
            RetrievalLane.LARGE_FILE.value: 6,
            RetrievalLane.REASONING.value: 4,
            RetrievalLane.ARCHITECTURE.value: 4,
            RetrievalLane.FALLBACK_MIRROR.value: 4,
        },
        cluster_bonus=0.2,
        redundancy_penalty=0.65,
        contradiction_penalty=0.8,
        front_anchor_budget=2,
        back_anchor_budget=2,
        per_intent_weights=IntentWeights(),
    )


def get_policy(intent_family: IntentFamily | str) -> ContextSelectionPolicy:
    intent = IntentFamily(intent_family)
    policy = _base_policy(intent)

    if intent == IntentFamily.EXACT_RECALL:
        policy.per_intent_weights = IntentWeights(
            relevance=1.25,
            provenance=0.8,
            recency=0.65,
            freshness_trust=0.35,
            novelty=0.2,
            intent_fit=1.1,
            redundancy_penalty=0.75,
            contradiction_penalty=0.8,
        )
        # Exact recall should prioritize quote-ready episodic evidence and avoid
        # over-anchoring to unrelated fresh tail / compressed summaries.
        policy.lane_minima[RetrievalLane.FRESH_TAIL.value] = 1
        policy.lane_minima[RetrievalLane.EPISODIC.value] = 2
        policy.lane_maxima[RetrievalLane.SESSION_SUMMARY.value] = 0
        policy.lane_maxima[RetrievalLane.CORPUS.value] = 3
        policy.lane_maxima[RetrievalLane.LEXICAL.value] = 4
        policy.lane_maxima[RetrievalLane.SEMANTIC.value] = 4

    elif intent == IntentFamily.MEMORY_SUMMARY:
        policy.per_intent_weights = IntentWeights(
            relevance=0.9,
            provenance=0.95,
            recency=0.45,
            freshness_trust=0.75,
            novelty=0.55,
            intent_fit=1.1,
            redundancy_penalty=0.9,
            contradiction_penalty=0.85,
        )
        policy.lane_minima[RetrievalLane.SESSION_SUMMARY.value] = 2
        policy.lane_minima[RetrievalLane.PROMOTED.value] = 1
        policy.lane_maxima[RetrievalLane.EPISODIC.value] = 3
        policy.lane_maxima[RetrievalLane.LEXICAL.value] = 3

    elif intent == IntentFamily.ARCHITECTURE_EXPLANATION:
        policy.per_intent_weights = IntentWeights(
            relevance=0.85,
            provenance=1.1,
            recency=0.3,
            freshness_trust=0.7,
            novelty=0.25,
            intent_fit=1.3,
            redundancy_penalty=0.85,
            contradiction_penalty=0.95,
        )
        policy.lane_minima[RetrievalLane.ARCHITECTURE.value] = 1

    elif intent == IntentFamily.CORPUS_OVERVIEW:
        policy.per_intent_weights = IntentWeights(
            relevance=1.2,
            provenance=1.0,
            recency=0.35,
            freshness_trust=0.45,
            novelty=0.3,
            intent_fit=1.35,
            redundancy_penalty=0.7,
            contradiction_penalty=0.8,
        )
        policy.lane_minima[RetrievalLane.CORPUS.value] = 1
        policy.lane_maxima[RetrievalLane.EPISODIC.value] = 2
        policy.lane_maxima[RetrievalLane.LEXICAL.value] = 3

    elif intent == IntentFamily.PLANNING:
        policy.per_intent_weights = IntentWeights(
            relevance=1.05,
            provenance=0.85,
            recency=0.8,
            freshness_trust=0.55,
            novelty=0.45,
            intent_fit=1.15,
            redundancy_penalty=0.7,
            contradiction_penalty=0.8,
        )
        policy.lane_minima[RetrievalLane.EPISODIC.value] = 1
        policy.lane_maxima[RetrievalLane.SESSION_SUMMARY.value] = 0
        policy.lane_maxima[RetrievalLane.CORPUS.value] = 2
        policy.lane_maxima[RetrievalLane.LARGE_FILE.value] = 1
        policy.lane_maxima[RetrievalLane.PROMOTED.value] = 2

    elif intent == IntentFamily.INVESTIGATIVE:
        policy.per_intent_weights = IntentWeights(
            relevance=1.15,
            provenance=0.9,
            recency=0.4,
            freshness_trust=0.6,
            novelty=0.75,
            intent_fit=1.0,
            redundancy_penalty=0.6,
            contradiction_penalty=0.9,
        )

    return policy
