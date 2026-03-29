from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Set

from CognitiveRAG.crag.contracts.schemas import ContextCandidate


_HEDGE_TOKENS = {
    'maybe',
    'might',
    'unclear',
    'unknown',
    'possible',
    'possibly',
    'uncertain',
    'approximately',
}


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r'[^a-z0-9]+', str(text or '').lower()) if len(tok) >= 3]


def novelty_score(*, text: str, seen_tokens: Set[str]) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    unseen = [tok for tok in tokens if tok not in seen_tokens]
    return len(unseen) / float(len(tokens))


def uncertainty_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    hedges = sum(1 for tok in tokens if tok in _HEDGE_TOKENS)
    return min(1.0, hedges / float(max(1, len(tokens) // 4)))


@dataclass(frozen=True)
class CuriosityWeights:
    relevance: float = 0.35
    novelty: float = 0.3
    uncertainty: float = 0.15
    contradiction_value: float = 0.1
    frontier: float = 0.1


def score_candidate_curiosity(
    candidate: ContextCandidate,
    *,
    query: str,
    seen_tokens: Set[str],
    weights: CuriosityWeights = CuriosityWeights(),
) -> float:
    text = candidate.text
    query_tokens = set(tokenize(query))
    cand_tokens = set(tokenize(text))
    if query_tokens:
        relevance = len(query_tokens & cand_tokens) / float(len(query_tokens))
    else:
        relevance = 0.0

    novelty = novelty_score(text=text, seen_tokens=seen_tokens)
    uncertainty = uncertainty_score(text)
    contradiction_value = min(1.0, float(candidate.contradiction_risk))
    frontier = 1.0 if candidate.cluster_id and candidate.cluster_id not in seen_tokens else 0.0

    score = (
        weights.relevance * relevance
        + weights.novelty * novelty
        + weights.uncertainty * uncertainty
        + weights.contradiction_value * contradiction_value
        + weights.frontier * frontier
    )
    return round(max(0.0, min(1.0, score)), 6)


def update_seen_tokens(seen_tokens: Set[str], texts: Iterable[str]) -> Set[str]:
    out = set(seen_tokens)
    for text in texts:
        out.update(tokenize(text))
    return out
