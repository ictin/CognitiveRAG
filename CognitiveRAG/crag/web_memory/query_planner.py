from __future__ import annotations

from dataclasses import dataclass
from typing import List

from CognitiveRAG.crag.contracts.enums import IntentFamily


_FRESHNESS_TERMS = {
    "latest", "current", "today", "yesterday", "recent", "recently",
    "news", "update", "updated", "version", "release", "now",
}
_VERIFY_TERMS = {"verify", "confirm", "fact-check", "fact check", "source", "citation"}


@dataclass
class WebNeedDecision:
    needed: bool
    reason: str
    freshness_sensitive: bool
    volatility: str


@dataclass
class WebQueryPlan:
    query: str
    variants: List[str]
    source_hints: List[str]
    freshness_sensitive: bool
    max_results: int


def detect_web_need(
    *,
    query: str,
    intent_family: IntentFamily,
    local_evidence_count: int = 0,
) -> WebNeedDecision:
    q = (query or "").lower()
    freshness_sensitive = any(term in q for term in _FRESHNESS_TERMS)
    verification_sensitive = any(term in q for term in _VERIFY_TERMS)

    if freshness_sensitive:
        return WebNeedDecision(
            needed=True,
            reason="freshness_sensitive_query",
            freshness_sensitive=True,
            volatility="high",
        )

    if verification_sensitive and local_evidence_count <= 1:
        return WebNeedDecision(
            needed=True,
            reason="verification_sensitive_with_sparse_local_evidence",
            freshness_sensitive=False,
            volatility="medium",
        )

    if intent_family == IntentFamily.INVESTIGATIVE and local_evidence_count == 0:
        return WebNeedDecision(
            needed=True,
            reason="investigative_query_with_no_local_evidence",
            freshness_sensitive=False,
            volatility="medium",
        )

    return WebNeedDecision(
        needed=False,
        reason="local_memory_preferred",
        freshness_sensitive=False,
        volatility="low",
    )


def plan_web_queries(
    *,
    query: str,
    decision: WebNeedDecision,
    max_variants: int = 3,
) -> WebQueryPlan:
    base = " ".join((query or "").split())
    variants: List[str] = [base] if base else []
    lowered = base.lower()

    if decision.freshness_sensitive and base:
        variants.append(f"{base} latest update")
    if "synopsis" in lowered:
        variants.append(base.replace("synopsis", "summary"))
    if "what does" in lowered and "say" in lowered:
        variants.append(base.replace("what does", "key points from"))

    unique: List[str] = []
    for item in variants:
        if item and item not in unique:
            unique.append(item)
    variants = unique[: max(1, int(max_variants))]

    source_hints = ["official_docs", "publisher", "primary_source"]
    if decision.freshness_sensitive:
        source_hints.insert(0, "recent_sources")

    return WebQueryPlan(
        query=base,
        variants=variants,
        source_hints=source_hints,
        freshness_sensitive=decision.freshness_sensitive,
        max_results=5 if decision.freshness_sensitive else 3,
    )
