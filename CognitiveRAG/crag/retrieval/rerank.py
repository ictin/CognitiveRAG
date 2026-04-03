from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from CognitiveRAG.crag.contracts.enums import RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit


def _norm_words(text: str) -> set[str]:
    return {w for w in " ".join((text or "").lower().split()).split() if w}


def _overlap_ratio(query_words: set[str], text: str) -> float:
    if not query_words:
        return 0.0
    words = _norm_words(text)
    if not words:
        return 0.0
    return float(len(query_words & words)) / float(max(1, len(query_words)))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


@dataclass(frozen=True)
class RerankResult:
    hits: list[LaneHit]
    metadata: dict


def rerank_hits(
    *,
    query: str,
    hits: Sequence[LaneHit],
    plan_lanes: Iterable[RetrievalLane],
    hinted_categories: Iterable[str],
    category_strong: bool,
) -> RerankResult:
    if len(hits) <= 1:
        return RerankResult(
            hits=list(hits),
            metadata={
                "applied": False,
                "strategy": "bounded_signal_rerank_v1",
                "reason": "insufficient_hits",
                "moved_count": 0,
            },
        )

    query_words = _norm_words(query)
    if not query_words:
        return RerankResult(
            hits=list(hits),
            metadata={
                "applied": False,
                "strategy": "bounded_signal_rerank_v1",
                "reason": "empty_query_terms",
                "moved_count": 0,
            },
        )

    lane_order = list(plan_lanes)
    hinted = {c for c in hinted_categories if c}
    scored: list[tuple[float, str, LaneHit]] = []
    explanation_rows: list[dict] = []

    for idx, hit in enumerate(list(hits)):
        lane_index = lane_order.index(hit.lane) if hit.lane in lane_order else 999
        lane_priority = _clamp(1.0 - (lane_index * 0.07), 0.0, 1.0)
        base = (
            0.45 * float(hit.semantic_score or 0.0)
            + 0.35 * float(hit.lexical_score or 0.0)
            + 0.20 * float(hit.trust_score or 0.0)
        )
        lexical_fit = _overlap_ratio(query_words, hit.text)

        prov = dict(hit.provenance or {})
        category_rows = list(dict(prov.get("category_graph") or {}).get("categories") or [])
        category_ids = {str(r.get("category") or "") for r in category_rows if str(r.get("category") or "")}
        category_fit = 1.0 if (category_strong and hinted and (category_ids & hinted)) else 0.0

        graph_support_count = int(prov.get("graph_support_count") or 0)
        if not graph_support_count:
            graph_support_count = len(list(prov.get("supported_by_urls") or []))
        graph_support = _clamp(graph_support_count / 3.0, 0.0, 1.0)

        reuse_count = int(prov.get("reuse_count") or 0)
        reuse_signal = _clamp((reuse_count - 1) / 6.0, 0.0, 1.0)
        success_conf = _clamp(float(prov.get("success_confidence") or 0.0), 0.0, 1.0)

        freshness = str(prov.get("freshness_lifecycle_state") or "")
        freshness_penalty = 0.0
        if freshness == "stale":
            freshness_penalty = 1.0
        elif freshness == "revalidation_pending":
            freshness_penalty = 0.6

        contradiction_info = dict(prov.get("contradiction") or {})
        contradiction_penalty = 1.0 if (bool(contradiction_info.get("has_contradiction")) or float(hit.contradiction_risk or 0.0) >= 0.4) else 0.0

        tier = str(prov.get("promotion_tier") or "")
        tier_bonus = 1.0 if tier == "global" else (0.5 if tier == "workspace" else 0.0)

        helper = (
            0.10 * lexical_fit
            + 0.08 * category_fit
            + 0.05 * graph_support
            + 0.04 * reuse_signal
            + 0.03 * success_conf
            + 0.03 * tier_bonus
            - 0.08 * freshness_penalty
            - 0.10 * contradiction_penalty
        )
        helper = _clamp(helper, -0.22, 0.22)
        final_score = (0.55 * base) + (0.25 * lane_priority) + helper

        clone = hit.model_copy(deep=True)
        clone_prov = dict(clone.provenance or {})
        clone_prov["rerank"] = {
            "strategy": "bounded_signal_rerank_v1",
            "base_score": round(float(base), 6),
            "lane_priority": round(float(lane_priority), 6),
            "lexical_fit": round(float(lexical_fit), 6),
            "category_fit": round(float(category_fit), 6),
            "graph_support": round(float(graph_support), 6),
            "reuse_signal": round(float(reuse_signal), 6),
            "success_confidence": round(float(success_conf), 6),
            "tier_bonus": round(float(tier_bonus), 6),
            "freshness_penalty": round(float(freshness_penalty), 6),
            "contradiction_penalty": round(float(contradiction_penalty), 6),
            "helper_adjustment": round(float(helper), 6),
            "final_score": round(float(final_score), 6),
        }
        clone.provenance = clone_prov
        scored.append((final_score, clone.id, clone))
        explanation_rows.append({"id": clone.id, "score": round(float(final_score), 6), "lane": clone.lane.value, "before_index": idx})

    before = [h.id for h in hits]
    scored.sort(key=lambda row: (-float(row[0]), row[1]))
    reranked = [row[2] for row in scored]
    after = [h.id for h in reranked]
    moved = sum(1 for idx, hid in enumerate(before) if idx >= len(after) or hid != after[idx])

    weak_signal = all(abs(float(row[0])) < 0.25 for row in scored)
    if weak_signal:
        return RerankResult(
            hits=list(hits),
            metadata={
                "applied": False,
                "strategy": "bounded_signal_rerank_v1",
                "reason": "weak_signal_fallback",
                "moved_count": 0,
                "top_scores": explanation_rows[:5],
            },
        )

    return RerankResult(
        hits=reranked,
        metadata={
            "applied": True,
            "strategy": "bounded_signal_rerank_v1",
            "reason": "applied",
            "moved_count": moved,
            "top_scores": explanation_rows[:5],
        },
    )

