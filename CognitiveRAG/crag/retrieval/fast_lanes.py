from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _promoted_db_path(workdir: str) -> str:
    return os.path.join(workdir, "web_promoted_memory.sqlite3")


def _fast_lane_for_tier(tier: str) -> RetrievalLane:
    if tier == WebPromotedMemoryStore.TIER_GLOBAL:
        return RetrievalLane.GLOBAL_PROMOTED
    if tier == WebPromotedMemoryStore.TIER_WORKSPACE:
        return RetrievalLane.WORKSPACE_FAST
    return RetrievalLane.INSTALLATION_FAST


def _build_hit(item: Dict[str, Any], *, lane: RetrievalLane) -> LaneHit | None:
    text = str(item.get("canonical_fact") or "").strip()
    if not text:
        return None

    tier = str(item.get("promotion_tier") or WebPromotedMemoryStore.TIER_LOCAL)
    state = str(item.get("promotion_state") or WebPromotedMemoryStore.STATE_STAGED)
    lifecycle = str(item.get("freshness_lifecycle_state") or WebPromotedMemoryStore.FRESHNESS_STALE)
    contradiction = item.get("contradiction") or {}
    has_contradiction = bool(contradiction.get("has_contradiction"))

    fast_lane_safe = (
        state == WebPromotedMemoryStore.STATE_TRUSTED
        and lifecycle == WebPromotedMemoryStore.FRESHNESS_FRESH
        and not has_contradiction
    )

    trust_adjust = 0.0
    semantic_adjust = 0.0
    if fast_lane_safe:
        trust_adjust += 0.05
        semantic_adjust += 0.03
    if tier == WebPromotedMemoryStore.TIER_GLOBAL:
        trust_adjust += 0.03
        semantic_adjust += 0.02
    elif tier == WebPromotedMemoryStore.TIER_WORKSPACE:
        trust_adjust += 0.015
        semantic_adjust += 0.01

    if lifecycle == WebPromotedMemoryStore.FRESHNESS_REVALIDATION_PENDING:
        trust_adjust -= 0.08
        semantic_adjust -= 0.03
    elif lifecycle == WebPromotedMemoryStore.FRESHNESS_STALE:
        trust_adjust -= 0.10
        semantic_adjust -= 0.04

    if has_contradiction:
        trust_adjust -= 0.12
        semantic_adjust -= 0.05

    confidence = float(item.get("confidence") or 0.5)

    provenance = {
        "promoted_id": item.get("promoted_id"),
        "promotion_tier": tier,
        "origin_tier": item.get("origin_tier") or tier,
        "promotion_state": state,
        "freshness_lifecycle_state": lifecycle,
        "freshness_reason": item.get("freshness_reason") or "",
        "last_validated_at": item.get("last_validated_at"),
        "revalidation_requested_at": item.get("revalidation_requested_at"),
        "promoted_from_ids": item.get("promoted_from_ids") or [],
        "promotion_basis": item.get("promotion_basis") or {},
        "promotion_history": item.get("promotion_history") or [],
        "contradiction": contradiction,
        "metadata": item.get("metadata") or {},
        "source_class": "web_promoted",
        "fast_lane": {
            "lane": lane.value,
            "tier": tier,
            "cache_safe": fast_lane_safe,
        },
    }

    return LaneHit(
        id=f"{lane.value}:{item.get('promoted_id')}",
        lane=lane,
        memory_type=MemoryType.WEB_PROMOTED_FACT,
        text=text,
        provenance=provenance,
        lexical_score=0.5,
        semantic_score=max(0.0, min(1.0, 0.62 + semantic_adjust)),
        recency_score=0.45,
        freshness_score=0.75 if lifecycle == WebPromotedMemoryStore.FRESHNESS_FRESH else 0.45,
        trust_score=max(0.0, min(1.0, confidence + trust_adjust)),
        novelty_score=0.3,
        contradiction_risk=(0.7 if has_contradiction else 0.1),
        cluster_id=f"fast_lane:{tier}:{state}:{lifecycle}",
        compressible=True,
    ).with_token_estimate()


def retrieve_fast_lanes(
    *,
    workdir: str,
    query: str,
    intent_family: IntentFamily,
    top_k: int = 8,
) -> List[LaneHit]:
    # Keep fast lanes additive and bounded for memory-oriented routes.
    if intent_family not in {
        IntentFamily.MEMORY_SUMMARY,
        IntentFamily.PLANNING,
        IntentFamily.INVESTIGATIVE,
        IntentFamily.EXACT_RECALL,
    }:
        return []

    store = WebPromotedMemoryStore(_promoted_db_path(workdir))
    search_rows = store.search(query, top_k=max(4, top_k * 3))
    if not search_rows:
        return []

    rows: List[Dict[str, Any]] = []
    now = _now_iso()
    for row in search_rows:
        promoted_id = str(row.get("promoted_id") or "")
        if not promoted_id:
            continue
        refreshed = store.evaluate_freshness(promoted_id, now_iso=now) or row
        refreshed = dict(refreshed)
        refreshed["contradiction"] = store.get_contradiction_summary(promoted_id)
        rows.append(refreshed)

    lane_buckets: Dict[RetrievalLane, List[LaneHit]] = {
        RetrievalLane.INSTALLATION_FAST: [],
        RetrievalLane.WORKSPACE_FAST: [],
        RetrievalLane.GLOBAL_PROMOTED: [],
    }
    for row in rows:
        lane = _fast_lane_for_tier(str(row.get("promotion_tier") or WebPromotedMemoryStore.TIER_LOCAL))
        hit = _build_hit(row, lane=lane)
        if hit is not None:
            lane_buckets[lane].append(hit)

    # Keep each fast lane bounded to preserve selector balance.
    lane_cap = max(1, top_k // 2)
    ordered: List[LaneHit] = []
    for lane in (RetrievalLane.GLOBAL_PROMOTED, RetrievalLane.WORKSPACE_FAST, RetrievalLane.INSTALLATION_FAST):
        items = lane_buckets[lane]
        items.sort(key=lambda h: (-float(h.trust_score or 0.0), -float(h.semantic_score or 0.0), h.id))
        ordered.extend(items[:lane_cap])

    return ordered
