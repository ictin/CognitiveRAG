from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import List

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.graph_memory.enrichment import GraphRetrievalEnricher
from CognitiveRAG.crag.lifecycle.normalization import normalized_lifecycle_view
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.web_memory.evidence_store import WebEvidenceStore
from CognitiveRAG.crag.web_memory.fetch import WebFetcher
from CognitiveRAG.crag.web_memory.fetch_log import WebFetchLogStore
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore
from CognitiveRAG.crag.web_memory.query_planner import detect_web_need, plan_web_queries


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _db_paths(workdir: str):
    return (
        os.path.join(workdir, "web_evidence.sqlite3"),
        os.path.join(workdir, "web_fetch_log.sqlite3"),
        os.path.join(workdir, "web_promoted_memory.sqlite3"),
    )


def _id(prefix: str, text: str) -> str:
    return f"{prefix}:{hashlib.sha1((text or '').encode('utf-8')).hexdigest()[:16]}"


def retrieve(
    *,
    workdir: str,
    query: str,
    intent_family: IntentFamily,
    top_k: int = 6,
) -> List[LaneHit]:
    evidence_db, fetch_log_db, promoted_db = _db_paths(workdir)
    evidence_store = WebEvidenceStore(evidence_db)
    fetch_log = WebFetchLogStore(fetch_log_db)
    promoted_store = WebPromotedMemoryStore(promoted_db)

    local_cached = evidence_store.search(query, top_k=top_k)
    need = detect_web_need(
        query=query,
        intent_family=intent_family,
        local_evidence_count=len(local_cached),
    )
    plan = plan_web_queries(query=query, decision=need, max_variants=3)

    fetcher = WebFetcher(evidence_store=evidence_store, fetch_log=fetch_log)
    web_evidence = fetcher.fetch_plan(plan=plan, need=need, min_cache_hits=2)
    promoted_hits = promoted_store.search(query, top_k=max(1, top_k // 2))
    graph = GraphRetrievalEnricher(workdir)

    # Opportunistic lightweight promotion for stable non-freshness-sensitive evidence.
    if web_evidence and (not need.freshness_sensitive):
        first = web_evidence[0]
        promoted_store.stage_fact(
            promoted_id=_id("wp", (first.get("title") or first.get("url") or query)),
            canonical_fact=(first.get("snippet") or first.get("extracted_text") or "")[:280],
            evidence_ids=[str(first.get("evidence_id") or "")],
            confidence=0.65,
            freshness_state="warm",
            metadata={
                "source_url": first.get("url"),
                "created_from_query": query,
            },
            now_iso=_now_iso(),
        )
        # Deterministic backend approval rule may promote staged -> trusted.
        promoted_store.promote_if_eligible(_id("wp", (first.get("title") or first.get("url") or query)), now_iso=_now_iso())
        promoted_hits = promoted_store.search(query, top_k=max(1, top_k // 2))

    # Retrieval must not mutate/read-shift lifecycle ordering against synthetic
    # fixture time. Use stored lifecycle state as source-of-truth for ranking.

    hits: List[LaneHit] = []
    for item in promoted_hits:
        text = item.get("canonical_fact") or ""
        if not text:
            continue
        promoted_id = str(item.get("promoted_id") or "")
        contradiction_summary = promoted_store.get_contradiction_summary(promoted_id) if promoted_id else {
            "has_contradiction": False,
            "open_contradiction_count": 0,
            "contradiction_ids": [],
            "conflicting_claim_ids": [],
            "conflicting_source_classes": [],
            "contradictions": [],
        }
        graph_origins = graph.get_web_promoted_origins(promoted_id=str(item.get("promoted_id") or ""))
        lifecycle_state = str(item.get("freshness_lifecycle_state") or "")
        if not lifecycle_state:
            lifecycle_state = (
                WebPromotedMemoryStore.FRESHNESS_FRESH
                if str(item.get("freshness_state") or "").strip().lower() in {"hot", "warm"}
                else WebPromotedMemoryStore.FRESHNESS_STALE
            )
        provenance = {
            "promoted_id": item.get("promoted_id"),
            "evidence_ids": item.get("evidence_ids") or [],
            "freshness_state": item.get("freshness_state"),
            "promotion_state": item.get("promotion_state") or "staged",
            "promotion_tier": item.get("promotion_tier") or "local",
            "origin_tier": item.get("origin_tier") or (item.get("promotion_tier") or "local"),
            "promoted_from_ids": item.get("promoted_from_ids") or [],
            "promotion_basis": item.get("promotion_basis") or {},
            "promotion_history": item.get("promotion_history") or [],
            "freshness_lifecycle_state": lifecycle_state,
            "freshness_reason": item.get("freshness_reason") or "",
            "freshness_policy": item.get("freshness_policy") or {},
            "last_validated_at": item.get("last_validated_at"),
            "revalidation_requested_at": item.get("revalidation_requested_at"),
            "approval_reason": item.get("approval_reason") or "",
            "approval_basis": item.get("approval_basis") or {},
            "approved_at": item.get("approved_at"),
            "contradiction": contradiction_summary,
            "metadata": item.get("metadata") or {},
        }
        if graph_origins:
            provenance["graph_source_origins"] = graph_origins
            provenance["graph_source_origin_count"] = len(graph_origins)
            first_source = next((origin.get("source_url") for origin in graph_origins if origin.get("source_url")), None)
            if first_source:
                provenance["source_url"] = first_source
        provenance["trust_status"] = "trusted" if str(item.get("promotion_state") or "staged") == "trusted" else "unreviewed"
        provenance["approval_status"] = "approved" if str(item.get("promotion_state") or "staged") == "trusted" else "unreviewed"
        provenance["import_state"] = "local"
        provenance["authoritative"] = True
        provenance["lifecycle"] = normalized_lifecycle_view(source_class="web_promoted", provenance=provenance)
        state = str(item.get("promotion_state") or "staged")
        lifecycle = lifecycle_state
        tier = str(item.get("promotion_tier") or "local")
        has_contradiction = bool(contradiction_summary.get("has_contradiction"))
        if state == "trusted" and lifecycle == WebPromotedMemoryStore.FRESHNESS_FRESH:
            trust_adjust = 0.08
            semantic_adjust = 0.05
        elif state == "trusted" and lifecycle == WebPromotedMemoryStore.FRESHNESS_REVALIDATION_PENDING:
            trust_adjust = -0.10
            semantic_adjust = -0.03
        elif state == "trusted":
            trust_adjust = -0.08
            semantic_adjust = -0.02
        else:
            trust_adjust = -0.05
            semantic_adjust = 0.0
        # Tier preference is additive and bounded.
        if tier == WebPromotedMemoryStore.TIER_GLOBAL:
            trust_adjust += 0.06
            semantic_adjust += 0.03
        elif tier == WebPromotedMemoryStore.TIER_WORKSPACE:
            trust_adjust += 0.03
            semantic_adjust += 0.015
        if has_contradiction:
            # Contradictory promoted knowledge remains retrievable but should carry caution.
            trust_adjust -= 0.12
            semantic_adjust -= 0.04
        hits.append(
            LaneHit(
                id=f"webpromoted:{item.get('promoted_id')}",
                lane=RetrievalLane.WEB,
                memory_type=MemoryType.WEB_PROMOTED_FACT,
                text=text,
                provenance=provenance,
                lexical_score=0.4,
                semantic_score=0.55 + semantic_adjust,
                recency_score=0.4,
                freshness_score=0.7 if item.get("freshness_state") in {"hot", "warm"} else 0.4,
                trust_score=max(0.0, min(1.0, float(item.get("confidence") or 0.5) + trust_adjust)),
                novelty_score=0.35,
                contradiction_risk=(0.65 if has_contradiction else 0.1),
                cluster_id=f"web_promoted:{tier}:{state}:{lifecycle}",
                compressible=True,
            ).with_token_estimate()
        )

    for item in web_evidence[:top_k]:
        text = item.get("extracted_text") or item.get("snippet") or ""
        if not text:
            continue
        freshness = item.get("freshness_class") or "warm"
        hits.append(
            LaneHit(
                id=f"webevidence:{item.get('evidence_id') or _id('we', text)}",
                lane=RetrievalLane.WEB,
                memory_type=MemoryType.WEB_EVIDENCE,
                text=text,
                provenance={
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "source_id": item.get("source_id"),
                    "fetched_at": item.get("fetched_at"),
                    "published_at": item.get("published_at"),
                    "updated_at": item.get("updated_at"),
                    "freshness_class": freshness,
                    "evidence_id": item.get("evidence_id"),
                    "promotion_state": "staged",
                    "approval_status": "unreviewed",
                    "trust_status": "unreviewed",
                    "freshness_lifecycle_state": "unreviewed",
                },
                lexical_score=0.35,
                semantic_score=0.5,
                recency_score=0.55 if freshness == "hot" else 0.4,
                freshness_score=0.85 if freshness == "hot" else 0.6,
                trust_score=float(item.get("trust_score") or 0.5),
                novelty_score=0.45,
                contradiction_risk=0.15,
                cluster_id=str(item.get("source_id") or "web"),
                compressible=True,
            ).with_token_estimate()
        )
        hits[-1].provenance["lifecycle"] = normalized_lifecycle_view(source_class="web_evidence", provenance=hits[-1].provenance)

    def _sort_key(hit: LaneHit):
        if hit.memory_type == MemoryType.WEB_PROMOTED_FACT:
            state = str((hit.provenance or {}).get("promotion_state") or "staged")
            lifecycle = str((hit.provenance or {}).get("freshness_lifecycle_state") or "stale")
            tier = str((hit.provenance or {}).get("promotion_tier") or "local")
            contradiction_count = int(((hit.provenance or {}).get("contradiction") or {}).get("open_contradiction_count") or 0)
            state_rank = 0 if state == "trusted" else 1
            lifecycle_rank = 0 if lifecycle == "fresh" else (1 if lifecycle == "revalidation_pending" else 2)
            tier_rank = 0 if tier == "global" else (1 if tier == "workspace" else 2)
            contradiction_rank = 0 if contradiction_count == 0 else 1
            return (0, tier_rank, state_rank, lifecycle_rank, contradiction_rank, -float(hit.trust_score or 0.0), hit.id)
        return (1, 0, -float(hit.trust_score or 0.0), hit.id)

    hits.sort(key=_sort_key)
    return hits[: top_k]
