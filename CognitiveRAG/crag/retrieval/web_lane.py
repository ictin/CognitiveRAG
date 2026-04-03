from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import List

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.graph_memory.enrichment import GraphRetrievalEnricher
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

    hits: List[LaneHit] = []
    for item in promoted_hits:
        text = item.get("canonical_fact") or ""
        if not text:
            continue
        graph_origins = graph.get_web_promoted_origins(promoted_id=str(item.get("promoted_id") or ""))
        provenance = {
            "promoted_id": item.get("promoted_id"),
            "evidence_ids": item.get("evidence_ids") or [],
            "freshness_state": item.get("freshness_state"),
            "promotion_state": item.get("promotion_state") or "staged",
            "approval_reason": item.get("approval_reason") or "",
            "approval_basis": item.get("approval_basis") or {},
            "approved_at": item.get("approved_at"),
            "metadata": item.get("metadata") or {},
        }
        if graph_origins:
            provenance["graph_source_origins"] = graph_origins
            provenance["graph_source_origin_count"] = len(graph_origins)
            first_source = next((origin.get("source_url") for origin in graph_origins if origin.get("source_url")), None)
            if first_source:
                provenance["source_url"] = first_source
        state = str(item.get("promotion_state") or "staged")
        trust_adjust = 0.08 if state == "trusted" else -0.05
        semantic_adjust = 0.05 if state == "trusted" else 0.0
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
                contradiction_risk=0.1,
                cluster_id=f"web_promoted:{state}",
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

    def _sort_key(hit: LaneHit):
        if hit.memory_type == MemoryType.WEB_PROMOTED_FACT:
            state = str((hit.provenance or {}).get("promotion_state") or "staged")
            state_rank = 0 if state == "trusted" else 1
            return (0, state_rank, -float(hit.trust_score or 0.0), hit.id)
        return (1, 0, -float(hit.trust_score or 0.0), hit.id)

    hits.sort(key=_sort_key)
    return hits[: top_k]
