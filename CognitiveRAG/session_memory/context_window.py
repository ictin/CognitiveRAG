"""
Lossless compaction / first-class context selector foundation for session memory.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane
from CognitiveRAG.crag.cognition.controller import CognitiveController
from CognitiveRAG.crag.cognition.discovery import DiscoveryExecutor, discovery_items_to_candidates
from CognitiveRAG.crag.context_selection.candidate_builder import build_candidates_with_route
from CognitiveRAG.crag.context_selection.compatibility import load_runtime_compatibility_engine_from_env
from CognitiveRAG.crag.context_selection.lane_pruning import prune_lane_local
from CognitiveRAG.crag.context_selection.policies import get_policy
from CognitiveRAG.crag.context_selection.selector import select_context
from CognitiveRAG.crag.contracts.types import estimate_tokens
from CognitiveRAG.session_memory.compaction import (
    SessionCompactionStore,
    build_lineage,
    build_raw_snapshot,
    compute_eligible_messages,
    summarize_compaction_state,
)


WORKDIR = os.path.join(os.getcwd(), "data", "session_memory")
os.makedirs(WORKDIR, exist_ok=True)


def _load_fallback_summaries(session_id: str) -> List[Dict[str, Any]]:
    path = os.path.join(WORKDIR, f"summaries_{session_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_fallback_summaries(session_id: str, summaries: List[Dict[str, Any]]) -> None:
    path = os.path.join(WORKDIR, f"summaries_{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


def _default_summarizer(messages: List[Dict[str, Any]], max_chars: int = 200) -> Dict[str, Any]:
    text = "\n".join((m.get("text") or "") for m in messages)
    summary = text[:max_chars]
    return {"summary": summary, "source_count": len(messages), "token_est": estimate_tokens(summary)}


def _detect_intent_family(query: str | None) -> IntentFamily:
    q = (query or "").strip().lower()
    if not q:
        return IntentFamily.MEMORY_SUMMARY
    if "what did we say" in q or "earlier" in q or "quote" in q:
        return IntentFamily.EXACT_RECALL
    if "investigate" in q or "investigation" in q:
        return IntentFamily.INVESTIGATIVE
    if "memory organized" in q or "architecture" in q:
        return IntentFamily.ARCHITECTURE_EXPLANATION
    if "what do you remember" in q or "what do you know about me" in q:
        return IntentFamily.MEMORY_SUMMARY
    if "what can you tell me about" in q or "synopsis" in q or "book" in q or "corpus" in q:
        return IntentFamily.CORPUS_OVERVIEW
    if "plan" in q or "next step" in q:
        return IntentFamily.PLANNING
    return IntentFamily.INVESTIGATIVE


def _load_raw_messages(session_id: str) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    try:
        from CognitiveRAG.session_memory.conversation_store import ConversationStore

        store = ConversationStore()
        raw = list(store.get_messages(session_id))
    except Exception:
        raw = []

    if not raw:
        raw_path = os.path.join(WORKDIR, f"raw_{session_id}.json")
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

    for i, m in enumerate(raw):
        # Ensure deterministic sortable index even when store lacks explicit index.
        m.setdefault("index", i)
    return raw


def _load_summaries(session_id: str) -> List[Dict[str, Any]]:
    # Keep compatibility with current fallback summary file contract.
    return sorted(_load_fallback_summaries(session_id), key=lambda s: int(s.get("chunk_index", 0)))


def compact_session(
    session_id: str,
    older_than_index: int,
    summarizer: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Create additive derived summaries for older messages.

    This preserves raw history and only adds summary artifacts.
    """
    summarizer = summarizer or _default_summarizer
    raw_messages = _load_raw_messages(session_id)

    compaction_store = SessionCompactionStore(os.path.join(WORKDIR, "compaction.sqlite3"))
    existing_segments = compaction_store.list_segments(session_id)
    already_compacted = {
        str(row.get("message_key") or "")
        for seg in existing_segments
        for row in list(seg.get("lineage") or [])
        if str(row.get("message_key") or "")
    }

    compactable, quarantined = compute_eligible_messages(
        raw_messages=raw_messages,
        older_than_index=int(older_than_index),
        already_compacted_keys=already_compacted,
    )
    if not compactable and not quarantined:
        return []

    chunk_size = 20
    chunks = [compactable[i : i + chunk_size] for i in range(0, len(compactable), chunk_size)]

    existing = _load_fallback_summaries(session_id)
    created: List[Dict[str, Any]] = []
    base_chunk_index = max([int(s.get("chunk_index", -1)) for s in existing], default=-1) + 1

    for i, chunk in enumerate(chunks):
        artifact = summarizer(chunk)
        lineage = build_lineage(chunk)
        snapshot = build_raw_snapshot(chunk)
        start_index = min(int(m.get("index", 0)) for m in chunk)
        end_index = max(int(m.get("index", 0)) for m in chunk)
        import hashlib

        lineage_key = "||".join(f"{row.get('message_key')}|{row.get('index')}" for row in lineage)
        segment_id = f"compact:{hashlib.sha1(f'{session_id}|{lineage_key}'.encode('utf-8')).hexdigest()}"
        compaction_store.upsert_segment(
            session_id=session_id,
            segment_id=segment_id,
            chunk_index=base_chunk_index + i,
            start_index=start_index,
            end_index=end_index,
            summary=str(artifact.get("summary") or ""),
            source_count=int(artifact.get("source_count", len(chunk))),
            policy_reason="age_based_chunk_compaction",
            status="compacted",
            lineage=lineage,
            raw_snapshot=snapshot,
            metadata={
                "older_than_index": int(older_than_index),
                "created_by": "context_window.v3.compaction_policy",
            },
        )
        node = {
            "session_id": session_id,
            "chunk_index": base_chunk_index + i,
            "summary": artifact.get("summary"),
            "source_count": artifact.get("source_count", len(chunk)),
            "created_by": "context_window.v3.compaction_policy",
            "compaction": {
                "segment_id": segment_id,
                "status": "compacted",
                "policy_reason": "age_based_chunk_compaction",
                "source_index_start": start_index,
                "source_index_end": end_index,
                "lineage_count": len(lineage),
                "recoverability": "raw_or_snapshot",
            },
        }
        existing.append(node)
        created.append(node)

    for idx, qmsg in enumerate(quarantined):
        q_index = int(qmsg.get("index", idx))
        q_mid = str(qmsg.get("message_id") or "")
        message_key = q_mid and f"message_id:{q_mid}" or f"index:{q_index}"
        compaction_store.upsert_quarantined(
            session_id=session_id,
            message_key=message_key,
            msg_index=q_index,
            reason="low_value_quarantine",
            metadata={"preview": str(qmsg.get("text") or "")[:120]},
        )

    _save_fallback_summaries(session_id, existing)
    return created


def assemble_context(
    session_id: str,
    fresh_tail_count: int = 20,
    budget: int = 4096,
    query: str | None = None,
    intent_family: IntentFamily | str | None = None,
) -> Dict[str, Any]:
    """Assemble context using backend-first selector foundation.

    Compatibility:
    - preserves legacy top-level keys: fresh_tail, summaries
    - adds machine-readable selector explanation artifact
    """
    raw = sorted(_load_raw_messages(session_id), key=lambda m: int(m.get("index", 0)))
    summaries = _load_summaries(session_id)
    compaction_store = SessionCompactionStore(os.path.join(WORKDIR, "compaction.sqlite3"))

    fresh_tail = raw[-int(fresh_tail_count) :] if fresh_tail_count > 0 else []
    older_raw = raw[: max(0, len(raw) - len(fresh_tail))]

    controller = CognitiveController()
    discovery_plan = controller.build_plan(
        query=query or "",
        hinted_intent=intent_family,
        local_evidence_count=len(older_raw) + len(summaries),
    )
    detected_intent = discovery_plan.intent_family
    policy = get_policy(detected_intent)

    # Phase A: hard reservations
    reserved_fresh_tail_tokens = sum(estimate_tokens(m.get("text") or "") for m in fresh_tail)
    reserved_tokens = min(
        int(budget),
        int(policy.hard_reservation_tokens) + reserved_fresh_tail_tokens,
    )

    # Phase B + C + D + E + F + G + H
    route_plan, candidates = build_candidates_with_route(
        session_id=session_id,
        query=query or "",
        fresh_tail=fresh_tail,
        older_raw=older_raw,
        summaries=summaries,
        workdir=WORKDIR,
        intent_family=detected_intent,
    )
    pre_prune_count = len(candidates)
    pre_prune_lane_counts: Dict[str, int] = {}
    pre_prune_lane_tokens: Dict[str, int] = {}
    for candidate in candidates:
        lane = candidate.lane.value
        pre_prune_lane_counts[lane] = pre_prune_lane_counts.get(lane, 0) + 1
        pre_prune_lane_tokens[lane] = pre_prune_lane_tokens.get(lane, 0) + int(candidate.tokens)

    pruned = prune_lane_local(candidates)
    post_prune_count = len(pruned)
    pruned_count = max(0, pre_prune_count - post_prune_count)

    discovery_executor = DiscoveryExecutor()
    discovery_result = discovery_executor.run(plan=discovery_plan, candidate_pool=pruned)
    discovery_candidates = discovery_items_to_candidates(discovery_result.injected_discoveries)
    if discovery_candidates:
        pruned = pruned + discovery_candidates

    compatibility_engine, compatibility_state = load_runtime_compatibility_engine_from_env()
    selected_pairs, dropped, explanation = select_context(
        candidates=pruned,
        policy=policy,
        total_budget=int(budget),
        reserved_tokens=reserved_tokens,
        intent_family=detected_intent,
        compatibility_engine=compatibility_engine,
    )

    selected_candidates = [candidate for candidate, _ in selected_pairs]
    selected_lane_counts: Dict[str, int] = {}
    selected_lane_tokens: Dict[str, int] = {}
    for candidate in selected_candidates:
        lane = candidate.lane.value
        selected_lane_counts[lane] = selected_lane_counts.get(lane, 0) + 1
        selected_lane_tokens[lane] = selected_lane_tokens.get(lane, 0) + int(candidate.tokens)

    dropped_lane_counts: Dict[str, int] = {}
    dropped_lane_tokens: Dict[str, int] = {}
    drop_reason_counts: Dict[str, int] = {}
    for candidate, reason in dropped:
        lane = candidate.lane.value
        dropped_lane_counts[lane] = dropped_lane_counts.get(lane, 0) + 1
        dropped_lane_tokens[lane] = dropped_lane_tokens.get(lane, 0) + int(candidate.tokens)
        drop_reason_counts[reason] = drop_reason_counts.get(reason, 0) + 1

    selected_tokens_total = sum(int(c.tokens) for c in selected_candidates)
    discovery_count = len(discovery_result.injected_discoveries)
    discovery_tokens = sum(int(item.tokens) for item in discovery_result.injected_discoveries)
    available_budget = max(0, int(budget) - int(reserved_tokens))

    # Compatibility mapping for existing consumers.
    selected_fresh = []
    selected_summaries = []
    for candidate in selected_candidates:
        if candidate.lane == RetrievalLane.FRESH_TAIL:
            selected_fresh.append(candidate.provenance.get("message") or {"text": candidate.text})
        elif candidate.lane == RetrievalLane.SESSION_SUMMARY:
            selected_summaries.append(candidate.provenance.get("summary") or {"summary": candidate.text})

    if not selected_fresh:
        # Always preserve minimal fresh tail visibility contract.
        selected_fresh = fresh_tail[-max(1, policy.minimal_fresh_tail) :]
    if not selected_summaries and summaries:
        # Preserve pre-existing fallback behavior: surfaced summaries remain visible
        # even when selector excludes summary-lane blocks from final selection.
        selected_summaries = summaries[:1]

    return {
        "fresh_tail": selected_fresh,
        "summaries": selected_summaries,
        "selected_blocks": [
            {
                "id": c.id,
                "lane": c.lane.value,
                "memory_type": c.memory_type.value,
                "text": c.text,
                "tokens": c.tokens,
                "provenance": c.provenance,
            }
            for c in selected_candidates
        ],
        "dropped_blocks": [{"id": c.id, "reason": reason, "lane": c.lane.value} for c, reason in dropped],
        "explanation": explanation.model_dump(),
        "retrieval_route": {
            "intent_family": route_plan.intent_family.value,
            "lanes": [lane.value for lane in route_plan.lanes],
            "reason": route_plan.reason,
            "metadata": dict(route_plan.metadata or {}),
        },
        "selector_metrics": {
            "candidate_counts": {
                "pre_prune": pre_prune_count,
                "post_prune": post_prune_count,
                "pruned": pruned_count,
                "selected": len(selected_candidates),
                "dropped": len(dropped),
            },
            "lane_counts": {
                "pre_prune": pre_prune_lane_counts,
                "selected": selected_lane_counts,
                "dropped": dropped_lane_counts,
            },
            "lane_tokens": {
                "pre_prune": pre_prune_lane_tokens,
                "selected": selected_lane_tokens,
                "dropped": dropped_lane_tokens,
            },
            "budget": {
                "total_budget": int(budget),
                "reserved_tokens": int(reserved_tokens),
                "available_budget": int(available_budget),
                "selected_tokens": int(selected_tokens_total),
                "used_total_tokens": int(reserved_tokens + selected_tokens_total),
                "budget_utilization_ratio": float(selected_tokens_total / max(1, available_budget)),
            },
            "decision_stats": {
                "drop_reasons": drop_reason_counts,
                "route_intent_family": route_plan.intent_family.value,
                "route_lane_count": len(route_plan.lanes),
                "compatibility_engine": compatibility_state.__dict__,
            },
            "discovery": {
                "injected_count": discovery_count,
                "injected_tokens": discovery_tokens,
            },
        },
        "discovery_plan": discovery_plan.model_dump(mode="json"),
        "discovery": discovery_result.model_dump(mode="json"),
        "compaction": summarize_compaction_state(session_id=session_id, store=compaction_store),
        "recoverability": {
            "raw_message_count": len(raw),
            "summary_count": len(summaries),
            "lineage_recovery_mode": "raw_or_snapshot",
        },
    }
