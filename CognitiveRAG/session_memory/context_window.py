"""
Lossless compaction / first-class context selector foundation for session memory.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane
from CognitiveRAG.crag.context_selection.candidate_builder import build_candidates
from CognitiveRAG.crag.context_selection.lane_pruning import prune_lane_local
from CognitiveRAG.crag.context_selection.policies import get_policy
from CognitiveRAG.crag.context_selection.selector import select_context
from CognitiveRAG.crag.contracts.types import estimate_tokens


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

    older = [m for m in raw_messages if int(m.get("index", 0)) < int(older_than_index)]
    if not older:
        return []

    chunk_size = 20
    chunks = [older[i : i + chunk_size] for i in range(0, len(older), chunk_size)]

    existing = _load_fallback_summaries(session_id)
    created: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        artifact = summarizer(chunk)
        node = {
            "session_id": session_id,
            "chunk_index": i,
            "summary": artifact.get("summary"),
            "source_count": artifact.get("source_count", len(chunk)),
            "created_by": "context_window.v2.selector_foundation",
        }
        existing.append(node)
        created.append(node)

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

    fresh_tail = raw[-int(fresh_tail_count) :] if fresh_tail_count > 0 else []
    older_raw = raw[: max(0, len(raw) - len(fresh_tail))]

    detected_intent = IntentFamily(intent_family) if intent_family else _detect_intent_family(query)
    policy = get_policy(detected_intent)

    # Phase A: hard reservations
    reserved_fresh_tail_tokens = sum(estimate_tokens(m.get("text") or "") for m in fresh_tail)
    reserved_tokens = min(
        int(budget),
        int(policy.hard_reservation_tokens) + reserved_fresh_tail_tokens,
    )

    # Phase B + C + D + E + F + G + H
    candidates = build_candidates(
        session_id=session_id,
        query=query or "",
        fresh_tail=fresh_tail,
        older_raw=older_raw,
        summaries=summaries,
        workdir=WORKDIR,
        intent_family=detected_intent,
    )
    pruned = prune_lane_local(candidates)
    selected_pairs, dropped, explanation = select_context(
        candidates=pruned,
        policy=policy,
        total_budget=int(budget),
        reserved_tokens=reserved_tokens,
        intent_family=detected_intent,
    )

    selected_candidates = [candidate for candidate, _ in selected_pairs]

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
    }
