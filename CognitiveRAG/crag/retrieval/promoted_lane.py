from __future__ import annotations

import json
import os
import sqlite3
from typing import List

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.graph_memory.enrichment import GraphRetrievalEnricher
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.memory.reasoning_success import refresh_reasoning_success_signals


def _load_reasoning(workdir: str, limit: int = 10, db_path: str | None = None):
    db_path = str(db_path or os.path.join(workdir, "reasoning.sqlite3"))
    if not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as conn:
        try:
            return conn.execute(
                "SELECT pattern_id, solution_summary, confidence, provenance_json, memory_subtype, normalized_text, freshness_state, "
                "reuse_count, canonical_pattern_id, near_duplicate_of, success_signal_count, failure_signal_count, success_confidence, success_basis_json "
                "FROM reasoning_patterns ORDER BY rowid DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        except sqlite3.OperationalError:
            try:
                rows = conn.execute(
                    "SELECT pattern_id, solution_summary, confidence, provenance_json, memory_subtype, normalized_text, freshness_state, "
                    "reuse_count, canonical_pattern_id, near_duplicate_of "
                    "FROM reasoning_patterns ORDER BY rowid DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
                return [(*row, 0, 0, 0.0, None) for row in rows]
            except sqlite3.OperationalError:
                try:
                    rows = conn.execute(
                        "SELECT pattern_id, solution_summary, confidence, provenance_json, memory_subtype, normalized_text, freshness_state "
                        "FROM reasoning_patterns ORDER BY rowid DESC LIMIT ?",
                        (int(limit),),
                    ).fetchall()
                    return [(*row, 1, None, None, 0, 0, 0.0, None) for row in rows]
                except sqlite3.OperationalError:
                    rows = conn.execute(
                        "SELECT pattern_id, solution_summary, confidence, provenance_json FROM reasoning_patterns ORDER BY rowid DESC LIMIT ?",
                        (int(limit),),
                    ).fetchall()
                    return [(*row, None, None, None, 1, None, None, 0, 0, 0.0, None) for row in rows]


def retrieve(*, workdir: str, intent_family: IntentFamily, query: str, top_k: int = 6) -> List[LaneHit]:
    hits: List[LaneHit] = []
    # Keep success-signal accumulation backend-owned and deterministic.
    try:
        refresh_reasoning_success_signals(workdir=workdir)
    except Exception:
        pass
    db_path = None
    try:
        from CognitiveRAG.core.settings import settings as _settings

        db_path = str(getattr(_settings.store, "reasoning_db_path", "") or "")
    except Exception:
        db_path = None
    rows = _load_reasoning(workdir, limit=max(8, top_k), db_path=db_path)
    graph = GraphRetrievalEnricher(workdir)
    signature_matches = graph.find_problem_signature_matches(query=query, max_matches=5)
    graph_bonus_by_pattern: dict[str, float] = {}
    for match in signature_matches:
        prev = graph_bonus_by_pattern.get(match.pattern_id, 0.0)
        # Bounded additive helper signal, never a graph-only takeover.
        graph_bonus_by_pattern[match.pattern_id] = max(prev, min(0.2, 0.05 + 0.2 * float(match.score)))

    qtokens = set((query or "").lower().split())
    for (
        pattern_id,
        summary,
        confidence,
        provenance_json,
        memory_subtype,
        normalized_text,
        freshness_state,
        reuse_count,
        canonical_pattern_id,
        near_duplicate_of,
        success_signal_count,
        failure_signal_count,
        success_confidence,
        success_basis_json,
    ) in rows:
        try:
            provenance = {"reasoning_provenance": json.loads(provenance_json) if provenance_json else []}
        except Exception:
            provenance = {"reasoning_provenance": []}
        provenance["source_class"] = "promoted_memory"
        provenance["source_store"] = "reasoning_store"
        if memory_subtype:
            provenance["memory_subtype"] = memory_subtype
        if freshness_state:
            provenance["freshness_state"] = freshness_state
        if normalized_text:
            provenance["normalized_text"] = normalized_text
        provenance["reuse_count"] = int(reuse_count or 1)
        if canonical_pattern_id:
            provenance["canonical_pattern_id"] = canonical_pattern_id
        if near_duplicate_of:
            provenance["near_duplicate_of"] = near_duplicate_of
        provenance["success_signal_count"] = int(success_signal_count or 0)
        provenance["failure_signal_count"] = int(failure_signal_count or 0)
        provenance["success_confidence"] = float(success_confidence or 0.0)
        try:
            provenance["success_basis"] = json.loads(success_basis_json) if success_basis_json else {}
        except Exception:
            provenance["success_basis"] = {}
        support_links = graph.get_reasoning_support_links(pattern_id=pattern_id)
        if support_links:
            provenance["graph_support_links"] = support_links
            provenance["graph_support_count"] = len(support_links)

        linked_matches = [m for m in signature_matches if m.pattern_id == pattern_id]
        if linked_matches:
            provenance["graph_problem_signature_matches"] = [
                {
                    "problem_signature": m.problem_signature,
                    "overlap": m.overlap,
                    "query_token_count": m.query_token_count,
                    "match_score": m.score,
                }
                for m in linked_matches
            ]
        text = (summary or "").strip()
        if memory_subtype:
            text = f"[{memory_subtype}] {text}"

        lexical = 0.2 if query else 0.0
        if qtokens:
            hay = f"{text} {normalized_text or ''} {memory_subtype or ''}".lower()
            overlap = sum(1 for tok in qtokens if tok in hay)
            lexical += min(0.6, overlap * 0.08)
        semantic = 0.25 if query else 0.0
        if intent_family in {IntentFamily.MEMORY_SUMMARY, IntentFamily.PLANNING}:
            semantic += 0.2
        if memory_subtype and intent_family == IntentFamily.MEMORY_SUMMARY and "profile" in memory_subtype:
            semantic += 0.12
        if memory_subtype and intent_family == IntentFamily.EXACT_RECALL and "workflow" in memory_subtype:
            semantic += 0.1
        # Reuse is a bounded helper signal only.
        semantic += min(0.18, max(0, int(reuse_count or 1) - 1) * 0.03)
        # Repeated-success promotion helper is additive and capped.
        semantic += min(0.12, max(0.0, float(success_confidence or 0.0)) * 0.12)
        semantic += graph_bonus_by_pattern.get(pattern_id, 0.0)
        effective_trust = max(0.0, min(1.0, float(confidence or 0.5) + min(0.10, float(success_confidence or 0.0) * 0.10)))
        hits.append(
            LaneHit(
                id=f"promoted:{pattern_id}",
                lane=RetrievalLane.PROMOTED,
                memory_type=MemoryType.PROMOTED_FACT,
                text=text,
                provenance=provenance,
                lexical_score=lexical,
                semantic_score=semantic,
                recency_score=0.55,
                freshness_score=0.7,
                trust_score=effective_trust,
                novelty_score=0.45,
                contradiction_risk=0.0,
                cluster_id=memory_subtype or "promoted",
                must_include=False,
                compressible=True,
            ).with_token_estimate()
        )

    hits.sort(key=lambda h: (-(h.semantic_score + h.lexical_score + h.trust_score), h.id))
    return hits[:top_k]
