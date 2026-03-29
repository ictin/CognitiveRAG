from __future__ import annotations

from typing import Any, Dict, Iterable, List

from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit


def retrieve(
    *,
    session_id: str,
    fresh_tail: Iterable[Dict[str, Any]],
    older_raw: Iterable[Dict[str, Any]],
    summaries: Iterable[Dict[str, Any]] | None = None,
    top_k: int = 20,
) -> List[LaneHit]:
    hits: List[LaneHit] = []

    tail = list(fresh_tail)
    total_tail = max(1, len(tail))
    for i, msg in enumerate(tail):
        text = msg.get("text") or ""
        if not text:
            continue
        rec = 1.0 - float(total_tail - i - 1) / float(total_tail)
        msg_id = str(msg.get("message_id") or msg.get("index") or i)
        hits.append(
            LaneHit(
                id=f"fresh:{session_id}:{msg_id}",
                lane=RetrievalLane.FRESH_TAIL,
                memory_type=MemoryType.EPISODIC_RAW,
                text=text,
                provenance={"session_id": session_id, "message": msg},
                lexical_score=0.45,
                semantic_score=0.5,
                recency_score=rec,
                freshness_score=0.95,
                trust_score=0.8,
                novelty_score=0.35,
                contradiction_risk=0.0,
                cluster_id="fresh_tail",
                must_include=True,
                compressible=False,
            ).with_token_estimate()
        )

    older = list(older_raw)
    total_old = max(1, len(older))
    for i, msg in enumerate(older):
        text = msg.get("text") or ""
        if not text:
            continue
        rec = 1.0 - float(total_old - i) / float(total_old + 1)
        msg_id = str(msg.get("message_id") or msg.get("index") or i)
        hits.append(
            LaneHit(
                id=f"episodic:{session_id}:{msg_id}",
                lane=RetrievalLane.EPISODIC,
                memory_type=MemoryType.EPISODIC_RAW,
                text=text,
                provenance={"session_id": session_id, "message": msg},
                lexical_score=0.35,
                semantic_score=0.45,
                recency_score=max(0.05, rec),
                freshness_score=0.7,
                trust_score=0.8,
                novelty_score=0.4,
                contradiction_risk=0.0,
                cluster_id="session_history",
                must_include=False,
                compressible=True,
            ).with_token_estimate()
        )

    for i, summary in enumerate(list(summaries or [])):
        text = summary.get("summary") or summary.get("text") or ""
        if not text:
            continue
        chunk = summary.get("chunk_index", i)
        hits.append(
            LaneHit(
                id=f"summary:{session_id}:{chunk}",
                lane=RetrievalLane.SESSION_SUMMARY,
                memory_type=MemoryType.SUMMARY,
                text=text,
                provenance={"session_id": session_id, "summary": summary},
                lexical_score=0.2,
                semantic_score=0.35,
                recency_score=0.3,
                freshness_score=0.6,
                trust_score=0.75,
                novelty_score=0.5,
                contradiction_risk=0.0,
                cluster_id="session_summaries",
                must_include=False,
                compressible=True,
            ).with_token_estimate()
        )

    must = [h for h in hits if h.must_include]
    summary_hits = [h for h in hits if (not h.must_include and h.lane == RetrievalLane.SESSION_SUMMARY)]
    other = [h for h in hits if (not h.must_include and h.lane != RetrievalLane.SESSION_SUMMARY)]
    other.sort(key=lambda h: (-h.recency_score, h.id))
    if len(must) >= top_k:
        return must[:top_k]
    out = list(must)
    if summary_hits and len(out) < top_k:
        summary_hits.sort(key=lambda h: h.id)
        out.append(summary_hits[0])
    remaining = max(0, top_k - len(out))
    out.extend(other[:remaining])
    return out
