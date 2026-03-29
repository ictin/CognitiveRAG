from __future__ import annotations

from typing import Any, Dict, Iterable, List

from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _overlap(query: str, text: str) -> float:
    q = set(_norm(query).split())
    t = set(_norm(text).split())
    if not q or not t:
        return 0.0
    return float(len(q & t)) / float(max(1, len(q)))


def _mk(hit_id: str, text: str, provenance: Dict[str, Any], score: float, *, memory_type: MemoryType, cluster: str | None = None) -> LaneHit:
    return LaneHit(
        id=hit_id,
        lane=RetrievalLane.LEXICAL,
        memory_type=memory_type,
        text=text,
        provenance=provenance,
        lexical_score=score,
        semantic_score=max(0.0, score - 0.1),
        recency_score=provenance.get("recency", 0.3),
        freshness_score=0.6,
        trust_score=0.65,
        novelty_score=0.4,
        contradiction_risk=0.0,
        cluster_id=cluster,
    ).with_token_estimate()


def retrieve(
    *,
    query: str,
    session_id: str,
    fresh_tail: Iterable[Dict[str, Any]],
    older_raw: Iterable[Dict[str, Any]],
    summaries: Iterable[Dict[str, Any]],
    top_k: int = 8,
) -> List[LaneHit]:
    hits: List[LaneHit] = []

    for i, msg in enumerate(list(fresh_tail)):
        text = msg.get("text") or ""
        score = _overlap(query, text)
        if score <= 0:
            continue
        hits.append(_mk(f"lex:fresh:{session_id}:{i}", text, {"message": msg, "recency": 0.9}, score, memory_type=MemoryType.EPISODIC_RAW, cluster="fresh_tail"))

    for i, msg in enumerate(list(older_raw)):
        text = msg.get("text") or ""
        score = _overlap(query, text)
        if score <= 0:
            continue
        hits.append(_mk(f"lex:episodic:{session_id}:{i}", text, {"message": msg, "recency": 0.4}, score, memory_type=MemoryType.EPISODIC_RAW, cluster="episodic"))

    for i, summary in enumerate(list(summaries)):
        text = summary.get("summary") or summary.get("text") or ""
        score = _overlap(query, text)
        if score <= 0:
            continue
        hits.append(_mk(f"lex:summary:{session_id}:{i}", text, {"summary": summary, "recency": 0.35}, score, memory_type=MemoryType.SUMMARY, cluster="summaries"))

    hits.sort(key=lambda h: (-h.lexical_score, h.id))
    return hits[:top_k]
