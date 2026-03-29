from __future__ import annotations

from typing import Any, Dict, Iterable, List

from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit


def _norm_words(text: str) -> set[str]:
    return {w for w in " ".join((text or "").lower().split()).split() if w}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(max(1, len(a | b)))


def _mk(hit_id: str, text: str, provenance: Dict[str, Any], sem: float, *, memory_type: MemoryType, cluster: str | None = None) -> LaneHit:
    return LaneHit(
        id=hit_id,
        lane=RetrievalLane.SEMANTIC,
        memory_type=memory_type,
        text=text,
        provenance=provenance,
        lexical_score=max(0.0, sem - 0.05),
        semantic_score=sem,
        recency_score=provenance.get("recency", 0.3),
        freshness_score=0.6,
        trust_score=0.65,
        novelty_score=0.45,
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
    qwords = _norm_words(query)

    for i, msg in enumerate(list(fresh_tail)):
        text = msg.get("text") or ""
        sem = _jaccard(qwords, _norm_words(text))
        if sem <= 0:
            continue
        hits.append(_mk(f"sem:fresh:{session_id}:{i}", text, {"message": msg, "recency": 0.9}, sem, memory_type=MemoryType.EPISODIC_RAW, cluster="fresh_tail"))

    for i, msg in enumerate(list(older_raw)):
        text = msg.get("text") or ""
        sem = _jaccard(qwords, _norm_words(text))
        if sem <= 0:
            continue
        hits.append(_mk(f"sem:episodic:{session_id}:{i}", text, {"message": msg, "recency": 0.4}, sem, memory_type=MemoryType.EPISODIC_RAW, cluster="episodic"))

    for i, summary in enumerate(list(summaries)):
        text = summary.get("summary") or summary.get("text") or ""
        sem = _jaccard(qwords, _norm_words(text))
        if sem <= 0:
            continue
        hits.append(_mk(f"sem:summary:{session_id}:{i}", text, {"summary": summary, "recency": 0.35}, sem, memory_type=MemoryType.SUMMARY, cluster="summaries"))

    hits.sort(key=lambda h: (-h.semantic_score, h.id))
    return hits[:top_k]
