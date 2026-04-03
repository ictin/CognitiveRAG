from __future__ import annotations

from typing import Any, Dict, Iterable, List

from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.retrieval.vector_backend import VectorRecord, resolve_vector_backend


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
    vector_backend_name: str | None = None,
) -> List[LaneHit]:
    hits: List[LaneHit] = []
    backend, requested_backend, used_fallback = resolve_vector_backend(vector_backend_name)
    records: list[VectorRecord] = []

    for i, msg in enumerate(list(fresh_tail)):
        text = msg.get("text") or ""
        records.append(
            VectorRecord(
                record_id=f"sem:fresh:{session_id}:{i}",
                text=text,
                memory_type=MemoryType.EPISODIC_RAW,
                cluster_id="fresh_tail",
                source_type="fresh_tail",
                provenance={"message": msg, "recency": 0.9},
            )
        )

    for i, msg in enumerate(list(older_raw)):
        text = msg.get("text") or ""
        records.append(
            VectorRecord(
                record_id=f"sem:episodic:{session_id}:{i}",
                text=text,
                memory_type=MemoryType.EPISODIC_RAW,
                cluster_id="episodic",
                source_type="episodic",
                provenance={"message": msg, "recency": 0.4},
            )
        )

    for i, summary in enumerate(list(summaries)):
        text = summary.get("summary") or summary.get("text") or ""
        records.append(
            VectorRecord(
                record_id=f"sem:summary:{session_id}:{i}",
                text=text,
                memory_type=MemoryType.SUMMARY,
                cluster_id="summaries",
                source_type="summary",
                provenance={"summary": summary, "recency": 0.35},
            )
        )

    matches = backend.search(
        query=query,
        records=records,
        top_k=top_k,
        where=None,
    )

    for match in matches:
        provenance = dict(match.record.provenance or {})
        provenance["vector_backend"] = {
            "abstraction_used": True,
            "requested_backend": requested_backend,
            "selected_backend": match.backend,
            "fallback_used": bool(used_fallback),
            "source_type": match.debug.get("source_type"),
        }
        hits.append(
            _mk(
                match.record.record_id,
                match.record.text,
                provenance,
                sem=float(match.score),
                memory_type=match.record.memory_type,
                cluster=match.record.cluster_id,
            )
        )

    hits.sort(key=lambda h: (-h.semantic_score, h.id))
    return hits[:top_k]
