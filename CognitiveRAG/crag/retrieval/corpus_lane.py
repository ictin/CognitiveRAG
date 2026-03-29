from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List

from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit


def _read_context_items(workdir: str, limit: int = 30):
    db_path = os.path.join(workdir, "context_items.sqlite3")
    if not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT item_id, session_id, type, payload_json, created_at FROM context_items ORDER BY rowid DESC LIMIT ?",
            (int(limit),),
        ).fetchall()


def _is_corpus(type_name: str, payload: Dict[str, Any]) -> bool:
    t = (type_name or "").lower()
    return ("corpus" in t) or bool(payload.get("file_path")) or bool(payload.get("source_path"))


def retrieve(*, workdir: str, query: str, top_k: int = 8) -> List[LaneHit]:
    hits: List[LaneHit] = []
    rows = _read_context_items(workdir, limit=max(20, top_k * 2))
    qwords = set((query or "").lower().split())

    for item_id, session_id, type_name, payload_json, created_at in rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception:
            payload = {}
        if not _is_corpus(str(type_name), payload):
            continue

        text = payload.get("excerpt") or payload.get("summary") or payload.get("text") or ""
        if not text:
            continue

        words = set(text.lower().split())
        overlap = float(len(words & qwords)) / float(max(1, len(qwords))) if qwords else 0.0
        hits.append(
            LaneHit(
                id=f"corpus:{item_id}",
                lane=RetrievalLane.CORPUS,
                memory_type=MemoryType.CORPUS_CHUNK,
                text=text,
                provenance={"session_id": session_id, "type": type_name, "payload": payload, "created_at": created_at},
                lexical_score=overlap,
                semantic_score=max(0.2, overlap),
                recency_score=0.35,
                freshness_score=0.6,
                trust_score=0.75,
                novelty_score=0.5,
                contradiction_risk=0.0,
                cluster_id=str(payload.get("file_path") or payload.get("source_path") or "corpus"),
                must_include=False,
                compressible=True,
            ).with_token_estimate()
        )

    hits.sort(key=lambda h: (-(h.semantic_score + h.lexical_score), h.id))
    return hits[:top_k]
