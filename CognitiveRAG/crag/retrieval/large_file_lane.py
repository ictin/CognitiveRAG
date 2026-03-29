from __future__ import annotations

import json
import os
import sqlite3
from typing import List

from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit


def _read_large_files(workdir: str, limit: int = 12):
    db_path = os.path.join(workdir, "large_files.sqlite3")
    if not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT record_id, file_path, metadata_json, created_at FROM large_files ORDER BY rowid DESC LIMIT ?",
            (int(limit),),
        ).fetchall()


def retrieve(*, workdir: str, query: str, top_k: int = 6) -> List[LaneHit]:
    hits: List[LaneHit] = []
    qwords = set((query or "").lower().split())
    for record_id, file_path, metadata_json, created_at in _read_large_files(workdir, max(10, top_k * 2)):
        try:
            meta = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            meta = {}
        text = meta.get("excerpt") or meta.get("summary") or meta.get("title") or f"Large file: {file_path}"
        words = set((text or "").lower().split())
        overlap = float(len(words & qwords)) / float(max(1, len(qwords))) if qwords else 0.0

        hits.append(
            LaneHit(
                id=f"large_file:{record_id}",
                lane=RetrievalLane.LARGE_FILE,
                memory_type=MemoryType.LARGE_FILE_EXCERPT,
                text=text,
                provenance={"file_path": file_path, "metadata": meta, "created_at": created_at},
                lexical_score=overlap,
                semantic_score=max(0.2, overlap),
                recency_score=0.4,
                freshness_score=0.55,
                trust_score=0.72,
                novelty_score=0.5,
                contradiction_risk=0.0,
                cluster_id=file_path,
                must_include=False,
                compressible=True,
            ).with_token_estimate()
        )

    hits.sort(key=lambda h: (-(h.semantic_score + h.lexical_score), h.id))
    return hits[:top_k]
