from __future__ import annotations

import json
import os
import sqlite3
from typing import List

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.retrieval.models import LaneHit


def _load_reasoning(workdir: str, limit: int = 10):
    db_path = os.path.join(workdir, "reasoning.sqlite3")
    if not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT pattern_id, solution_summary, confidence, provenance_json FROM reasoning_patterns ORDER BY rowid DESC LIMIT ?",
            (int(limit),),
        ).fetchall()


def retrieve(*, workdir: str, intent_family: IntentFamily, query: str, top_k: int = 6) -> List[LaneHit]:
    hits: List[LaneHit] = []
    rows = _load_reasoning(workdir, limit=max(8, top_k))

    for pattern_id, summary, confidence, provenance_json in rows:
        try:
            provenance = {"reasoning_provenance": json.loads(provenance_json) if provenance_json else []}
        except Exception:
            provenance = {"reasoning_provenance": []}
        text = summary or ""
        lexical = 0.2 if query else 0.0
        semantic = 0.25 if query else 0.0
        if intent_family in {IntentFamily.MEMORY_SUMMARY, IntentFamily.PLANNING}:
            semantic += 0.2
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
                trust_score=max(0.0, min(1.0, float(confidence or 0.5))),
                novelty_score=0.45,
                contradiction_risk=0.0,
                cluster_id="promoted",
                must_include=False,
                compressible=True,
            ).with_token_estimate()
        )

    hits.sort(key=lambda h: (-(h.semantic_score + h.lexical_score + h.trust_score), h.id))
    return hits[:top_k]
