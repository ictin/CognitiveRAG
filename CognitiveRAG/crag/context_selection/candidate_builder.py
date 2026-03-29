from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.contracts.types import estimate_tokens


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _overlap_score(query: str, text: str) -> float:
    q = set(_norm(query).split())
    t = set(_norm(text).split())
    if not q or not t:
        return 0.0
    return float(len(q & t)) / float(max(1, len(q)))


def _make_candidate(
    *,
    cid: str,
    lane: RetrievalLane,
    memory_type: MemoryType,
    text: str,
    provenance: Dict[str, Any],
    query: str,
    recency_score: float,
    freshness_score: float,
    trust_score: float,
    cluster_id: str | None,
    must_include: bool,
    compressible: bool,
) -> ContextCandidate:
    lexical = _overlap_score(query, text)
    semantic = lexical  # deterministic placeholder until semantic model hook is plugged in.
    novelty = 1.0 if lexical > 0 else 0.35
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=memory_type,
        text=text,
        tokens=estimate_tokens(text),
        provenance=provenance,
        lexical_score=lexical,
        semantic_score=semantic,
        recency_score=recency_score,
        freshness_score=freshness_score,
        trust_score=trust_score,
        novelty_score=novelty,
        contradiction_risk=0.0,
        cluster_id=cluster_id,
        must_include=must_include,
        compressible=compressible,
    )


def _load_reasoning_candidates(workdir: str, query: str) -> List[ContextCandidate]:
    out: List[ContextCandidate] = []
    db_path = os.path.join(workdir, "reasoning.sqlite3")
    if not os.path.exists(db_path):
        return out
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT pattern_id, solution_summary, confidence, provenance_json FROM reasoning_patterns ORDER BY rowid DESC LIMIT 8"
        ).fetchall()
    for pattern_id, summary, confidence, provenance_json in rows:
        try:
            provenance = {"reasoning_provenance": json.loads(provenance_json) if provenance_json else []}
        except Exception:
            provenance = {"reasoning_provenance": []}
        out.append(
            _make_candidate(
                cid=f"promoted:{pattern_id}",
                lane=RetrievalLane.PROMOTED,
                memory_type=MemoryType.PROMOTED_FACT,
                text=summary or "",
                provenance=provenance,
                query=query,
                recency_score=0.55,
                freshness_score=0.7,
                trust_score=max(0.0, min(1.0, float(confidence or 0.5))),
                cluster_id="promoted",
                must_include=False,
                compressible=True,
            )
        )
    return out


def _load_large_file_candidates(workdir: str, query: str) -> List[ContextCandidate]:
    out: List[ContextCandidate] = []
    db_path = os.path.join(workdir, "large_files.sqlite3")
    if not os.path.exists(db_path):
        return out
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT record_id, file_path, metadata_json, created_at FROM large_files ORDER BY rowid DESC LIMIT 8"
        ).fetchall()
    for record_id, file_path, metadata_json, created_at in rows:
        text = ""
        try:
            meta = json.loads(metadata_json) if metadata_json else {}
            text = (meta.get("summary") or meta.get("excerpt") or meta.get("title") or "")
        except Exception:
            meta = {}
        if not text:
            text = f"Large file reference: {file_path}"
        out.append(
            _make_candidate(
                cid=f"large_file:{record_id}",
                lane=RetrievalLane.LARGE_FILE,
                memory_type=MemoryType.LARGE_FILE_EXCERPT,
                text=text,
                provenance={"file_path": file_path, "metadata": meta, "created_at": created_at},
                query=query,
                recency_score=0.45,
                freshness_score=0.55,
                trust_score=0.7,
                cluster_id=file_path,
                must_include=False,
                compressible=True,
            )
        )
    return out


def _load_context_item_candidates(workdir: str, query: str) -> List[ContextCandidate]:
    out: List[ContextCandidate] = []
    db_path = os.path.join(workdir, "context_items.sqlite3")
    if not os.path.exists(db_path):
        return out
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT item_id, session_id, type, payload_json, created_at FROM context_items ORDER BY rowid DESC LIMIT 20"
        ).fetchall()
    for item_id, session_id, type_name, payload_json, created_at in rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception:
            payload = {}
        text = payload.get("text") or payload.get("summary") or payload.get("excerpt") or ""
        if not text:
            continue
        lane = RetrievalLane.CORPUS if "corpus" in str(type_name).lower() else RetrievalLane.EPISODIC
        mem_type = MemoryType.CORPUS_CHUNK if lane == RetrievalLane.CORPUS else MemoryType.EPISODIC_RAW
        out.append(
            _make_candidate(
                cid=f"context_item:{item_id}",
                lane=lane,
                memory_type=mem_type,
                text=text,
                provenance={"session_id": session_id, "type": type_name, "created_at": created_at, "payload": payload},
                query=query,
                recency_score=0.35,
                freshness_score=0.6,
                trust_score=0.65,
                cluster_id=str(payload.get("source") or payload.get("file_path") or "context_item"),
                must_include=False,
                compressible=True,
            )
        )
    return out


def build_candidates(
    *,
    session_id: str,
    query: str,
    fresh_tail: Iterable[Dict[str, Any]],
    older_raw: Iterable[Dict[str, Any]],
    summaries: Iterable[Dict[str, Any]],
    workdir: str,
    intent_family: IntentFamily,
) -> List[ContextCandidate]:
    candidates: List[ContextCandidate] = []

    fresh_tail_list = list(fresh_tail)
    total_tail = max(1, len(fresh_tail_list))
    for idx, msg in enumerate(fresh_tail_list):
        text = msg.get("text") or ""
        msg_id = str(msg.get("message_id") or msg.get("index") or idx)
        recency = 1.0 - (float(total_tail - idx - 1) / float(total_tail))
        candidates.append(
            _make_candidate(
                cid=f"fresh:{session_id}:{msg_id}",
                lane=RetrievalLane.FRESH_TAIL,
                memory_type=MemoryType.EPISODIC_RAW,
                text=text,
                provenance={"session_id": session_id, "message": msg},
                query=query,
                recency_score=recency,
                freshness_score=0.95,
                trust_score=0.8,
                cluster_id="session_tail",
                must_include=True,
                compressible=False,
            )
        )

    older_raw_list = list(older_raw)
    total_old = max(1, len(older_raw_list))
    for idx, msg in enumerate(older_raw_list):
        text = msg.get("text") or ""
        if not text:
            continue
        msg_id = str(msg.get("message_id") or msg.get("index") or idx)
        recency = 1.0 - (float(total_old - idx) / float(total_old + 1))
        candidates.append(
            _make_candidate(
                cid=f"episodic:{session_id}:{msg_id}",
                lane=RetrievalLane.EPISODIC,
                memory_type=MemoryType.EPISODIC_RAW,
                text=text,
                provenance={"session_id": session_id, "message": msg},
                query=query,
                recency_score=max(0.05, recency),
                freshness_score=0.65,
                trust_score=0.8,
                cluster_id="session_history",
                must_include=False,
                compressible=True,
            )
        )

    for sidx, summary in enumerate(list(summaries)):
        text = summary.get("summary") or summary.get("text") or ""
        if not text:
            continue
        chunk_index = summary.get("chunk_index", sidx)
        candidates.append(
            _make_candidate(
                cid=f"summary:{session_id}:{chunk_index}",
                lane=RetrievalLane.SESSION_SUMMARY,
                memory_type=MemoryType.SUMMARY,
                text=text,
                provenance={"session_id": session_id, "summary": summary},
                query=query,
                recency_score=0.4,
                freshness_score=0.6,
                trust_score=0.75,
                cluster_id="session_summaries",
                must_include=False,
                compressible=True,
            )
        )

    if intent_family in {IntentFamily.MEMORY_SUMMARY, IntentFamily.PLANNING, IntentFamily.INVESTIGATIVE}:
        candidates.extend(_load_reasoning_candidates(workdir, query))

    if intent_family in {IntentFamily.CORPUS_OVERVIEW, IntentFamily.INVESTIGATIVE}:
        candidates.extend(_load_context_item_candidates(workdir, query))
        candidates.extend(_load_large_file_candidates(workdir, query))

    if intent_family in {IntentFamily.ARCHITECTURE_EXPLANATION, IntentFamily.MEMORY_SUMMARY}:
        candidates.append(
            _make_candidate(
                cid="architecture:stack",
                lane=RetrievalLane.ARCHITECTURE,
                memory_type=MemoryType.ARCHITECTURE_NOTE,
                text=(
                    "Memory stack: active context engine + backend/session memory + "
                    "lossless session memory + corpus/large-file retrieval + markdown mirror fallback."
                ),
                provenance={"source": "backend_contract"},
                query=query,
                recency_score=0.5,
                freshness_score=0.9,
                trust_score=0.95,
                cluster_id="architecture",
                must_include=False,
                compressible=False,
            )
        )

    return candidates
