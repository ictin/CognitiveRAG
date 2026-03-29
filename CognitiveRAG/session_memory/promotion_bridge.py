from __future__ import annotations
import os
import json
from typing import List, Dict, Any
from pathlib import Path

from CognitiveRAG.crag.promotion.bridge import promote_summaries_to_patterns
from CognitiveRAG.schemas.memory import ReasoningPattern


WORKDIR = os.path.join(os.getcwd(), "data", "session_memory")


def _load_fallback_summaries(session_id: str) -> List[Dict[str, Any]]:
    path = os.path.join(WORKDIR, f"summaries_{session_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def promote_session_summaries(session_id: str, reasoning_store=None, dry_run: bool = False) -> List[ReasoningPattern]:
    """Promote derived session summaries into durable ReasoningStore.

    - session_id: session identifier
    - reasoning_store: optional ReasoningStore instance; if omitted, the function will try to import CognitiveRAG.memory.reasoning_store.ReasoningStore
    - dry_run: if True, do not persist, just return the list of ReasoningPattern objects

    Behavior:
    - Reads summaries from SummaryNodeStore if available, otherwise fallback to data/session_memory/summaries_<session_id>.json
    - For each summary chunk creates deterministic pattern_id: sessprom:{session_id}:{sha1_16(summary_text)}
    - Builds ReasoningPattern with confidence=0.5 and provenance containing a single object with session_id, summary_chunk_index, source
    - Upserts into reasoning_store if provided or available
    - Returns the list of ReasoningPattern objects created
    """
    # Try to load summaries via SummaryNodeStore
    summaries = []
    try:
        from CognitiveRAG.session_memory.summary_nodes import SummaryNodeStore
        sstore = SummaryNodeStore()
        summaries = list(sstore.list_for_session(session_id))
    except Exception:
        summaries = _load_fallback_summaries(session_id)

    if not summaries:
        return []

    normalized = []
    for s in summaries:
        if isinstance(s, dict):
            normalized.append(s)
        else:
            normalized.append(
                {
                    "chunk_index": getattr(s, "chunk_index", None),
                    "summary": getattr(s, "summary", None),
                }
            )
    patterns = promote_summaries_to_patterns(session_id, normalized)

    if dry_run:
        return patterns

    # Persist to reasoning store if available
    rs = reasoning_store
    if rs is None:
        try:
            from CognitiveRAG.memory.reasoning_store import ReasoningStore
            try:
                from CognitiveRAG.core.settings import settings as _settings

                dbp = Path(getattr(_settings.store, "reasoning_db_path", Path(WORKDIR) / "reasoning.sqlite3"))
            except Exception:
                dbp = Path(WORKDIR) / "reasoning.sqlite3"
            rs = ReasoningStore(dbp)
        except Exception:
            rs = None

    if rs is not None:
        for p in patterns:
            try:
                rs.upsert(p)
            except Exception:
                # non-fatal: continue
                pass
    return patterns
