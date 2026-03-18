"""
Lossless compaction / context-window foundation for session memory.

Placed under the canonical CognitiveRAG package path.
"""
from __future__ import annotations
import json
import os
from typing import Callable, List, Dict, Any, Optional

# Use package-local data dir relative to repo workspace
WORKDIR = os.path.join(os.getcwd(), "data", "session_memory")
os.makedirs(WORKDIR, exist_ok=True)


def _load_fallback_summaries(session_id: str) -> List[Dict[str, Any]]:
    path = os.path.join(WORKDIR, f"summaries_{session_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_fallback_summaries(session_id: str, summaries: List[Dict[str, Any]]):
    path = os.path.join(WORKDIR, f"summaries_{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


# Simple deterministic summarizer: join messages and truncate to N chars.
def _default_summarizer(messages: List[Dict[str, Any]], max_chars: int = 200) -> Dict[str, Any]:
    text = "\n".join(m.get("text", "") for m in messages)
    summary = text[:max_chars]
    return {"summary": summary, "source_count": len(messages), "token_est": len(summary)}


def compact_session(session_id: str, older_than_index: int, summarizer: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """Create derived summaries for messages with index < older_than_index.

    Same behavior as the workspace-level implementation but under the package.
    """
    summarizer = summarizer or _default_summarizer

    # Try to import existing ConversationStore API if available.
    try:
        from CognitiveRAG.session_memory.stores import ConversationStore
    except Exception:
        ConversationStore = None

    raw_messages = []
    if ConversationStore is not None:
        try:
            store = ConversationStore()
            raw_messages = store.list_messages(session_id)
        except Exception:
            ConversationStore = None

    if ConversationStore is None:
        raw_path = os.path.join(WORKDIR, f"raw_{session_id}.json")
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_messages = json.load(f)
        else:
            raw_messages = []

    older = [m for m in raw_messages if int(m.get("index", 0)) < int(older_than_index)]
    if not older:
        return []

    CHUNK_SIZE = 20
    chunks = [older[i : i + CHUNK_SIZE] for i in range(0, len(older), CHUNK_SIZE)]

    created = []
    for i, chunk in enumerate(chunks):
        artifact = summarizer(chunk)
        node = {
            "session_id": session_id,
            "chunk_index": i,
            "summary": artifact.get("summary"),
            "source_count": artifact.get("source_count", len(chunk)),
            "created_by": "context_window.v1",
        }
        try:
            from CognitiveRAG.session_memory.stores import SummaryNodeStore
            sstore = SummaryNodeStore()
            sstore.upsert(node)
        except Exception:
            existing = _load_fallback_summaries(session_id)
            existing.append(node)
            _save_fallback_summaries(session_id, existing)
        created.append(node)

    return created


def assemble_context(session_id: str, fresh_tail_count: int = 20, budget: int = 4096) -> Dict[str, Any]:
    """Assemble context from recent raw messages + older summaries.
    """
    try:
        from CognitiveRAG.session_memory.stores import ConversationStore
        store = ConversationStore()
        raw = list(store.list_messages(session_id))
    except Exception:
        raw_path = os.path.join(WORKDIR, f"raw_{session_id}.json")
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = []

    fresh_tail = sorted(raw, key=lambda m: int(m.get("index", 0)))[-fresh_tail_count:]

    summaries = []
    try:
        from CognitiveRAG.session_memory.stores import SummaryNodeStore
        sstore = SummaryNodeStore()
        summaries = list(sstore.list_for_session(session_id))
    except Exception:
        summaries = _load_fallback_summaries(session_id)

    summaries = sorted(summaries, key=lambda s: int(s.get("chunk_index", 0)))

    selected = []
    used = 0
    for s in summaries:
        size = len(str(s.get("summary") or ""))
        if used + size > budget:
            break
        selected.append(s)
        used += size

    return {"fresh_tail": fresh_tail, "summaries": selected}
