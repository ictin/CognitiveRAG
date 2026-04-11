from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .conversation_store import ConversationStore
from .compaction import SessionCompactionStore
from .message_parts_store import MessagePartsStore


def _parts_db_from_prefix(db_prefix: Optional[str]) -> Optional[str]:
    if not db_prefix:
        return None

    legacy = os.path.join(db_prefix, "parts.sqlite3")
    canonical = os.path.join(db_prefix, "message_parts.sqlite3")
    if os.path.exists(legacy):
        return legacy
    if os.path.exists(canonical):
        return canonical
    return canonical


def _compaction_db_from_prefix(db_prefix: Optional[str]) -> Optional[str]:
    if not db_prefix:
        return None
    return os.path.join(db_prefix, "compaction.sqlite3")


def export_session_with_parts(session_id: str, db_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Export session as structured messages + structured message parts.

    Output shape is intentionally deterministic and stable for tests.
    """
    conv = ConversationStore(db_path=(db_prefix + "/conversations.sqlite3") if db_prefix else None)
    parts = MessagePartsStore(db_path=_parts_db_from_prefix(db_prefix) if db_prefix else None)
    compaction = SessionCompactionStore(db_path=_compaction_db_from_prefix(db_prefix) if db_prefix else None)

    messages = conv.get_messages(session_id)
    exported: List[Dict[str, Any]] = []
    total_parts = 0

    for message in messages:
        message_id = message["message_id"]
        message_parts = parts.get_parts(session_id, message_id)
        total_parts += len(message_parts)
        exported.append(
            {
                "message_id": message_id,
                "sender": message.get("sender"),
                "text": message.get("text"),
                "created_at": message.get("created_at"),
                "parts": message_parts,
            }
        )

    compaction_segments = compaction.list_segments(session_id)
    quarantined = compaction.list_quarantined(session_id)
    compaction_export = [
        {
            "segment_id": seg.get("segment_id"),
            "chunk_index": seg.get("chunk_index"),
            "start_index": seg.get("start_index"),
            "end_index": seg.get("end_index"),
            "summary": seg.get("summary"),
            "source_count": seg.get("source_count"),
            "policy_reason": seg.get("policy_reason"),
            "status": seg.get("status"),
            "created_at": seg.get("created_at"),
            "lineage": list(seg.get("lineage") or []),
            "raw_snapshot": list(seg.get("raw_snapshot") or []),
            "metadata": dict(seg.get("metadata") or {}),
        }
        for seg in compaction_segments
    ]

    return {
        "session_id": session_id,
        "structured_parts": True,
        "messages": exported,
        "part_stats": {
            "message_count": len(exported),
            "part_count": total_parts,
        },
        "compaction": {
            "segment_count": len(compaction_export),
            "quarantined_count": len(quarantined),
            "segments": compaction_export,
            "quarantined": quarantined,
        },
    }
