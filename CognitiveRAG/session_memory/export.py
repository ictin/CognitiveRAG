from __future__ import annotations

from typing import Any, Dict, List, Optional

from .conversation_store import ConversationStore
from .message_parts_store import MessagePartsStore


def _parts_db_from_prefix(db_prefix: Optional[str]) -> Optional[str]:
    if not db_prefix:
        return None
    import os

    legacy = os.path.join(db_prefix, "parts.sqlite3")
    canonical = os.path.join(db_prefix, "message_parts.sqlite3")
    if os.path.exists(legacy):
        return legacy
    if os.path.exists(canonical):
        return canonical
    return canonical


def export_session_with_parts(session_id: str, db_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Export session as structured messages + structured message parts.

    Output shape is intentionally deterministic and stable for tests.
    """
    conv = ConversationStore(db_path=(db_prefix + "/conversations.sqlite3") if db_prefix else None)
    parts = MessagePartsStore(db_path=_parts_db_from_prefix(db_prefix) if db_prefix else None)

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

    return {
        "session_id": session_id,
        "structured_parts": True,
        "messages": exported,
        "part_stats": {
            "message_count": len(exported),
            "part_count": total_parts,
        },
    }
