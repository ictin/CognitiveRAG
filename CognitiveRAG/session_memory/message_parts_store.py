import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

class MessagePartsStore:
    """Structured store for message parts.

    Backwards compatibility:
    - legacy calls can still use add_part/upsert_part with text/meta_json only.
    - get_parts keeps legacy keys while adding structured fields.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or 'data/session_memory/message_parts.sqlite3')
        self._init_db()

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS message_parts (
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                part_index INTEGER NOT NULL,
                text TEXT,
                meta_json TEXT,
                part_type TEXT,
                status TEXT,
                retry_of_part_index INTEGER,
                tool_name TEXT,
                tool_call_id TEXT,
                file_refs_json TEXT,
                PRIMARY KEY(session_id, message_id, part_index)
            )
            ''')
            existing = {row[1] for row in conn.execute("PRAGMA table_info(message_parts)").fetchall()}
            required = {
                "part_type": "TEXT",
                "status": "TEXT",
                "retry_of_part_index": "INTEGER",
                "tool_name": "TEXT",
                "tool_call_id": "TEXT",
                "file_refs_json": "TEXT",
            }
            for col, col_type in required.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE message_parts ADD COLUMN {col} {col_type}")

    def _coerce_meta(self, meta_json: Any) -> Dict[str, Any]:
        if meta_json is None:
            return {}
        if isinstance(meta_json, dict):
            return dict(meta_json)
        if isinstance(meta_json, str):
            try:
                parsed = json.loads(meta_json)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _normalize_structured_fields(
        self,
        text: Optional[str],
        meta_json: Any = None,
        part_type: Optional[str] = None,
        status: Optional[str] = None,
        retry_of_part_index: Optional[int] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        file_refs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        meta = self._coerce_meta(meta_json)
        normalized_file_refs = file_refs
        if normalized_file_refs is None:
            raw_file_refs = meta.get("file_refs", [])
            normalized_file_refs = raw_file_refs if isinstance(raw_file_refs, list) else []

        return {
            "text": text,
            "meta": meta,
            "part_type": part_type or meta.get("part_type") or "text",
            "status": status if status is not None else meta.get("status"),
            "retry_of_part_index": (
                retry_of_part_index
                if retry_of_part_index is not None
                else meta.get("retry_of_part_index")
            ),
            "tool_name": tool_name or meta.get("tool_name"),
            "tool_call_id": tool_call_id or meta.get("tool_call_id"),
            "file_refs": normalized_file_refs or [],
        }

    def add_part(self, session_id: str, message_id: str, part_index: int, text: str, meta_json: Any = None):
        norm = self._normalize_structured_fields(text=text, meta_json=meta_json)
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT INTO message_parts("
                    "session_id, message_id, part_index, text, meta_json, part_type, status, "
                    "retry_of_part_index, tool_name, tool_call_id, file_refs_json"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id, message_id, part_index) DO UPDATE SET "
                    "text=excluded.text, meta_json=excluded.meta_json, part_type=excluded.part_type, "
                    "status=excluded.status, retry_of_part_index=excluded.retry_of_part_index, "
                    "tool_name=excluded.tool_name, tool_call_id=excluded.tool_call_id, "
                    "file_refs_json=excluded.file_refs_json"
                ),
                (
                    session_id,
                    message_id,
                    part_index,
                    norm["text"],
                    json.dumps(norm["meta"]) if norm["meta"] else None,
                    norm["part_type"],
                    norm["status"],
                    norm["retry_of_part_index"],
                    norm["tool_name"],
                    norm["tool_call_id"],
                    json.dumps(norm["file_refs"]) if norm["file_refs"] else None,
                )
            )
        return True

    def upsert_part(self, session_id: str, message_id: str, part_index: int, text: str, meta_json: Any = None):
        return self.add_part(session_id, message_id, part_index, text, meta_json)

    def upsert_structured_part(
        self,
        session_id: str,
        message_id: str,
        part_index: int,
        text: Optional[str] = None,
        meta_json: Any = None,
        part_type: Optional[str] = None,
        status: Optional[str] = None,
        retry_of_part_index: Optional[int] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        file_refs: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        norm = self._normalize_structured_fields(
            text=text,
            meta_json=meta_json,
            part_type=part_type,
            status=status,
            retry_of_part_index=retry_of_part_index,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            file_refs=file_refs,
        )
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT INTO message_parts("
                    "session_id, message_id, part_index, text, meta_json, part_type, status, "
                    "retry_of_part_index, tool_name, tool_call_id, file_refs_json"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id, message_id, part_index) DO UPDATE SET "
                    "text=excluded.text, meta_json=excluded.meta_json, part_type=excluded.part_type, "
                    "status=excluded.status, retry_of_part_index=excluded.retry_of_part_index, "
                    "tool_name=excluded.tool_name, tool_call_id=excluded.tool_call_id, "
                    "file_refs_json=excluded.file_refs_json"
                ),
                (
                    session_id,
                    message_id,
                    part_index,
                    norm["text"],
                    json.dumps(norm["meta"]) if norm["meta"] else None,
                    norm["part_type"],
                    norm["status"],
                    norm["retry_of_part_index"],
                    norm["tool_name"],
                    norm["tool_call_id"],
                    json.dumps(norm["file_refs"]) if norm["file_refs"] else None,
                ),
            )
        return True

    def _decode_json(self, raw: Optional[str]) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _row_to_part(self, row: sqlite3.Row) -> Dict[str, Any]:
        meta = self._decode_json(row[2])
        file_refs = self._decode_json(row[8]) or []
        return {
            "part_index": row[0],
            "text": row[1],
            "meta_json": row[2],
            "part_type": row[3] or ((meta or {}).get("part_type")) or "text",
            "status": row[4] if row[4] is not None else (meta or {}).get("status"),
            "retry_of_part_index": row[5] if row[5] is not None else (meta or {}).get("retry_of_part_index"),
            "tool_name": row[6] if row[6] is not None else (meta or {}).get("tool_name"),
            "tool_call_id": row[7] if row[7] is not None else (meta or {}).get("tool_call_id"),
            "file_refs": file_refs if isinstance(file_refs, list) else [],
            "meta": meta if isinstance(meta, dict) else {},
        }

    def get_parts(self, session_id: str, message_id: str):
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT part_index, text, meta_json, part_type, status, retry_of_part_index, "
                    "tool_name, tool_call_id, file_refs_json "
                    "FROM message_parts WHERE session_id=? AND message_id=? ORDER BY part_index"
                ),
                (session_id, message_id),
            ).fetchall()
            return [self._row_to_part(r) for r in rows]

    def get_parts_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT message_id, part_index, text, meta_json, part_type, status, retry_of_part_index, "
                    "tool_name, tool_call_id, file_refs_json "
                    "FROM message_parts WHERE session_id=? ORDER BY message_id, part_index"
                ),
                (session_id,),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                part = self._row_to_part(
                    (
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                        row[9],
                    )
                )
                part["message_id"] = row[0]
                out.append(part)
            return out
