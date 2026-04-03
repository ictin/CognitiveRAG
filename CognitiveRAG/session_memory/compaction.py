from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_index(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _message_key(message: Dict[str, Any], idx: int) -> str:
    mid = str(message.get("message_id") or "").strip()
    if mid:
        return f"message_id:{mid}"
    return f"index:{_norm_index(message.get('index'), idx)}"


def _segment_id(session_id: str, lineage: Sequence[Dict[str, Any]]) -> str:
    tokens: list[str] = []
    for idx, row in enumerate(lineage):
        msg_key = str(row.get("message_key") or f"index:{idx}")
        msg_index = str(_norm_index(row.get("index"), idx))
        tokens.append(f"{msg_key}|{msg_index}")
    payload = f"{session_id}|{'||'.join(tokens)}"
    return f"compact:{hashlib.sha1(payload.encode('utf-8')).hexdigest()}"


class SessionCompactionStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or "data/session_memory/compaction.sqlite3")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compacted_segments (
                    session_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_index INTEGER NOT NULL,
                    end_index INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    source_count INTEGER NOT NULL,
                    policy_reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    lineage_json TEXT NOT NULL,
                    raw_snapshot_json TEXT NOT NULL,
                    metadata_json TEXT,
                    PRIMARY KEY (session_id, segment_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quarantined_messages (
                    session_id TEXT NOT NULL,
                    message_key TEXT NOT NULL,
                    msg_index INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    PRIMARY KEY (session_id, message_key)
                )
                """
            )

    def upsert_segment(
        self,
        *,
        session_id: str,
        segment_id: str,
        chunk_index: int,
        start_index: int,
        end_index: int,
        summary: str,
        source_count: int,
        policy_reason: str,
        status: str,
        lineage: List[Dict[str, Any]],
        raw_snapshot: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO compacted_segments(
                    session_id, segment_id, chunk_index, start_index, end_index,
                    summary, source_count, policy_reason, status, created_at,
                    lineage_json, raw_snapshot_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, segment_id) DO UPDATE SET
                    chunk_index=excluded.chunk_index,
                    start_index=excluded.start_index,
                    end_index=excluded.end_index,
                    summary=excluded.summary,
                    source_count=excluded.source_count,
                    policy_reason=excluded.policy_reason,
                    status=excluded.status,
                    lineage_json=excluded.lineage_json,
                    raw_snapshot_json=excluded.raw_snapshot_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    session_id,
                    segment_id,
                    int(chunk_index),
                    int(start_index),
                    int(end_index),
                    str(summary or ""),
                    int(source_count),
                    str(policy_reason),
                    str(status),
                    _now_iso(),
                    json.dumps(lineage, ensure_ascii=False),
                    json.dumps(raw_snapshot, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def get_segment(self, session_id: str, segment_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    session_id, segment_id, chunk_index, start_index, end_index,
                    summary, source_count, policy_reason, status, created_at,
                    lineage_json, raw_snapshot_json, metadata_json
                FROM compacted_segments
                WHERE session_id=? AND segment_id=?
                """,
                (session_id, segment_id),
            ).fetchone()
        if not row:
            return None
        return {
            "session_id": row[0],
            "segment_id": row[1],
            "chunk_index": int(row[2]),
            "start_index": int(row[3]),
            "end_index": int(row[4]),
            "summary": row[5],
            "source_count": int(row[6]),
            "policy_reason": row[7],
            "status": row[8],
            "created_at": row[9],
            "lineage": json.loads(row[10] or "[]"),
            "raw_snapshot": json.loads(row[11] or "[]"),
            "metadata": json.loads(row[12] or "{}"),
        }

    def list_segments(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT segment_id FROM compacted_segments
                WHERE session_id=?
                ORDER BY chunk_index, segment_id
                """,
                (session_id,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            seg = self.get_segment(session_id, row[0])
            if seg:
                out.append(seg)
        return out

    def upsert_quarantined(
        self,
        *,
        session_id: str,
        message_key: str,
        msg_index: int,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO quarantined_messages(session_id, message_key, msg_index, reason, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, message_key) DO UPDATE SET
                    msg_index=excluded.msg_index,
                    reason=excluded.reason,
                    metadata_json=excluded.metadata_json
                """,
                (
                    session_id,
                    str(message_key),
                    int(msg_index),
                    str(reason),
                    _now_iso(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def list_quarantined(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT message_key, msg_index, reason, created_at, metadata_json
                FROM quarantined_messages
                WHERE session_id=?
                ORDER BY msg_index, message_key
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "message_key": row[0],
                "index": int(row[1]),
                "reason": row[2],
                "created_at": row[3],
                "metadata": json.loads(row[4] or "{}"),
            }
            for row in rows
        ]

    def stats(self, session_id: str) -> Dict[str, int]:
        with self._connect() as conn:
            seg_count = int(
                conn.execute("SELECT COUNT(*) FROM compacted_segments WHERE session_id=?", (session_id,)).fetchone()[0]
            )
            quarantine_count = int(
                conn.execute("SELECT COUNT(*) FROM quarantined_messages WHERE session_id=?", (session_id,)).fetchone()[0]
            )
        return {
            "compacted_segments": seg_count,
            "quarantined_messages": quarantine_count,
        }


def is_low_value_message(message: Dict[str, Any]) -> bool:
    text = _normalize_text(message.get("text"))
    if not text:
        return True
    if len(text) <= 2:
        return True
    if text.lower() in {"ok", "k", "yes", "no", "thx", "thanks", "done"} and len(text) <= 6:
        return True
    return False


def compute_eligible_messages(
    *,
    raw_messages: Sequence[Dict[str, Any]],
    older_than_index: int,
    already_compacted_keys: set[str],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    older = [m for m in list(raw_messages) if _norm_index(m.get("index"), 0) < int(older_than_index)]
    older.sort(key=lambda m: _norm_index(m.get("index"), 0))
    compactable: list[Dict[str, Any]] = []
    quarantined: list[Dict[str, Any]] = []
    for idx, row in enumerate(older):
        mkey = _message_key(row, idx)
        if mkey in already_compacted_keys:
            continue
        if is_low_value_message(row):
            quarantined.append(row)
            continue
        compactable.append(row)
    return compactable, quarantined


def build_lineage(messages: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    lineage: list[Dict[str, Any]] = []
    for idx, row in enumerate(messages):
        lineage.append(
            {
                "message_key": _message_key(row, idx),
                "message_id": row.get("message_id"),
                "index": _norm_index(row.get("index"), idx),
                "created_at": row.get("created_at"),
            }
        )
    return lineage


def build_raw_snapshot(messages: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for idx, row in enumerate(messages):
        out.append(
            {
                "message_key": _message_key(row, idx),
                "message_id": row.get("message_id"),
                "index": _norm_index(row.get("index"), idx),
                "sender": row.get("sender"),
                "text": _normalize_text(row.get("text")),
                "created_at": row.get("created_at"),
            }
        )
    return out


def recover_segment_messages(
    *,
    segment: Dict[str, Any],
    raw_messages: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    lineage = list(segment.get("lineage") or [])
    if not lineage:
        return []
    by_message_id = {
        str(row.get("message_id")): row
        for row in list(raw_messages)
        if str(row.get("message_id") or "")
    }
    by_index = {int(_norm_index(row.get("index"), i)): row for i, row in enumerate(list(raw_messages))}
    recovered: list[Dict[str, Any]] = []
    for idx, lrow in enumerate(lineage):
        mid = str(lrow.get("message_id") or "")
        msg_index = _norm_index(lrow.get("index"), idx)
        raw = by_message_id.get(mid) if mid else None
        if raw is None:
            raw = by_index.get(msg_index)
        if raw is not None:
            recovered.append(dict(raw))
            continue
        # Fallback to snapshot if raw source is unavailable.
        snaps = list(segment.get("raw_snapshot") or [])
        snap = None
        for srow in snaps:
            if mid and str(srow.get("message_id") or "") == mid:
                snap = srow
                break
            if int(_norm_index(srow.get("index"), -1)) == msg_index:
                snap = srow
                break
        if snap is not None:
            recovered.append(
                {
                    "message_id": snap.get("message_id"),
                    "index": _norm_index(snap.get("index"), msg_index),
                    "sender": snap.get("sender"),
                    "text": _normalize_text(snap.get("text")),
                    "created_at": snap.get("created_at"),
                    "recovered_from": "compaction_snapshot",
                }
            )
    recovered.sort(key=lambda row: _norm_index(row.get("index"), 0))
    return recovered


def summarize_compaction_state(session_id: str, store: SessionCompactionStore) -> Dict[str, Any]:
    stats = store.stats(session_id)
    segs = store.list_segments(session_id)
    return {
        "session_id": session_id,
        "stats": stats,
        "segments": [
            {
                "segment_id": s["segment_id"],
                "chunk_index": s["chunk_index"],
                "start_index": s["start_index"],
                "end_index": s["end_index"],
                "source_count": s["source_count"],
                "status": s["status"],
                "policy_reason": s["policy_reason"],
            }
            for s in segs
        ],
        "quarantined": store.list_quarantined(session_id),
    }
