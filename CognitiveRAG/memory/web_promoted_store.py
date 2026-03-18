from __future__ import annotations

import sqlite3
from pathlib import Path
import json
from datetime import datetime


class WebPromotedStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_promoted (
                    record_id TEXT PRIMARY KEY,
                    source_url TEXT,
                    page_content TEXT,
                    metadata_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )

    def upsert(self, record_id: str, source_url: str, page_content: str, metadata: dict) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_promoted(record_id, source_url, page_content, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    source_url=excluded.source_url,
                    page_content=excluded.page_content,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record_id,
                    source_url,
                    page_content,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )

    def get(self, record_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT record_id, source_url, page_content, metadata_json, created_at, updated_at FROM web_promoted WHERE record_id=?", (record_id,)).fetchone()
            if not row:
                return None
            rid, src, page, meta_json, created, updated = row
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except Exception:
                meta = {}
            return {
                'record_id': rid,
                'source_url': src,
                'page_content': page,
                'metadata': meta,
                'created_at': created,
                'updated_at': updated,
            }

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Very small heuristic search over page_content using LIKE. Returns up to top_k matches ordered by simple score (count of occurrences)."""
        q = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute("SELECT record_id, source_url, page_content, metadata_json FROM web_promoted WHERE page_content LIKE ?", (q,)).fetchall()
            scored = []
            for record_id, source_url, page_content, meta_json in rows:
                score = page_content.lower().count(query.lower())
                try:
                    meta = json.loads(meta_json) if meta_json else {}
                except Exception:
                    meta = {}
                scored.append((score, {
                    'record_id': record_id,
                    'source_url': source_url,
                    'page_content': page_content,
                    'metadata': meta,
                }))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item for score, item in scored[:top_k]]
