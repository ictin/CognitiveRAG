from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class WebFetchLogStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    query_variant TEXT,
                    status TEXT,
                    http_status INTEGER,
                    error TEXT,
                    result_count INTEGER,
                    fetched_at TEXT
                )
                """
            )

    def append(
        self,
        *,
        query: str,
        query_variant: str,
        status: str,
        http_status: int | None,
        error: str | None,
        result_count: int,
        fetched_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_fetch_log(query, query_variant, status, http_status, error, result_count, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (query, query_variant, status, http_status, error, int(result_count), fetched_at),
            )

    def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, query, query_variant, status, http_status, error, result_count, fetched_at
                FROM web_fetch_log ORDER BY id DESC LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]
