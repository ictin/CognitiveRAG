from __future__ import annotations

import sqlite3
from pathlib import Path


class MetadataStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                """
            )

    def upsert_document(self, document_id: str, source_path: str, content_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(document_id, source_path, content_hash)
                VALUES (?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    content_hash=excluded.content_hash
                """,
                (document_id, source_path, content_hash),
            )

    def replace_chunks(self, document_id: str, chunks: list[tuple[str, int, str, str]]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, document_id, chunk_index, text, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                chunks,
            )
