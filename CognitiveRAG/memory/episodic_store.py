from __future__ import annotations

import sqlite3
from pathlib import Path

from CognitiveRAG.schemas.retrieval import RetrievedChunk
from CognitiveRAG.schemas.memory import EpisodicEvent


class EpisodicStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    result TEXT,
                    success_score REAL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def upsert(self, event: EpisodicEvent) -> None:
        # Ensure minimal metadata contract for episodic events
        meta = dict(event.metadata or {})
        # required fields
        meta.setdefault("source_type", "episodic")
        meta.setdefault("project", meta.get("project", "cognitiverag"))
        meta.setdefault("origin_id", meta.get("origin_id") or event.event_id)
        # timestamps
        ts = event.timestamp.isoformat()
        meta.setdefault("created_at", meta.get("created_at", ts))
        meta["updated_at"] = ts

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events(event_id, timestamp, event_type, goal, result, success_score, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    event_type=excluded.event_type,
                    goal=excluded.goal,
                    result=excluded.result,
                    success_score=excluded.success_score,
                    metadata_json=excluded.metadata_json
                """,
                (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.goal,
                    event.result,
                    event.success_score,
                    str(meta),
                ),
            )

    def query(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, goal, result
                FROM events
                WHERE goal LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (f"%{query}%", top_k),
            ).fetchall()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document_id=None,
                text=f"Goal: {row[1]}\nResult: {row[2] or ''}",
                source_type="episodic",
                score=1.0,
                metadata={},
            )
            for row in rows
        ]
