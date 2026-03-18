from __future__ import annotations

import sqlite3
from pathlib import Path

from CognitiveRAG.schemas.memory import ProfileFact


class ProfileStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL
                )
                """
            )

    def upsert(self, fact: ProfileFact) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO profile_facts(key, value, source, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    source=excluded.source,
                    confidence=excluded.confidence
                """,
                (fact.key, fact.value, fact.source, fact.confidence),
            )

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM profile_facts WHERE key = ?", (key,)).fetchone()
            return None if row is None else row[0]

    def latest_chunk(self) -> dict:
        """Return the most recent profile fact as a RetrievedChunk-compatible dict."""
        with self._connect() as conn:
            row = conn.execute("SELECT key, value, source FROM profile_facts ORDER BY rowid DESC LIMIT 1").fetchone()
            if not row:
                return {}
            key, value, source = row
            return {
                "chunk_id": key,
                "document_id": None,
                "text": value,
                "source_type": "profile",
                "score": 0.0,
                "metadata": {"source": source},
            }

    def query(self, query: str, top_k: int = 5) -> list:
        """Simple query over profile facts: score by token overlap against key and value."""
        qtokens = set(query.lower().split())
        results: list[tuple[float, dict]] = []
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value, source FROM profile_facts").fetchall()
            for key, value, source in rows:
                tokens = set((key + ' ' + value).lower().split())
                score = len(qtokens & tokens)
                if score > 0:
                    results.append((float(score), {
                        "chunk_id": key,
                        "document_id": None,
                        "text": value,
                        "source_type": "profile",
                        "score": float(score),
                        "metadata": {"source": source},
                    }))
        results.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
        out = [r[1] for r in results[:top_k]]
        if not out:
            lc = self.latest_chunk()
            if lc:
                out = [lc]
        return out
