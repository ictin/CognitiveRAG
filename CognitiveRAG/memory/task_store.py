from __future__ import annotations

import sqlite3
from pathlib import Path

from CognitiveRAG.schemas.memory import TaskRecord


class TaskStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def upsert(self, task: TaskRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(task_id, title, status, summary, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    summary=excluded.summary,
                    metadata_json=excluded.metadata_json
                """,
                (task.task_id, task.title, task.status, task.summary, str(task.metadata)),
            )

    def latest_chunk(self) -> dict:
        """Return the most recent task as a RetrievedChunk-compatible dict."""
        with self._connect() as conn:
            row = conn.execute("SELECT task_id, title, summary, metadata_json FROM tasks ORDER BY rowid DESC LIMIT 1").fetchone()
            if not row:
                return {}
            task_id, title, summary, metadata_json = row
            return {
                "chunk_id": task_id,
                "document_id": None,
                "text": f"{title}: {summary}",
                "source_type": "task",
                "score": 0.0,
                "metadata": {"metadata_json": metadata_json},
            }

    def query(self, query: str, top_k: int = 5) -> list:
        """Simple query over tasks: score by token overlap against title and summary."""
        qtokens = set(query.lower().split())
        results: list[tuple[float, dict]] = []
        with self._connect() as conn:
            rows = conn.execute("SELECT task_id, title, summary, metadata_json FROM tasks").fetchall()
            for task_id, title, summary, metadata_json in rows:
                tokens = set((title + ' ' + summary).lower().split())
                score = len(qtokens & tokens)
                if score > 0:
                    results.append((float(score), {
                        "chunk_id": task_id,
                        "document_id": None,
                        "text": f"{title}: {summary}",
                        "source_type": "task",
                        "score": float(score),
                        "metadata": {"metadata_json": metadata_json},
                    }))
        results.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
        out = [r[1] for r in results[:top_k]]
        if not out:
            lc = self.latest_chunk()
            if lc:
                out = [lc]
        return out
