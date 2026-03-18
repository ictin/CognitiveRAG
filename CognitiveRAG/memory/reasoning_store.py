from __future__ import annotations

import sqlite3
from pathlib import Path

from CognitiveRAG.schemas.memory import ReasoningPattern
import json


class ReasoningStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reasoning_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    problem_signature TEXT NOT NULL,
                    reasoning_steps_json TEXT NOT NULL,
                    solution_summary TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    provenance_json TEXT
                )
                """
            )

    def upsert(self, pattern: ReasoningPattern) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reasoning_patterns(pattern_id, problem_signature, reasoning_steps_json, solution_summary, confidence, provenance_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    problem_signature=excluded.problem_signature,
                    reasoning_steps_json=excluded.reasoning_steps_json,
                    solution_summary=excluded.solution_summary,
                    confidence=excluded.confidence,
                    provenance_json=excluded.provenance_json
                """,
                (
                    pattern.pattern_id,
                    pattern.problem_signature,
                    str(pattern.reasoning_steps),
                    pattern.solution_summary,
                    pattern.confidence,
                    json.dumps(pattern.provenance or []),
                ),
            )

    def latest_chunk(self) -> dict:
        """Return the most recent reasoning pattern as a RetrievedChunk-compatible dict."""
        with self._connect() as conn:
            try:
                row = conn.execute("SELECT pattern_id, solution_summary, provenance_json FROM reasoning_patterns ORDER BY rowid DESC LIMIT 1").fetchone()
            except sqlite3.OperationalError:
                row = conn.execute("SELECT pattern_id, solution_summary FROM reasoning_patterns ORDER BY rowid DESC LIMIT 1").fetchone()
                if not row:
                    return {}
                pattern_id, solution_summary = row
                return {
                    "chunk_id": pattern_id,
                    "document_id": None,
                    "text": solution_summary,
                    "source_type": "reasoning",
                    "score": 0.0,
                    "metadata": {},
                }
            if not row:
                return {}
            pattern_id, solution_summary, provenance_json = row
            meta = {}
            try:
                meta['provenance'] = json.loads(provenance_json) if provenance_json else []
            except Exception:
                meta['provenance'] = []
            return {
                "chunk_id": pattern_id,
                "document_id": None,
                "text": solution_summary,
                "source_type": "reasoning",
                "score": 0.0,
                "metadata": meta,
            }

    def query(self, query: str, top_k: int = 5) -> list:
        """Simple query over reasoning patterns: score by token overlap against signature and solution_summary."""
        qtokens = set(query.lower().split())
        results: list[tuple[float, dict]] = []
        with self._connect() as conn:
            try:
                rows = conn.execute("SELECT pattern_id, problem_signature, solution_summary, provenance_json FROM reasoning_patterns").fetchall()
                extracted = []
                for pattern_id, problem_signature, solution_summary, provenance_json in rows:
                    extracted.append((pattern_id, problem_signature, solution_summary, provenance_json))
            except sqlite3.OperationalError:
                # Older DB schema without provenance_json; fall back gracefully
                rows = conn.execute("SELECT pattern_id, problem_signature, solution_summary FROM reasoning_patterns").fetchall()
                extracted = [(pattern_id, problem_signature, solution_summary, None) for (pattern_id, problem_signature, solution_summary) in rows]

            for pattern_id, problem_signature, solution_summary, provenance_json in extracted:
                tokens = set((problem_signature + ' ' + solution_summary).lower().split())
                score = len(qtokens & tokens)
                if score > 0:
                    meta = {}
                    try:
                        meta['provenance'] = json.loads(provenance_json) if provenance_json else []
                    except Exception:
                        meta['provenance'] = []
                    results.append((float(score), {
                        "chunk_id": pattern_id,
                        "document_id": None,
                        "text": solution_summary,
                        "source_type": "reasoning",
                        "score": float(score),
                        "metadata": meta,
                    }))
        results.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
        out = [r[1] for r in results[:top_k]]
        if not out:
            lc = self.latest_chunk()
            if lc:
                out = [lc]
        return out
