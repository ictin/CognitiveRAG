from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class WebPromotedMemoryStore:
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
                CREATE TABLE IF NOT EXISTS web_promoted_memory (
                    promoted_id TEXT PRIMARY KEY,
                    canonical_fact TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    freshness_state TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )

    def upsert_fact(
        self,
        *,
        promoted_id: str,
        canonical_fact: str,
        evidence_ids: List[str],
        confidence: float,
        freshness_state: str,
        metadata: Dict[str, Any] | None = None,
        now_iso: str | None = None,
    ) -> None:
        now = now_iso or ""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_promoted_memory(
                    promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(promoted_id) DO UPDATE SET
                    canonical_fact=excluded.canonical_fact,
                    evidence_ids_json=excluded.evidence_ids_json,
                    confidence=excluded.confidence,
                    freshness_state=excluded.freshness_state,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    promoted_id,
                    canonical_fact,
                    json.dumps(list(evidence_ids or [])),
                    float(confidence),
                    freshness_state,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = f"%{(query or '').strip()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT promoted_id, canonical_fact, evidence_ids_json, confidence, freshness_state, metadata_json, created_at, updated_at
                FROM web_promoted_memory
                WHERE canonical_fact LIKE ?
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (q, int(top_k)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                evidence_ids = json.loads(r["evidence_ids_json"]) if r["evidence_ids_json"] else []
            except Exception:
                evidence_ids = []
            try:
                metadata = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                metadata = {}
            out.append(
                {
                    "promoted_id": r["promoted_id"],
                    "canonical_fact": r["canonical_fact"],
                    "evidence_ids": evidence_ids,
                    "confidence": r["confidence"],
                    "freshness_state": r["freshness_state"],
                    "metadata": metadata,
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
            )
        return out
