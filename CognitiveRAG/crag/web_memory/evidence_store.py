from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class WebEvidenceStore:
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
                CREATE TABLE IF NOT EXISTS web_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    query TEXT,
                    query_variant TEXT,
                    source_id TEXT,
                    url TEXT,
                    title TEXT,
                    snippet TEXT,
                    extracted_text TEXT,
                    fetched_at TEXT,
                    published_at TEXT,
                    updated_at TEXT,
                    trust_score REAL,
                    freshness_class TEXT,
                    content_hash TEXT,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_web_evidence_source_id ON web_evidence(source_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_web_evidence_query ON web_evidence(query)"
            )

    def upsert_evidence(self, evidence: Dict[str, Any]) -> str:
        source_id = str(evidence.get("source_id") or "")
        content_hash = str(evidence.get("content_hash") or "")
        evidence_id = f"{source_id}::{content_hash}" if source_id or content_hash else str(evidence.get("url") or "")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_evidence(
                    evidence_id, query, query_variant, source_id, url, title, snippet, extracted_text,
                    fetched_at, published_at, updated_at, trust_score, freshness_class, content_hash, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO UPDATE SET
                    query=excluded.query,
                    query_variant=excluded.query_variant,
                    source_id=excluded.source_id,
                    url=excluded.url,
                    title=excluded.title,
                    snippet=excluded.snippet,
                    extracted_text=excluded.extracted_text,
                    fetched_at=excluded.fetched_at,
                    published_at=excluded.published_at,
                    updated_at=excluded.updated_at,
                    trust_score=excluded.trust_score,
                    freshness_class=excluded.freshness_class,
                    content_hash=excluded.content_hash,
                    metadata_json=excluded.metadata_json
                """,
                (
                    evidence_id,
                    evidence.get("query"),
                    evidence.get("query_variant"),
                    evidence.get("source_id"),
                    evidence.get("url"),
                    evidence.get("title"),
                    evidence.get("snippet"),
                    evidence.get("extracted_text"),
                    evidence.get("fetched_at"),
                    evidence.get("published_at"),
                    evidence.get("updated_at"),
                    float(evidence.get("trust_score", 0.0)),
                    evidence.get("freshness_class"),
                    evidence.get("content_hash"),
                    json.dumps(evidence.get("raw") or {}),
                ),
            )
        return evidence_id

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = f"%{(query or '').strip()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT evidence_id, query, source_id, url, title, snippet, extracted_text, fetched_at,
                       published_at, updated_at, trust_score, freshness_class, content_hash, metadata_json
                FROM web_evidence
                WHERE query LIKE ? OR title LIKE ? OR snippet LIKE ? OR extracted_text LIKE ?
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                (q, q, q, q, int(top_k)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                raw = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                raw = {}
            out.append(
                {
                    "evidence_id": r["evidence_id"],
                    "query": r["query"],
                    "source_id": r["source_id"],
                    "url": r["url"],
                    "title": r["title"],
                    "snippet": r["snippet"],
                    "extracted_text": r["extracted_text"],
                    "fetched_at": r["fetched_at"],
                    "published_at": r["published_at"],
                    "updated_at": r["updated_at"],
                    "trust_score": r["trust_score"],
                    "freshness_class": r["freshness_class"],
                    "content_hash": r["content_hash"],
                    "raw": raw,
                }
            )
        return out
