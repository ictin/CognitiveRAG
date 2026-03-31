from __future__ import annotations

import json
import sqlite3
import re
from pathlib import Path
from typing import Iterable, List

from CognitiveRAG.crag.skill_memory.schemas import SkillArtifact, artifact_to_record


class SkillMemoryStore:
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
                CREATE TABLE IF NOT EXISTS skill_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    title TEXT,
                    canonical_text TEXT NOT NULL,
                    normalized_key TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    tags_json TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    links_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skill_namespace ON skill_artifacts(namespace)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skill_type ON skill_artifacts(artifact_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skill_norm ON skill_artifacts(normalized_key)")

    def upsert(self, artifact: SkillArtifact) -> None:
        rec = artifact_to_record(artifact)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skill_artifacts(
                    artifact_id, artifact_type, namespace, title, canonical_text, normalized_key, confidence,
                    tags_json, source_refs_json, links_json, metadata_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    artifact_type=excluded.artifact_type,
                    namespace=excluded.namespace,
                    title=excluded.title,
                    canonical_text=excluded.canonical_text,
                    normalized_key=excluded.normalized_key,
                    confidence=excluded.confidence,
                    tags_json=excluded.tags_json,
                    source_refs_json=excluded.source_refs_json,
                    links_json=excluded.links_json,
                    metadata_json=excluded.metadata_json,
                    payload_json=excluded.payload_json
                """,
                (
                    rec["artifact_id"],
                    rec["artifact_type"],
                    rec["namespace"],
                    rec["title"],
                    rec["canonical_text"],
                    rec["normalized_key"],
                    rec["confidence"],
                    rec["tags_json"],
                    rec["source_refs_json"],
                    rec["links_json"],
                    rec["metadata_json"],
                    rec["payload_json"],
                ),
            )

    def upsert_many(self, artifacts: Iterable[SkillArtifact]) -> int:
        count = 0
        for artifact in artifacts:
            self.upsert(artifact)
            count += 1
        return count

    def list_namespace(self, namespace: str) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM skill_artifacts WHERE namespace = ? ORDER BY artifact_id",
                (namespace,),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def search(self, query: str, *, namespace: str | None = None, top_k: int = 5) -> List[dict]:
        tokens = [t for t in re.split(r"\W+", (query or "").strip().lower()) if t]
        if not tokens:
            return []
        params: list[object] = []
        per_token = "(LOWER(canonical_text) LIKE ? OR LOWER(title) LIKE ? OR LOWER(normalized_key) LIKE ?)"
        where = " AND ".join(per_token for _ in tokens)
        for token in tokens:
            q = f"%{token}%"
            params.extend([q, q, q])
        if namespace:
            where = f"namespace = ? AND ({where})"
            params = [namespace, *params]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload_json FROM skill_artifacts
                WHERE {where}
                ORDER BY confidence DESC, artifact_id
                LIMIT ?
                """,
                (*params, int(top_k)),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]
