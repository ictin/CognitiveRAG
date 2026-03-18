from __future__ import annotations

import sqlite3
from pathlib import Path

from CognitiveRAG.schemas.retrieval import RetrievedChunk


class GraphStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS relations (
                    relation_id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL
                );
                """
            )

    def upsert_extractions(self, document_id: str, extractions: list[dict]) -> None:
        # TODO: normalize extracted entities and relations into graph tables
        pass

    def query(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # TODO: entity lookup + neighbor expansion + supporting chunk lookup
        return []
