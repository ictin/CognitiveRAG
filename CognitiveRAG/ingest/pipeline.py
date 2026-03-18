from __future__ import annotations

import json
from pathlib import Path

from CognitiveRAG.core.settings import Settings
from CognitiveRAG.ingest.chunking import chunk_text
from CognitiveRAG.ingest.loaders import load_path


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        metadata_store,
        vector_store,
        lexical_store,
        graph_store,
    ):
        self.settings = settings
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.lexical_store = lexical_store
        self.graph_store = graph_store

    def ingest_path(self, path: Path) -> list[str]:
        loaded = load_path(path)
        self.metadata_store.upsert_document(
            document_id=loaded.document_id,
            source_path=loaded.source_path,
            content_hash=loaded.content_hash,
        )

        chunks = chunk_text(
            document_id=loaded.document_id,
            text=loaded.content,
            chunk_size=self.settings.retrieval.chunk_size,
            chunk_overlap=self.settings.retrieval.chunk_overlap,
            base_metadata=loaded.metadata,
        )

        # Build payloads and enrich metadata BEFORE persisting to metadata store and vector store.
        chunk_payloads = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": dict(chunk.metadata or {}),
            }
            for chunk in chunks
        ]

        # Enrich metadata with required fields for future policy-based filtering.
        from datetime import datetime
        for p in chunk_payloads:
            m = p.setdefault("metadata", {})
            m.setdefault("source_type", "document")
            m.setdefault("project", loaded.metadata.get("project", "cognitiverag"))
            tr = loaded.metadata.get("test_run_id")
            m.setdefault("test_run_id", tr if tr is not None else "")
            m.setdefault("document_kind", m.get("document_kind", "chunk"))
            m.setdefault("content_hash", loaded.content_hash)
            now = datetime.utcnow().isoformat() + "Z"
            m.setdefault("created_at", now)
            m["updated_at"] = now

        # Persist enriched metadata to metadata store (SQLite)
        self.metadata_store.replace_chunks(loaded.document_id, [
            (p["chunk_id"], loaded.document_id, p.get("metadata", {}).get("chunk_index", 0), p["text"], json.dumps(p.get("metadata", {})))
            for p in chunk_payloads
        ])

        # Upsert to vector and lexical stores with enriched metadata
        self.vector_store.upsert_chunks(chunk_payloads)
        self.lexical_store.upsert_chunks(chunk_payloads)

        # TODO: entity extraction -> graph store
        self.graph_store.upsert_extractions(loaded.document_id, extractions=[])

        return [loaded.document_id]
