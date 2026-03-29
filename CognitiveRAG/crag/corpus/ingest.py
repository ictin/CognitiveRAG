from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from CognitiveRAG.crag.corpus.chunkers import chunk_document


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_chunk_payloads(
    *,
    document_id: str,
    source_path: str,
    content: str,
    content_hash: str,
    chunk_size: int,
    chunk_overlap: int,
    base_metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    chunks = chunk_document(
        document_id=document_id,
        source_path=source_path,
        text=content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        base_metadata=base_metadata or {},
    )

    now = _utcnow_iso()
    payloads: List[Dict[str, Any]] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        metadata.setdefault("source_type", "document")
        metadata.setdefault("project", (base_metadata or {}).get("project", "cognitiverag"))
        metadata.setdefault("test_run_id", (base_metadata or {}).get("test_run_id", ""))
        metadata.setdefault("content_hash", content_hash)
        metadata.setdefault("created_at", now)
        metadata["updated_at"] = now
        payloads.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": metadata,
            }
        )
    return payloads
