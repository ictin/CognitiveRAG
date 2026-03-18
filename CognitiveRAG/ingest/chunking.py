from __future__ import annotations

from dataclasses import dataclass


CHUNK_ID_TEMPLATE = "{document_id}_chunk_{index:04d}"


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: dict


def make_chunk_id(document_id: str, index: int) -> str:
    """Generate a stable chunk id for a document and chunk index."""
    return CHUNK_ID_TEMPLATE.format(document_id=document_id, index=index)


def chunk_text(document_id: str, text: str, chunk_size: int, chunk_overlap: int, base_metadata: dict) -> list[Chunk]:
    """Split text into deterministic chunks and return Chunk objects.

    Chunk IDs follow the format: {document_id}_chunk_{index:04d}
    Example: doc_52cabc84f90d7c70_chunk_0000
    """
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_value = text[start:end]
        chunk_id = make_chunk_id(document_id, index)
        metadata = dict(base_metadata)
        metadata["chunk_index"] = index
        metadata["char_start"] = start
        metadata["char_end"] = end

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=index,
                text=chunk_text_value,
                metadata=metadata,
            )
        )

        if end == len(text):
            break
        start = max(0, end - chunk_overlap)
        index += 1

    return chunks
