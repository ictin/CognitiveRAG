from __future__ import annotations

from CognitiveRAG.schemas.retrieval import RetrievedChunk


def rerank_chunks(chunks: list[RetrievedChunk], max_items: int) -> list[RetrievedChunk]:
    deduped: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        existing = deduped.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            deduped[chunk.chunk_id] = chunk

    ranked = sorted(deduped.values(), key=lambda c: c.score, reverse=True)
    return ranked[:max_items]
