from __future__ import annotations

from collections import defaultdict

from CognitiveRAG.schemas.retrieval import RetrievedChunk


class LexicalStore:
    def __init__(self):
        self._chunks: dict[str, dict] = {}
        self._by_document: dict[str, list[str]] = defaultdict(list)

    def upsert_chunks(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            self._chunks[chunk_id] = chunk
            self._by_document[chunk["document_id"]].append(chunk_id)

    def delete_document(self, document_id: str) -> None:
        for chunk_id in self._by_document.get(document_id, []):
            self._chunks.pop(chunk_id, None)
        self._by_document.pop(document_id, None)

    def query(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_terms = {term.lower() for term in query.split()}
        scored: list[tuple[int, dict]] = []
        for chunk in self._chunks.values():
            text_terms = set(chunk["text"].lower().split())
            score = len(query_terms.intersection(text_terms))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=item[1]["chunk_id"],
                document_id=item[1]["document_id"],
                text=item[1]["text"],
                source_type="lexical",
                score=float(item[0]),
                metadata=item[1].get("metadata", {}),
            )
            for item in scored[:top_k]
        ]
