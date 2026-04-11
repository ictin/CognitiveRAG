from __future__ import annotations

from pathlib import Path
from typing import Any, List
import json

from CognitiveRAG.schemas.retrieval import RetrievedChunk

import chromadb
from chromadb.utils import embedding_functions
import logging


class VectorStore:
    """Persistent on-disk Chroma-backed vector store.

    - Uses chromadb.Client with persist_directory pointing to the provided path
    - Collection name: "cognitive_chunks"
    - upsert_chunks performs upsert by chunk_id (stable IDs)
    """

    COLLECTION_NAME = "cognitive_chunks"

    @staticmethod
    def _sanitize_metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
                continue
            if isinstance(value, list):
                # Keep only primitive-compatible list items for Chroma metadata.
                if all(isinstance(v, (str, int, float, bool)) or v is None for v in value):
                    sanitized[key] = value
                else:
                    sanitized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                continue
            # Nested dict/object values are serialized to preserve provenance without failing upsert.
            sanitized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return sanitized

    def __init__(self, path: Path, embedding_model: str | None = None, backing_impl: str | None = None):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.backing_impl = backing_impl or "duckdb+parquet"
        # Prefer modern PersistentClient. Avoid deprecated legacy settings fallback.
        try:
            self._client = chromadb.PersistentClient(path=str(self.path))
            logging.info('VectorStore: using chromadb.PersistentClient with path=%s', str(self.path))
            try:
                self._collection = self._client.get_collection(self.COLLECTION_NAME)
                logging.info('VectorStore: existing collection found; not changing embedding function')
            except Exception:
                logging.info('VectorStore: collection missing; will create with embedding function')
                embedding_fn = self._build_embedding_function()
                self._collection = self._client.create_collection(name=self.COLLECTION_NAME, embedding_function=embedding_fn)
        except Exception as exc:
            logging.warning("VectorStore: PersistentClient init failed; falling back to in-process client: %s", exc)
            self._client = chromadb.Client()
            try:
                self._collection = self._client.get_collection(self.COLLECTION_NAME)
            except Exception:
                self._collection = self._client.create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self._build_embedding_function(),
                )

    def _build_embedding_function(self):
        if not self.embedding_model:
            return None
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        except Exception as exc:
            logging.warning(
                "VectorStore: sentence-transformers embedding unavailable for '%s'; using default embedding: %s",
                self.embedding_model,
                exc,
            )
            try:
                return embedding_functions.DefaultEmbeddingFunction()
            except Exception as default_exc:
                logging.warning("VectorStore: default embedding init failed; continuing without explicit embedding fn: %s", default_exc)
                return None

    def upsert_chunks(self, chunks: List[dict[str, Any]]) -> None:
        # Each chunk dict: {chunk_id, document_id, text, metadata}
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [self._sanitize_metadata_for_chroma(c.get("metadata", {})) for c in chunks]
        # Store document-level metadata plus document_id for lookup
        for m, c in zip(metadatas, chunks):
            m.setdefault("document_id", c.get("document_id"))
        # Upsert into Chroma collection (will update existing ids)
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        # persist to disk
        try:
            self._client.persist()
        except Exception:
            # older chroma clients may not have persist method; ignore
            pass

    def query(self, query: str, top_k: int = 5, where: dict | None = None) -> List[RetrievedChunk]:
        # Use the collection to query by text, optionally filtering by metadata (where)
        try:
            if where:
                results = self._collection.query(query_texts=[query], n_results=top_k, include=['metadatas','distances','documents'], where=where)
            else:
                results = self._collection.query(query_texts=[query], n_results=top_k, include=['metadatas','distances','documents'])
        except TypeError:
            # fallback for older chroma versions that may not support where
            try:
                results = self._collection.query(query_texts=[query], n_results=top_k, include=['metadatas','distances','documents'])
            except Exception:
                results = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        chunks: List[RetrievedChunk] = []
        if not results or len(results.get('ids', [])) == 0:
            # some chroma returns nested lists
            ids = results.get('ids', [[]])[0]
            docs = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0] if 'distances' in results else [0.0]*len(ids)
        else:
            ids = results['ids'][0]
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results.get('distances',[[]])[0] if 'distances' in results else [0.0]*len(ids)

        for cid, doc, meta, dist in zip(ids, docs, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=meta.get('document_id'),
                    text=doc,
                    source_type='vector',
                    score=float(1.0 - dist) if isinstance(dist, (int, float)) else 0.0,
                    metadata=meta,
                )
            )
        return chunks

    def delete_document(self, document_id: str) -> None:
        # delete all chunk ids that have this document_id in metadata
        # Chroma doesn't provide direct delete by metadata; iterate collection metadata
        query = f"document_id:{document_id}"
        # fetch all ids and filter
        all_data = self._collection.get(include=['metadatas','documents'])
        metadatas = all_data.get('metadatas', [])
        # some chroma returns nested lists
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        # collection.get doesn't always return ids; instead infer ids from metadata if present
        ids = [m.get('id') or m.get('chunk_id') for m in metadatas]
        # fallback: use metadatas keys
        to_delete = [i for i,m in zip(ids, metadatas) if m.get('document_id')==document_id and i]
        if to_delete:
            self._collection.delete(ids=to_delete)
            try:
                self._client.persist()
            except Exception:
                pass
