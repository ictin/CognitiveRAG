from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import chromadb
from fastapi.testclient import TestClient
from CognitiveRAG.app import app
from CognitiveRAG.core.settings import settings

def _post_query(client: TestClient, retrieval_mode: str) -> dict:
    r = client.post(
        "/query",
        json={"query": "What is this document about?", "retrieval_mode": retrieval_mode},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_documents_only_returns_only_doc_ids():
    with TestClient(app) as client:
        resp = _post_query(client, "documents_only")
        summary = resp.get("trace", {}).get("retrieval_summary", [])
        assert summary, resp
        assert all(x.startswith("doc_") for x in summary), summary


def test_regression_test_returns_only_doc_ids():
    with TestClient(app) as client:
        resp = _post_query(client, "regression_test")
        summary = resp.get("trace", {}).get("retrieval_summary", [])
        assert summary, resp
        assert all(x.startswith("doc_") for x in summary), summary


def test_task_memory_returns_only_non_episodic_ids():
    with TestClient(app) as client:
        resp = _post_query(client, "task_memory")
        summary = resp.get("trace", {}).get("retrieval_summary", [])
        assert summary, resp
        assert all(not x.startswith("evt_") for x in summary), summary


def test_full_memory_returns_200():
    with TestClient(app) as client:
        resp = _post_query(client, "full_memory")
        summary = resp.get("trace", {}).get("retrieval_summary", [])
        assert isinstance(summary, list), resp
        assert len(summary) > 0, resp


def test_newest_episodic_row_has_metadata_contract():
    with TestClient(app) as client:
        _post_query(client, "documents_only")
    db = str(settings.store.episodic_db_path)
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            """
            SELECT * FROM events ORDER BY rowid DESC LIMIT 1
            """
        ).fetchone()
        assert row is not None
        metadata_json = row[-1]
    finally:
        conn.close()
    meta = metadata_json
    if isinstance(meta, str):
        meta = meta.replace("'", '"')
        data = json.loads(meta)
    else:
        data = dict(meta)
    assert data.get("source_type") == "episodic", data
    assert data.get("project"), data
    assert data.get("origin_id"), data
    assert data.get("created_at"), data
    assert data.get("updated_at"), data


def test_newest_document_chunk_has_metadata_contract():
    marker = "phase3-metadata-contract"
    p = Path("data/source_documents/phase3_metadata_contract.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"Metadata contract test. Marker: {marker}\n", encoding="utf-8")
    with TestClient(app) as client:
        r = client.post("/ingest", json={"path": str(p.resolve()), "recursive": False})
        assert r.status_code == 200, r.text
    db = str(settings.store.metadata_db_path)
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            """
            SELECT chunk_id, metadata_json FROM chunks ORDER BY rowid DESC LIMIT 1
            """
        ).fetchone()
        assert row is not None
        chunk_id, metadata_json = row
    finally:
        conn.close()
    meta = json.loads(metadata_json)
    assert meta.get("source_type") == "document", meta
    assert meta.get("project"), meta
    assert meta.get("document_kind"), meta
    assert "test_run_id" in meta, meta
    assert meta.get("content_hash"), meta
    assert meta.get("created_at"), meta
    assert meta.get("updated_at"), meta
    client = chromadb.PersistentClient(path=str(settings.store.vector_store_path))
    collection = client.get_collection("cognitive_chunks")
    res = collection.get(ids=[chunk_id], include=["metadatas", "documents"]) 
    metas = res.get("metadatas") or []
    assert metas, res
    meta2 = metas[0]
    assert meta2.get("source_type") == "document", meta2
    assert meta2.get("project"), meta2
    assert meta2.get("document_kind"), meta2
    assert "test_run_id" in meta2, meta2
    assert meta2.get("content_hash"), meta2
    assert meta2.get("created_at"), meta2
    assert meta2.get("updated_at"), meta2
