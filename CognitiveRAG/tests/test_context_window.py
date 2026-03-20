import os
import json
import shutil
from fastapi.testclient import TestClient
from CognitiveRAG.session_memory.context_window import compact_session, assemble_context, WORKDIR


def make_raw(session_id, count=50):
    raw = []
    for i in range(count):
        raw.append({"index": i, "text": f"message {i}", "meta": {}})
    path = os.path.join(WORKDIR, f"raw_{session_id}.json")
    os.makedirs(WORKDIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    return raw


def test_compaction_creates_summaries_and_preserves_raw(tmp_path):
    session_id = "sess1"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    raw = make_raw(session_id, count=50)

    created = compact_session(session_id, older_than_index=30)
    path = os.path.join(WORKDIR, f"raw_{session_id}.json")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == raw

    sum_path = os.path.join(WORKDIR, f"summaries_{session_id}.json")
    assert os.path.exists(sum_path)
    with open(sum_path, "r", encoding="utf-8") as f:
        sums = json.load(f)
    assert len(sums) > 0
    assert all("source_count" in s and "chunk_index" in s for s in sums)


def test_assemble_context_returns_fresh_tail_and_summaries(tmp_path):
    session_id = "sess2"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    make_raw(session_id, count=40)
    ctx = assemble_context(session_id, fresh_tail_count=5, budget=1000)
    assert "fresh_tail" in ctx and "summaries" in ctx
    assert len(ctx["fresh_tail"]) == 5

    compact_session(session_id, older_than_index=20)
    ctx2 = assemble_context(session_id, fresh_tail_count=5, budget=1000)
    assert len(ctx2["fresh_tail"]) == 5
    assert len(ctx2["summaries"]) >= 1


def test_session_assemble_context_endpoint(tmp_path):
    session_id = "sess3"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    make_raw(session_id, count=30)
    compact_session(session_id, older_than_index=20)

    from CognitiveRAG.main_server import app
    client = TestClient(app)

    resp = client.post('/session_assemble_context', json={
        'session_id': session_id,
        'fresh_tail_count': 4,
        'budget': 1000,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert 'fresh_tail' in body and 'summaries' in body
    assert len(body['fresh_tail']) == 4
    assert len(body['summaries']) >= 1
