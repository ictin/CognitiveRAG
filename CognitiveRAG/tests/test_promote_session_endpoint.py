from fastapi.testclient import TestClient
import os, json, types, sys
from pathlib import Path

def test_promote_session_endpoint(tmp_path):
    session_id = 'op_sess'
    data_dir = Path(os.getcwd()) / 'data' / 'session_memory'
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        {'chunk_index': 0, 'summary': 'Op summary one.'},
        {'chunk_index': 1, 'summary': 'Op summary two.'},
    ]
    sum_file = data_dir / f'summaries_{session_id}.json'
    with open(sum_file, 'w', encoding='utf-8') as f:
        json.dump(summaries, f)

    # Use TestClient against real app; ensure real Orchestrator not needed for this endpoint
    from CognitiveRAG.main_server import app
    client = TestClient(app)

    resp = client.post('/promote_session', json={'session_id': session_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body['promoted_count'] == 2
    assert len(body['promoted_pattern_ids']) == 2

    # calling again is idempotent
    resp2 = client.post('/promote_session', json={'session_id': session_id})
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2['promoted_count'] == 2
    assert body2['promoted_pattern_ids'] == body['promoted_pattern_ids']

    # cleanup
    sum_file.unlink()
