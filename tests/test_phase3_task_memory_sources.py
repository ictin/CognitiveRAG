from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from CognitiveRAG.app import app
from CognitiveRAG.core.settings import settings


def test_task_profile_reasoning_included_in_task_memory():
    # insert one task, one profile fact, one reasoning pattern directly into DBs
    tdb = str(settings.store.task_db_path)
    conn = sqlite3.connect(tdb)
    try:
        conn.execute("INSERT OR REPLACE INTO tasks(task_id,title,status,summary,metadata_json) VALUES(?,?,?,?,?)", ("task_test_1","t","open","summary","{}"))
        conn.commit()
    finally:
        conn.close()

    pdb = str(settings.store.profile_db_path)
    conn = sqlite3.connect(pdb)
    try:
        conn.execute("INSERT OR REPLACE INTO profile_facts(key,value,source,confidence) VALUES(?,?,?,?)", ("pf_test","v","s",0.9))
        conn.commit()
    finally:
        conn.close()

    rdb = str(settings.store.reasoning_db_path)
    conn = sqlite3.connect(rdb)
    try:
        conn.execute("INSERT OR REPLACE INTO reasoning_patterns(pattern_id,problem_signature,reasoning_steps_json,solution_summary,confidence) VALUES(?,?,?,?,?)", ("pattern_test","sig","[]","sol",0.9))
        conn.commit()
    finally:
        conn.close()

    with TestClient(app) as client:
        r = client.post("/query", json={"query":"What is this document about?","retrieval_mode":"task_memory"})
        assert r.status_code == 200, r.text
        j = r.json()
        summary = j.get("trace", {}).get("retrieval_summary", [])
        assert summary, j
        # ensure no episodic ids
        assert all(not x.startswith("evt_") for x in summary), summary
        # ensure at least one of our inserted ids is present
        assert any(x in ("task_test_1","pf_test","pattern_test") for x in summary), summary
