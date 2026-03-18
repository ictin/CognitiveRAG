import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient
from CognitiveRAG.app import app
from CognitiveRAG.core.settings import settings


def test_task_profile_reasoning_query_selectivity():
    # Prepare two tasks, only one matches
    tdb = str(settings.store.task_db_path)
    conn = sqlite3.connect(tdb)
    try:
        conn.execute("INSERT OR REPLACE INTO tasks(task_id,title,status,summary,metadata_json) VALUES(?,?,?,?,?)", ("task_keep","Keep this task","open","contains apple banana","{}"))
        conn.execute("INSERT OR REPLACE INTO tasks(task_id,title,status,summary,metadata_json) VALUES(?,?,?,?,?)", ("task_drop","Drop this task","open","unrelated content","{}"))
        conn.commit()
    finally:
        conn.close()

    pdb = str(settings.store.profile_db_path)
    conn = sqlite3.connect(pdb)
    try:
        conn.execute("INSERT OR REPLACE INTO profile_facts(key,value,source,confidence) VALUES(?,?,?,?)", ("pf_keep","apple banana","s",0.9))
        conn.execute("INSERT OR REPLACE INTO profile_facts(key,value,source,confidence) VALUES(?,?,?,?)", ("pf_drop","something else","s",0.9))
        conn.commit()
    finally:
        conn.close()

    rdb = str(settings.store.reasoning_db_path)
    conn = sqlite3.connect(rdb)
    try:
        conn.execute("INSERT OR REPLACE INTO reasoning_patterns(pattern_id,problem_signature,reasoning_steps_json,solution_summary,confidence) VALUES(?,?,?,?,?)", ("pattern_keep","apple","[]","solution about apple",0.9))
        conn.execute("INSERT OR REPLACE INTO reasoning_patterns(pattern_id,problem_signature,reasoning_steps_json,solution_summary,confidence) VALUES(?,?,?,?,?)", ("pattern_drop","banana2","[]","irrelevant",0.9))
        conn.commit()
    finally:
        conn.close()

    with TestClient(app) as client:
        r = client.post("/query", json={"query":"apple","retrieval_mode":"task_memory"})
        assert r.status_code == 200, r.text
        j = r.json()
        summary = j.get("trace", {}).get("retrieval_summary", [])
        assert summary, j
        # ensure matching ids present
        assert any(x in ("task_keep","pf_keep","pattern_keep") for x in summary), summary
        # ensure unrelated ids not dominating (at least one dropped id should not be the only thing)
        assert not all(x in ("task_drop","pf_drop","pattern_drop") for x in summary), summary
        # ensure no episodic ids
        assert all(not x.startswith("evt_") for x in summary), summary
