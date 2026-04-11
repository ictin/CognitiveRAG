import json
import os
import shutil
import sqlite3

from CognitiveRAG.session_memory.context_window import WORKDIR, assemble_context


def _seed_reasoning(workdir: str):
    db = sqlite3.connect(os.path.join(workdir, "reasoning.sqlite3"))
    db.execute(
        "CREATE TABLE IF NOT EXISTS reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, solution_summary TEXT, confidence REAL, provenance_json TEXT)"
    )
    db.execute(
        "INSERT OR REPLACE INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?)",
        ("p1", "about me", "[]", "User works with .NET and Postgres.", 0.9, "[]"),
    )
    db.commit()
    db.close()


def _seed_corpus(workdir: str):
    db = sqlite3.connect(os.path.join(workdir, "context_items.sqlite3"))
    db.execute("CREATE TABLE IF NOT EXISTS context_items (item_id TEXT, session_id TEXT, type TEXT, payload_json TEXT, created_at TEXT)")
    payload = {"summary": "YouTube Secrets: synopsis emphasizes retention systems.", "file_path": "books/youtube_secrets.md"}
    db.execute("INSERT INTO context_items VALUES (?, ?, ?, ?, ?)", ("c1", "sess", "corpus_chunk", json.dumps(payload), "2026-01-01"))
    db.commit()
    db.close()


def _seed_session(session_id: str):
    os.makedirs(WORKDIR, exist_ok=True)
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump([
            {"index": 0, "text": "we discussed token ABC", "message_id": "m0"},
            {"index": 1, "text": "another detail", "message_id": "m1"},
            {"index": 2, "text": "recent tail message", "message_id": "m2"},
        ], f)
    with open(os.path.join(WORKDIR, f"summaries_{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump([{"chunk_index": 0, "summary": "summary block"}], f)


def test_router_to_selector_exact_recall_prefers_episodic():
    session_id = "int1"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    _seed_session(session_id)

    out = assemble_context(session_id, query="what did we say earlier about token ABC", budget=500, fresh_tail_count=1)
    assert out["retrieval_route"]["intent_family"] == "exact_recall"
    assert out["retrieval_route"]["lanes"][0] == "episodic"
    lanes = [b["lane"] for b in out["selected_blocks"]]
    assert "episodic" in lanes


def test_router_to_selector_memory_summary_prefers_promoted(tmp_path):
    session_id = "int2"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    _seed_session(session_id)
    _seed_reasoning(WORKDIR)

    out = assemble_context(session_id, query="what do you know about me", budget=600, fresh_tail_count=1)
    assert out["retrieval_route"]["intent_family"] == "memory_summary"
    assert "promoted" in out["retrieval_route"]["lanes"]
    lanes = [b["lane"] for b in out["selected_blocks"]]
    assert "promoted" in lanes


def test_router_to_selector_corpus_prefers_corpus_large_file():
    session_id = "int3"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    _seed_session(session_id)
    _seed_corpus(WORKDIR)

    out = assemble_context(session_id, query="what does this book say about retention", budget=700, fresh_tail_count=1)
    assert out["retrieval_route"]["intent_family"] == "corpus_overview"
    assert out["retrieval_route"]["lanes"][0] == "corpus"
    lanes = [b["lane"] for b in out["selected_blocks"]]
    assert "corpus" in lanes


def test_router_to_selector_mixed_investigative_produces_mixed_pool():
    session_id = "int4"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    _seed_session(session_id)
    _seed_reasoning(WORKDIR)
    _seed_corpus(WORKDIR)

    out = assemble_context(session_id, query="investigate what we said and what the book says", budget=900, fresh_tail_count=1)
    assert out["retrieval_route"]["intent_family"] == "investigative"
    lanes = {b["lane"] for b in out["selected_blocks"]}
    assert "episodic" in lanes
    assert ("corpus" in lanes) or ("large_file" in lanes)


def test_exact_recall_prefers_exact_span_and_avoids_summary_noise():
    session_id = "int5"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {"index": 0, "text": "generic mirror summary boilerplate", "message_id": "m0"},
                {
                    "index": 1,
                    "text": "Exact quote anchor: checksum 9247A and deploy window Friday 10:00 UTC.",
                    "message_id": "m1",
                },
                {"index": 2, "text": "another generic mirror-ish line", "message_id": "m2"},
                {"index": 3, "text": "recent filler line", "message_id": "m3"},
            ],
            f,
        )
    with open(os.path.join(WORKDIR, f"summaries_{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump([{"chunk_index": 0, "summary": "summary generic mirror text"}], f)

    out = assemble_context(
        session_id,
        query="quote the exact earlier span with checksum 9247A",
        budget=420,
        fresh_tail_count=1,
    )
    selected = out.get("selected_blocks", [])
    selected_ids = [b.get("id") for b in selected]
    selected_texts = [str(b.get("text") or "") for b in selected]

    assert out["retrieval_route"]["intent_family"] == "exact_recall"
    assert "episodic:int5:m1" in selected_ids
    assert any("checksum 9247A" in text for text in selected_texts)
    assert not any((b.get("lane") == "session_summary") for b in selected)


def test_task_state_query_prefers_session_grounded_evidence_over_stale_promoted():
    session_id = "int6"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {"index": 0, "text": "Task kickoff: stabilize phase3 runtime proof.", "message_id": "m0"},
                {
                    "index": 1,
                    "text": "Current blocker: gateway timeout on heavy benchmark and mirror-heavy task-state answer.",
                    "message_id": "m1",
                },
                {
                    "index": 2,
                    "text": "Next step: prioritize session-grounded task-state retrieval and rerun canary.",
                    "message_id": "m2",
                },
                {
                    "index": 3,
                    "text": "Acceptance criteria: task-state answer must cite blocker and next step.",
                    "message_id": "m3",
                },
            ],
            f,
        )
    with open(os.path.join(WORKDIR, f"summaries_{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump([{"chunk_index": 0, "summary": "generic mirror summary old blocker legacy query path"}], f)

    import sqlite3

    with sqlite3.connect(os.path.join(WORKDIR, "reasoning.sqlite3")) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns ("
            "pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, "
            "freshness_state TEXT, reuse_count INTEGER, canonical_pattern_id TEXT, near_duplicate_of TEXT, "
            "success_signal_count INTEGER, failure_signal_count INTEGER, success_confidence REAL, success_basis_json TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO reasoning_patterns("
            "pattern_id, problem_signature, reasoning_steps_json, solution_summary, confidence, provenance_json, "
            "memory_subtype, normalized_text, freshness_state, reuse_count, success_signal_count, failure_signal_count, "
            "success_confidence, success_basis_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "p_stale",
                "legacy-path",
                "[]",
                "Current blocker: legacy /query path. Next step: generic fix.",
                0.9,
                "[]",
                "workflow",
                "current blocker legacy query path",
                "fresh",
                2,
                2,
                0,
                0.8,
                "{}",
            ),
        )
        conn.commit()

    out = assemble_context(
        session_id,
        query="What is the current task state in this session, what changed, and what are blockers?",
        budget=420,
        fresh_tail_count=2,
    )
    selected = out.get("selected_blocks", [])
    selected_texts = [str(b.get("text") or "") for b in selected]

    assert out["retrieval_route"]["intent_family"] == "planning"
    assert out["discovery_plan"]["discovery_mode"] == "off"
    assert out["selector_metrics"]["discovery"]["injected_count"] == 0
    assert any("gateway timeout" in text for text in selected_texts)
    assert any("prioritize session-grounded task-state retrieval" in text for text in selected_texts)
    assert not any((b.get("lane") == "session_summary") for b in selected)
    assert out.get("summaries") == []
