#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from CognitiveRAG.session_memory.context_window import WORKDIR, compact_session
from CognitiveRAG.session_memory.compaction import SessionCompactionStore, summarize_compaction_state


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f016_compaction_trigger_slice"
    outdir.mkdir(parents=True, exist_ok=True)

    session_id = f"f016-{stamp.lower()}"
    os.makedirs(WORKDIR, exist_ok=True)
    raw_path = Path(WORKDIR) / f"raw_{session_id}.json"
    summaries_path = Path(WORKDIR) / f"summaries_{session_id}.json"
    compaction_db = Path(WORKDIR) / "compaction.sqlite3"

    rows = []
    for i in range(24):
        rows.append(
            {
                "message_id": f"m{i}",
                "index": i,
                "sender": "assistant" if i % 2 else "user",
                "text": f"F016 compaction payload message {i} with exact anchor",
                "created_at": f"2026-05-04T10:{i:02d}:00Z",
            }
        )
    raw_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if summaries_path.exists():
        summaries_path.unlink()

    before = {
        "raw_message_count": len(rows),
        "raw_hash": _hash_file(raw_path),
        "summary_exists": summaries_path.exists(),
        "summary_count": 0,
    }
    (outdir / "before_memory_state.json").write_text(json.dumps(before, indent=2), encoding="utf-8")

    result = compact_session(
        session_id=session_id,
        older_than_index=20,
        trigger_config={"min_compactable_count": 5, "chunk_size": 10},
        return_meta=True,
    )
    created = list(result.get("created") or [])
    trigger = dict(result.get("trigger") or {})
    (outdir / "compaction_trigger_artifact.json").write_text(
        json.dumps({"session_id": session_id, "trigger": trigger, "created_count": len(created)}, indent=2),
        encoding="utf-8",
    )

    summaries = []
    if summaries_path.exists():
        summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    store = SessionCompactionStore(str(compaction_db))
    state = summarize_compaction_state(session_id, store)

    after = {
        "raw_message_count": len(json.loads(raw_path.read_text(encoding="utf-8"))),
        "raw_hash": _hash_file(raw_path),
        "summary_exists": summaries_path.exists(),
        "summary_count": len(summaries),
        "compaction_state": state,
    }
    (outdir / "after_memory_state.json").write_text(json.dumps(after, indent=2), encoding="utf-8")

    proof = {
        "raw_preserved_hash_equal": before["raw_hash"] == after["raw_hash"],
        "raw_preserved_count_equal": before["raw_message_count"] == after["raw_message_count"],
        "derived_summary_added": after["summary_count"] >= len(created) and after["summary_count"] >= 1,
        "segment_lineage_present": any(len(list(seg.get("lineage") or [])) >= 1 for seg in store.list_segments(session_id)),
        "segment_snapshot_present": any(len(list(seg.get("raw_snapshot") or [])) >= 1 for seg in store.list_segments(session_id)),
    }
    (outdir / "raw_preservation_proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")

    summary = {
        "schemaVersion": "f016_compaction_trigger_slice.v1",
        "artifactDir": str(outdir),
        "checks": {
            "trigger_reason_visible": bool(trigger.get("trigger_reason")),
            "trigger_threshold_visible": "min_compactable_count" in trigger and "chunk_size" in trigger,
            "trigger_fired_for_threshold_met": bool(trigger.get("trigger_fired")),
            "raw_preserved": bool(proof["raw_preserved_hash_equal"] and proof["raw_preserved_count_equal"]),
            "compaction_additive": bool(proof["derived_summary_added"] and proof["segment_lineage_present"] and proof["segment_snapshot_present"]),
        },
        "traceability": {
            "features": ["F-016", "F-002"],
            "requirements": ["REQ-002", "REQ-003"],
            "invariants": ["INV-002", "INV-003"],
            "workflow": ["WF-002"],
            "scenario": ["TC-002"],
        },
    }
    summary["passed"] = all(summary["checks"].values())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
