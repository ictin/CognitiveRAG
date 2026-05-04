from __future__ import annotations

import json
import os
import shutil

from CognitiveRAG.session_memory import context_window


def _seed_discovery_fixture(session_id: str) -> None:
    os.makedirs(context_window.WORKDIR, exist_ok=True)
    rows = [
        {"index": 0, "text": "Feature flag is enabled in production for all users.", "meta": {"lane_hint": "semantic"}},
        {"index": 1, "text": "Feature flag is not enabled in production for all users.", "meta": {"lane_hint": "semantic"}},
        {"index": 2, "text": "Rollback rehearsal was skipped in the last rollout drill.", "meta": {"lane_hint": "episodic"}},
        {"index": 3, "text": "Missing rollback rehearsal increases latent incident risk.", "meta": {"lane_hint": "corpus"}},
        {"index": 4, "text": "Deployment looked successful in initial smoke checks.", "meta": {"lane_hint": "episodic"}},
        {"index": 5, "text": "Audit note: safety gate depends on contradiction visibility.", "meta": {"lane_hint": "corpus"}},
        {"index": 6, "text": "Non-obvious risk: stale fallback assumptions can hide unresolved blockers.", "meta": {"lane_hint": "semantic"}},
        {"index": 7, "text": "Operator requested bounded discovery with explicit provenance.", "meta": {"lane_hint": "task"}},
        {"index": 8, "text": "A previous run skipped contradiction checks due haste.", "meta": {"lane_hint": "episodic"}},
        {"index": 9, "text": "A current run requires contradiction checks before rollout approval.", "meta": {"lane_hint": "episodic"}},
        {"index": 10, "text": "Unknown unknown probe: verify hidden dependencies before cutover.", "meta": {"lane_hint": "corpus"}},
    ]
    with open(os.path.join(context_window.WORKDIR, f"raw_{session_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def test_tc004_tc006_discovery_slice_is_bounded_contradiction_aware_and_truthful(tmp_path):
    session_id = "tc004-discovery-proof-slice"
    query = "What else matters here that I may not be asking?"
    context_window.WORKDIR = str(tmp_path / "session_memory")
    shutil.rmtree(context_window.WORKDIR, ignore_errors=True)
    os.makedirs(context_window.WORKDIR, exist_ok=True)
    _seed_discovery_fixture(session_id)

    out = context_window.assemble_context(
        session_id=session_id,
        fresh_tail_count=12,
        budget=1400,
        query=query,
    )

    plan = out["discovery_plan"]
    discovery = out["discovery"]
    explanation = out["explanation"]
    metrics = out["selector_metrics"]

    assert plan["intent_family"] == "investigative"
    assert discovery["bounded"] is True
    assert int(discovery["used_tokens"]) <= int(discovery["budget_tokens"])
    assert len(discovery["injected_discoveries"]) >= 1
    assert metrics["discovery"]["injected_count"] == len(discovery["injected_discoveries"])

    contradiction_count = len(discovery.get("contradictions") or [])
    ledger = discovery.get("ledger") or {}
    rejected = list(ledger.get("rejected_branches") or [])
    unresolved = list(ledger.get("unresolved_questions") or [])
    weak_signal_count = len(rejected) + len(unresolved)
    assert contradiction_count >= 1 or weak_signal_count >= 1

    injected_items = [str(item.get("text", "")).strip().lower() for item in discovery["injected_discoveries"]]
    assert any(item and item != query.lower().strip() for item in injected_items)

    selected_discovery_blocks = [b for b in explanation["selected_blocks"] if b.get("lane") == "discovery"]
    assert selected_discovery_blocks, "Discovery contribution must be visible in explanation selected blocks."
    for block in selected_discovery_blocks:
        prov = block.get("provenance") or {}
        assert prov.get("branch_id"), "Discovery provenance must include branch_id."
        assert prov.get("lane"), "Discovery provenance must include source lane."
        assert prov.get("memory_type"), "Discovery provenance must include source memory_type."
