#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.context_selection.explanation import build_explanation
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType
from CognitiveRAG.crag.retrieval.promoted_lane import retrieve as retrieve_promoted
from CognitiveRAG.crag.retrieval.web_lane import retrieve as retrieve_web
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _seed_reasoning_db(workdir: Path) -> None:
    db_path = workdir / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns ("
            "pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT, "
            "reuse_count INTEGER, canonical_pattern_id TEXT, near_duplicate_of TEXT, success_signal_count INTEGER, failure_signal_count INTEGER, success_confidence REAL, success_basis_json TEXT)"
        )
        db.execute(
            "INSERT OR REPLACE INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "rp:f015-proof",
                "lifecycle readback proof",
                "[]",
                "Use explicit lifecycle metadata for promoted/reused facts.",
                0.82,
                json.dumps([{"source": "f015-proof"}]),
                "workflow_pattern",
                "lifecycle metadata proof",
                "current",
                2,
                None,
                None,
                1,
                0,
                0.7,
                json.dumps({"basis": "seed"}),
            ),
        )
        db.commit()


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f015_lifecycle_slice"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    store = WebPromotedMemoryStore(workdir / "web_promoted_memory.sqlite3")
    store.stage_fact(
        promoted_id="wp_unreviewed",
        canonical_fact="Unreviewed staged claim.",
        evidence_ids=["ev1"],
        confidence=0.62,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/unreviewed"},
        now_iso="2026-05-01T10:00:00Z",
    )
    store.upsert_fact(
        promoted_id="wp_approved",
        canonical_fact="Approved and fresh claim.",
        evidence_ids=["ev2", "ev3"],
        confidence=0.87,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/approved", "freshness_ttl_hours": 100},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_FRESH,
        now_iso="2026-05-03T10:00:00Z",
    )
    store.upsert_fact(
        promoted_id="wp_stale",
        canonical_fact="Old stale claim.",
        evidence_ids=["ev4", "ev5"],
        confidence=0.86,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/stale", "freshness_ttl_hours": 1},
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        now_iso="2026-05-01T10:00:00Z",
    )
    store.evaluate_freshness("wp_stale", now_iso="2026-05-04T10:00:00Z")
    store.request_revalidation("wp_stale", reason="ttl_expired_requires_revalidation", now_iso="2026-05-04T10:01:00Z")

    _seed_reasoning_db(workdir)
    web_hits = retrieve_web(workdir=str(workdir), query="claim", intent_family=IntentFamily.INVESTIGATIVE, top_k=6)
    promoted_hits = retrieve_promoted(
        workdir=str(workdir),
        query="lifecycle metadata",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=3,
    )
    web_promoted = [h for h in web_hits if h.memory_type == MemoryType.WEB_PROMOTED_FACT]
    by_id = {str((h.provenance or {}).get("promoted_id") or ""): h for h in web_promoted}

    lifecycle_artifact = {
        "schemaVersion": "f015_lifecycle_slice.v1",
        "artifactDir": str(outdir),
        "webPromotedLifecycle": {
            "wp_unreviewed": (by_id.get("wp_unreviewed").provenance if by_id.get("wp_unreviewed") else {}),
            "wp_approved": (by_id.get("wp_approved").provenance if by_id.get("wp_approved") else {}),
            "wp_stale": (by_id.get("wp_stale").provenance if by_id.get("wp_stale") else {}),
        },
        "reasoningPromotedLifecycle": (promoted_hits[0].provenance if promoted_hits else {}),
    }
    (outdir / "lifecycle_readback_artifact.json").write_text(json.dumps(lifecycle_artifact, indent=2), encoding="utf-8")

    # Build explicit explanation artifact proving lifecycle visibility/truthfulness in explanation surfaces.
    explanation_candidates: list[ContextCandidate] = []
    for hit in web_promoted[:2]:
        explanation_candidates.append(
            ContextCandidate(
                id=hit.id,
                lane=hit.lane,
                memory_type=hit.memory_type,
                text=hit.text,
                tokens=hit.tokens,
                provenance=dict(hit.provenance or {}),
                lexical_score=float(hit.lexical_score or 0.0),
                semantic_score=float(hit.semantic_score or 0.0),
                recency_score=float(hit.recency_score or 0.0),
                freshness_score=float(hit.freshness_score or 0.0),
                trust_score=float(hit.trust_score or 0.0),
                novelty_score=float(hit.novelty_score or 0.0),
                contradiction_risk=float(hit.contradiction_risk or 0.0),
                cluster_id=hit.cluster_id,
            )
        )
    if promoted_hits:
        top = promoted_hits[0]
        explanation_candidates.append(
            ContextCandidate(
                id=top.id,
                lane=top.lane,
                memory_type=top.memory_type,
                text=top.text,
                tokens=top.tokens,
                provenance=dict(top.provenance or {}),
                lexical_score=float(top.lexical_score or 0.0),
                semantic_score=float(top.semantic_score or 0.0),
                recency_score=float(top.recency_score or 0.0),
                freshness_score=float(top.freshness_score or 0.0),
                trust_score=float(top.trust_score or 0.0),
                novelty_score=float(top.novelty_score or 0.0),
                contradiction_risk=float(top.contradiction_risk or 0.0),
                cluster_id=top.cluster_id,
            )
        )
    selected = [(c, float(c.semantic_score + c.lexical_score + c.trust_score)) for c in explanation_candidates]
    explanation = build_explanation(
        intent_family=IntentFamily.MEMORY_SUMMARY,
        total_budget=1200,
        reserved_tokens=180,
        selected=selected,
        dropped=[],
    ).model_dump(mode="json")
    (outdir / "explanation_lifecycle_artifact.json").write_text(json.dumps(explanation, indent=2), encoding="utf-8")

    checks = {
        "explicit_lifecycle_state_present": bool(
            dict(lifecycle_artifact["webPromotedLifecycle"]["wp_unreviewed"]).get("lifecycle")
            and dict(lifecycle_artifact["webPromotedLifecycle"]["wp_approved"]).get("lifecycle")
            and dict(lifecycle_artifact["webPromotedLifecycle"]["wp_stale"]).get("lifecycle")
            and dict(lifecycle_artifact["reasoningPromotedLifecycle"]).get("lifecycle")
        ),
        "unreviewed_state_testable": dict(lifecycle_artifact["webPromotedLifecycle"]["wp_unreviewed"]).get("lifecycle", {}).get("lifecycle_state")
        == "unreviewed",
        "stale_revalidation_testable": dict(lifecycle_artifact["webPromotedLifecycle"]["wp_stale"]).get("lifecycle", {}).get("lifecycle_state")
        in {"revalidation_required", "stale"},
        "provenance_preserved": bool(
            dict(lifecycle_artifact["webPromotedLifecycle"]["wp_approved"]).get("metadata")
            and dict(lifecycle_artifact["reasoningPromotedLifecycle"]).get("reasoning_provenance")
        ),
        "explanation_lifecycle_visible": any(
            bool(dict(block.get("provenance") or {}).get("lifecycle"))
            for block in list(explanation.get("selected_blocks") or [])
        ),
    }
    summary = {
        "schemaVersion": "f015_lifecycle_slice_summary.v1",
        "artifactDir": str(outdir),
        "checks": checks,
        "traceability": {
            "features": ["F-015", "F-024(min-boundary)"],
            "requirements": ["REQ-032"],
            "workflow": ["WF-005"],
            "scenario": ["TC-005"],
            "invariants": ["INV-009", "INV-016"],
        },
        "passed": all(checks.values()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
