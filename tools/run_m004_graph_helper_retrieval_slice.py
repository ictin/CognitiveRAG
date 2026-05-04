#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.graph_memory.relations import (
    record_problem_signature_resolved_by,
    record_reasoning_pattern_supported_by,
    record_web_promoted_derived_from,
)
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.crag.retrieval.promoted_lane import retrieve as retrieve_promoted
from CognitiveRAG.crag.retrieval.router import LaneRouter
from CognitiveRAG.crag.retrieval.web_lane import retrieve as retrieve_web
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore
from CognitiveRAG.schemas.memory import ReasoningPattern


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _seed_reasoning_db(workdir: Path, pattern: ReasoningPattern) -> None:
    db_path = workdir / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns ("
            "pattern_id TEXT PRIMARY KEY, "
            "problem_signature TEXT, "
            "reasoning_steps_json TEXT, "
            "solution_summary TEXT, "
            "confidence REAL, "
            "provenance_json TEXT, "
            "memory_subtype TEXT, "
            "normalized_text TEXT, "
            "freshness_state TEXT"
            ")"
        )
        db.execute(
            "INSERT OR REPLACE INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pattern.pattern_id,
                pattern.problem_signature,
                json.dumps(pattern.reasoning_steps),
                pattern.solution_summary,
                float(pattern.confidence),
                json.dumps(pattern.provenance),
                pattern.memory_subtype or "workflow_pattern",
                "graph helper retrieval bounded metadata",
                "current",
            ),
        )
        db.commit()


def _has_helper_signal(rows: list[Any]) -> bool:
    for row in rows:
        prov = dict(getattr(row, "provenance", {}) or {})
        ghs = dict(prov.get("graph_helper_signal") or {})
        if ghs.get("mode") == "helper_only":
            return True
    return False


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_m004_graph_helper_retrieval_slice"
    outdir.mkdir(parents=True, exist_ok=True)

    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    pattern = ReasoningPattern(
        pattern_id="rp:m004-helper-1",
        item_id="rp:m004-helper-1",
        problem_signature="route promoted retrieval using helper graph links",
        reasoning_steps=["retrieve promoted", "use helper graph bonus only", "preserve selector authority"],
        solution_summary="Graph adds bounded helper signals but selector remains authoritative.",
        confidence=0.84,
        provenance=['{"source":"m004-helper","kind":"reasoning"}'],
        memory_subtype="workflow_pattern",
    )
    _seed_reasoning_db(workdir, pattern)

    web_promoted = WebPromotedMemoryStore(workdir / "web_promoted_memory.sqlite3")
    web_promoted.upsert_fact(
        promoted_id="wp:m004-helper-web",
        canonical_fact="Helper graph origins should remain additive.",
        evidence_ids=["ev:m004-helper"],
        confidence=0.72,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/helper-origin"},
        now_iso="2026-05-04T00:00:00Z",
    )

    query = "route promoted retrieval using helper graph links"
    router = LaneRouter()
    plan_before = router.route(query=query, intent_family=IntentFamily.INVESTIGATIVE)

    promoted_before = retrieve_promoted(
        workdir=str(workdir),
        intent_family=IntentFamily.INVESTIGATIVE,
        query=query,
        top_k=4,
    )
    web_before = retrieve_web(
        workdir=str(workdir),
        query="helper graph origins",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=3,
    )

    graph_store = GraphMemoryStore(workdir / "graph.sqlite3")
    record_reasoning_pattern_supported_by(
        graph_store,
        pattern=pattern,
        source={"source_url": "https://example.com/helper-doc", "source": "spec"},
        provenance={"proof_run": stamp, "feature": "F-014"},
    )
    record_problem_signature_resolved_by(
        graph_store,
        problem_signature=pattern.problem_signature,
        pattern=pattern,
        provenance={"proof_run": stamp, "feature": "F-014"},
    )
    record_web_promoted_derived_from(
        graph_store,
        promoted_id="wp:m004-helper-web",
        source_url="https://example.com/helper-origin",
        metadata={"memory_class": "web_promoted"},
        provenance={"proof_run": stamp, "feature": "F-014"},
    )

    plan_after = router.route(query=query, intent_family=IntentFamily.INVESTIGATIVE)
    promoted_after = retrieve_promoted(
        workdir=str(workdir),
        intent_family=IntentFamily.INVESTIGATIVE,
        query=query,
        top_k=4,
    )
    web_after = retrieve_web(
        workdir=str(workdir),
        query="helper graph origins",
        intent_family=IntentFamily.INVESTIGATIVE,
        top_k=3,
    )

    helper_artifact = {
        "schemaVersion": "m004_graph_helper_retrieval.v1",
        "stamp": stamp,
        "selectorAuthority": {
            "planBefore": {"intent": plan_before.intent_family.value, "lanes": [x.value for x in plan_before.lanes], "reason": plan_before.reason},
            "planAfter": {"intent": plan_after.intent_family.value, "lanes": [x.value for x in plan_after.lanes], "reason": plan_after.reason},
            "unchangedByGraphHelper": [x.value for x in plan_before.lanes] == [x.value for x in plan_after.lanes],
        },
        "helperSignal": {
            "promoted_before_helper_signal": _has_helper_signal(promoted_before),
            "promoted_after_helper_signal": _has_helper_signal(promoted_after),
            "web_before_helper_signal": _has_helper_signal(web_before),
            "web_after_helper_signal": _has_helper_signal(web_after),
            "promoted_top_provenance": dict((promoted_after[0].provenance if promoted_after else {}) or {}),
            "web_top_provenance": dict((web_after[0].provenance if web_after else {}) or {}),
        },
    }
    (outdir / "graph_helper_retrieval_proof.json").write_text(json.dumps(helper_artifact, indent=2), encoding="utf-8")

    graph_path = workdir / "graph.sqlite3"
    if graph_path.exists():
        os.remove(graph_path)
    promoted_fallback = retrieve_promoted(
        workdir=str(workdir),
        intent_family=IntentFamily.INVESTIGATIVE,
        query=query,
        top_k=4,
    )
    fallback_artifact = {
        "schemaVersion": "m004_graph_helper_fallback.v1",
        "graphDbPresent": graph_path.exists(),
        "fallbackHitCount": len(promoted_fallback),
        "fallbackHasHelperSignal": _has_helper_signal(promoted_fallback),
    }
    (outdir / "graph_disabled_core_runtime_check.json").write_text(json.dumps(fallback_artifact, indent=2), encoding="utf-8")

    summary = {
        "schemaVersion": "m004_graph_helper_slice_summary.v1",
        "artifactDir": str(outdir),
        "checks": {
            "graph_helper_retrieval_artifact_written": True,
            "selector_authority_preserved": bool(helper_artifact["selectorAuthority"]["unchangedByGraphHelper"]),
            "helper_signal_visible_when_graph_links_exist": bool(
                helper_artifact["helperSignal"]["promoted_after_helper_signal"]
                or helper_artifact["helperSignal"]["web_after_helper_signal"]
            ),
            "core_runtime_works_with_graph_absent": fallback_artifact["fallbackHitCount"] > 0,
        },
        "traceability": {
            "features": ["F-014"],
            "requirements": ["REQ-029", "REQ-030"],
            "invariants": ["INV-010", "INV-011", "INV-014"],
            "workflow": ["WF-005"],
        },
    }
    summary["passed"] = all(summary["checks"].values())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
