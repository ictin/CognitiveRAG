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
from CognitiveRAG.crag.graph_memory.schemas import GraphRelationType, stable_node_id
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.crag.retrieval.promoted_lane import retrieve as retrieve_promoted
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
                "route-safe fallback and provenance checks",
                "current",
            ),
        )
        db.commit()


def _as_edge_dict(edge: Any) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "source_node_id": edge.source_node_id,
        "relation_type": edge.relation_type,
        "target_node_id": edge.target_node_id,
        "properties": edge.properties,
        "provenance": edge.provenance,
    }


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_m004_first_graph_write_read_slice"
    outdir.mkdir(parents=True, exist_ok=True)

    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    pattern = ReasoningPattern(
        pattern_id="rp:m004-proof-1",
        item_id="rp:m004-proof-1",
        problem_signature="preserve provenance while adding graph relations",
        reasoning_steps=["write stable relation set", "read back and verify", "check non-graph fallback"],
        solution_summary="Attach graph edges only to stable promoted/reasoning/provenance objects.",
        confidence=0.86,
        provenance=['{"source":"m004-proof","kind":"reasoning"}'],
        memory_subtype="workflow_pattern",
    )

    _seed_reasoning_db(workdir, pattern)

    graph_path = workdir / "graph.sqlite3"
    store = GraphMemoryStore(graph_path)

    edge_supported = record_reasoning_pattern_supported_by(
        store,
        pattern=pattern,
        source={"source_url": "https://example.com/provenance-spec", "source": "spec"},
        provenance={"proof_run": stamp, "invariant": "INV-009"},
    )
    edge_derived = record_web_promoted_derived_from(
        store,
        promoted_id="wp:m004-proof-1",
        source_url="https://example.com/promoted-origin",
        metadata={"memory_class": "web_promoted"},
        provenance={"proof_run": stamp, "requirement": "REQ-030"},
    )
    edge_resolved = record_problem_signature_resolved_by(
        store,
        problem_signature="preserve provenance while adding graph relations",
        pattern=pattern,
        provenance={"proof_run": stamp, "workflow": "WF-005"},
    )

    roundtrip_edges = [
        store.get_edge(edge_supported.edge_id),
        store.get_edge(edge_derived.edge_id),
        store.get_edge(edge_resolved.edge_id),
    ]
    roundtrip_ok = all(e is not None for e in roundtrip_edges)

    relation_artifact = {
        "schemaVersion": "m004_first_graph_slice.v1",
        "stamp": stamp,
        "graphPath": str(graph_path),
        "stableNodeIds": {
            "reasoning_pattern": stable_node_id("reasoning_pattern", pattern.pattern_id),
            "problem_signature": stable_node_id("problem_signature", pattern.problem_signature),
            "web_promoted": stable_node_id("web_promoted", "wp:m004-proof-1"),
        },
        "relationsWritten": [
            _as_edge_dict(edge_supported),
            _as_edge_dict(edge_derived),
            _as_edge_dict(edge_resolved),
        ],
        "relationsReadBack": [_as_edge_dict(e) for e in roundtrip_edges if e is not None],
    }
    (outdir / "graph_relation_roundtrip.json").write_text(json.dumps(relation_artifact, indent=2), encoding="utf-8")

    # Graph helper-only safety proof: with graph DB absent, core promoted retrieval still works.
    if graph_path.exists():
        os.remove(graph_path)

    fallback_hits = retrieve_promoted(
        workdir=str(workdir),
        intent_family=IntentFamily.INVESTIGATIVE,
        query="preserve provenance while adding graph relations",
        top_k=3,
    )
    fallback_first = fallback_hits[0] if fallback_hits else None
    fallback_artifact = {
        "schemaVersion": "m004_graph_helper_only_fallback.v1",
        "graphDbPresent": graph_path.exists(),
        "fallbackHitCount": len(fallback_hits),
        "fallbackFirstHitId": fallback_first.id if fallback_first else "",
        "fallbackFirstProvenanceHasGraphKeys": bool(
            fallback_first
            and (
                "graph_support_links" in (fallback_first.provenance or {})
                or "graph_problem_signature_matches" in (fallback_first.provenance or {})
            )
        ),
    }
    (outdir / "graph_disabled_core_runtime_check.json").write_text(json.dumps(fallback_artifact, indent=2), encoding="utf-8")

    rel_types = [e.relation_type for e in roundtrip_edges if e is not None]
    summary = {
        "schemaVersion": "m004_first_graph_slice_summary.v1",
        "artifactDir": str(outdir),
        "checks": {
            "stable_graph_relation_set_written": len(relation_artifact["relationsWritten"]) == 3,
            "graph_roundtrip_readback_ok": bool(roundtrip_ok),
            "provenance_preserved_on_relations": all(bool((e.provenance or {})) for e in roundtrip_edges if e is not None),
            "relation_types_expected": sorted(rel_types)
            == sorted(
                [
                    GraphRelationType.SUPPORTED_BY,
                    GraphRelationType.DERIVED_FROM,
                    GraphRelationType.RESOLVED_BY,
                ]
            ),
            "core_runtime_works_without_graph_dependency": fallback_artifact["fallbackHitCount"] > 0
            and not fallback_artifact["fallbackFirstProvenanceHasGraphKeys"],
        },
        "traceability": {
            "features": ["F-013"],
            "requirements": ["REQ-029", "REQ-030"],
            "invariants": ["INV-009", "INV-014"],
            "workflow": ["WF-005"],
        },
    }
    summary["passed"] = all(summary["checks"].values())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
