#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.graph_memory.category_graph import (
    categories_for_hit_from_graph,
    read_categories_for_node,
    record_category_relations_for_hits,
)
from CognitiveRAG.crag.graph_memory.schemas import GraphRelationType, stable_node_id
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.crag.retrieval.models import LaneHit
from CognitiveRAG.crag.retrieval.router import clear_routing_caches, route_and_retrieve


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _seed_hit() -> LaneHit:
    return (
        LaneHit(
            id="webpromoted:wp_pg",
            lane=RetrievalLane.WEB,
            memory_type=MemoryType.WEB_PROMOTED_FACT,
            text="Postgres migration rollback checklist for backend API service.",
            provenance={"promoted_id": "wp_pg", "source_url": "https://example.com/pg"},
            lexical_score=0.4,
            semantic_score=0.6,
            recency_score=0.4,
            freshness_score=0.7,
            trust_score=0.8,
            novelty_score=0.2,
            contradiction_risk=0.0,
        ).with_token_estimate()
    )


def _run_route(workdir: str, *, disable_helper: bool) -> tuple[list[str], dict, list[dict]]:
    clear_routing_caches()
    if disable_helper:
        os.environ["CRAG_DISABLE_CATEGORY_GRAPH"] = "1"
    else:
        os.environ.pop("CRAG_DISABLE_CATEGORY_GRAPH", None)
    plan, hits = route_and_retrieve(
        query="postgres migration schema rollback",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="f019-helper-slice",
        fresh_tail=[],
        older_raw=[],
        summaries=[],
        workdir=workdir,
        top_k_per_lane=6,
    )
    hit_rows = [
        {
            "id": h.id,
            "lane": h.lane.value,
            "memory_type": h.memory_type.value,
            "source_class": (h.provenance or {}).get("source_class"),
            "category_graph": (h.provenance or {}).get("category_graph"),
        }
        for h in hits
    ]
    return [lane.value for lane in plan.lanes], dict((plan.metadata or {}).get("category_routing") or {}), hit_rows


def _seed_reasoning_db(workdir: Path) -> None:
    db_path = workdir / "reasoning.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (pattern_id TEXT PRIMARY KEY, problem_signature TEXT, reasoning_steps_json TEXT, "
            "solution_summary TEXT, confidence REAL, provenance_json TEXT, memory_subtype TEXT, normalized_text TEXT, freshness_state TEXT)"
        )
        db.execute(
            "INSERT OR REPLACE INTO reasoning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "p_f019",
                "postgres migration rollback",
                "[]",
                "Use staged rollback checklist with transaction guard.",
                0.82,
                "[]",
                "workflow_pattern",
                "postgres migration rollback workflow",
                "current",
            ),
        )
        db.commit()


def _seed_corpus_db(workdir: Path) -> None:
    db_path = workdir / "context_items.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS context_items (item_id TEXT, session_id TEXT, type TEXT, payload_json TEXT, created_at TEXT)"
        )
        payload = {
            "summary": "Postgres migration rollback checklist and backend API safety gates.",
            "file_path": "books/postgres_ops.md",
        }
        db.execute(
            "INSERT INTO context_items VALUES (?, ?, ?, ?, ?)",
            ("f019-corpus-1", "f019-session", "corpus_chunk", json.dumps(payload), "2026-05-04T12:00:00Z"),
        )
        db.commit()


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f019_category_graph_helper_slice"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    _seed_reasoning_db(workdir)
    _seed_corpus_db(workdir)

    store = GraphMemoryStore(workdir / "graph_memory.sqlite3")
    hit = _seed_hit()
    created = record_category_relations_for_hits(store, hits=[hit], now_iso="2026-05-04T12:00:00Z")
    node_rows = read_categories_for_node(store, node_type="web_promoted", node_key="wp_pg")
    helper_rows = categories_for_hit_from_graph(store, hit)
    node_id = stable_node_id("web_promoted", "wp_pg")
    edges = store.get_edges_for_node(node_id, direction="outgoing")

    (outdir / "category_graph_build_write_read_artifact.json").write_text(
        json.dumps(
            {
                "created": created,
                "node_id": node_id,
                "readback_rows": node_rows,
                "helper_rows": helper_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "category_provenance_artifact.json").write_text(
        json.dumps(
            [
                {
                    "edge_id": e.edge_id,
                    "relation_type": e.relation_type,
                    "properties": e.properties,
                    "provenance": e.provenance,
                }
                for e in edges
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "category_navigation_helper_artifact.json").write_text(
        json.dumps({"categories": helper_rows, "count": len(helper_rows)}, indent=2),
        encoding="utf-8",
    )

    lanes_enabled, meta_enabled, hits_enabled = _run_route(str(workdir), disable_helper=False)
    lanes_disabled, meta_disabled, hits_disabled = _run_route(str(workdir), disable_helper=True)
    os.environ.pop("CRAG_DISABLE_CATEGORY_GRAPH", None)

    (outdir / "selector_authority_preservation_artifact.json").write_text(
        json.dumps(
            {
                "lanes_enabled": lanes_enabled,
                "lanes_disabled": lanes_disabled,
                "selector_authority_preserved": lanes_enabled == lanes_disabled,
                "category_routing_enabled": meta_enabled,
                "category_routing_disabled": meta_disabled,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "category_graph_disabled_fallback_artifact.json").write_text(
        json.dumps(
            {
                "hits_enabled_count": len(hits_enabled),
                "hits_disabled_count": len(hits_disabled),
                "disabled_has_no_category_graph_payload": all(not h.get("category_graph") for h in hits_disabled),
                "enabled_sample": hits_enabled[:4],
                "disabled_sample": hits_disabled[:4],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    checks = {
        "graph_relations_written_read_back": bool(created) and bool(node_rows),
        "provenance_preserved_on_edges": all(
            e.relation_type == GraphRelationType.BELONGS_TO_CATEGORY and e.provenance.get("reason")
            == "deterministic_keyword_category_inference"
            for e in edges
        ),
        "helper_navigation_inspectable": bool(helper_rows),
        "selector_authority_preserved": lanes_enabled == lanes_disabled,
        "graph_disabled_core_fallback_works": len(hits_disabled) > 0
        and all(not h.get("category_graph") for h in hits_disabled),
    }
    summary = {
        "schemaVersion": "f019_category_graph_helper_slice.v1",
        "artifactDir": str(outdir),
        "checks": checks,
        "traceability": {
            "features": ["F-019", "F-013", "F-014"],
            "requirements": ["REQ-029", "REQ-030"],
            "invariants": ["INV-009", "INV-014"],
            "workflows": ["WF-005"],
        },
    }
    summary["passed"] = all(checks.values())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
