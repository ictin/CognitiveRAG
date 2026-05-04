#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.router import clear_routing_caches, route_and_retrieve


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


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
                "p_f021",
                "postgres migration rollback timeout",
                "[]",
                "Use staged rollback checklist with timeout fallback and verification.",
                0.84,
                "[]",
                "workflow_pattern",
                "postgres migration rollback timeout workflow",
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
        payloads = [
            {
                "id": "f021-corpus-1",
                "summary": "Postgres migration rollback checklist with timeout fallback and rollout safety gates.",
                "file_path": "books/postgres_ops.md",
            },
            {
                "id": "f021-corpus-2",
                "summary": "API reliability playbook for rollback verification and incident mitigation.",
                "file_path": "books/reliability.md",
            },
        ]
        for p in payloads:
            db.execute(
                "INSERT INTO context_items VALUES (?, ?, ?, ?, ?)",
                (
                    p["id"],
                    "f021-session",
                    "corpus_chunk",
                    json.dumps({"summary": p["summary"], "file_path": p["file_path"]}),
                    "2026-05-04T12:00:00Z",
                ),
            )
        db.commit()


def _run_route(workdir: str, *, disable_helper: bool):
    clear_routing_caches()
    if disable_helper:
        os.environ["CRAG_DISABLE_CLUSTERING_HELPER"] = "1"
    else:
        os.environ.pop("CRAG_DISABLE_CLUSTERING_HELPER", None)

    plan, hits = route_and_retrieve(
        query="postgres migration rollback timeout",
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        session_id="f021-helper-slice",
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
            "cluster_id": h.cluster_id,
            "source_class": (h.provenance or {}).get("source_class"),
            "lifecycle_state": (h.provenance or {}).get("lifecycle_state"),
            "topic_graph": (h.provenance or {}).get("topic_graph"),
            "category_graph": (h.provenance or {}).get("category_graph"),
            "clustering_helper": (h.provenance or {}).get("clustering_helper"),
        }
        for h in hits
    ]
    meta = dict((plan.metadata or {}).get("clustering_helper") or {})
    return [l.value for l in plan.lanes], meta, hit_rows


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f021_clustering_helper_slice"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    _seed_reasoning_db(workdir)
    _seed_corpus_db(workdir)

    lanes_enabled, meta_enabled, hits_enabled = _run_route(str(workdir), disable_helper=False)
    lanes_disabled, meta_disabled, hits_disabled = _run_route(str(workdir), disable_helper=True)
    os.environ.pop("CRAG_DISABLE_CLUSTERING_HELPER", None)

    cluster_map = {}
    for row in hits_enabled:
        cid = str(row.get("cluster_id") or "")
        if not cid:
            continue
        cluster_map.setdefault(cid, []).append(
            {
                "hit_id": row["id"],
                "source_class": row.get("source_class"),
                "lifecycle_state": row.get("lifecycle_state"),
                "topic_graph_present": bool(row.get("topic_graph")),
                "category_graph_present": bool(row.get("category_graph")),
            }
        )

    (outdir / "clustering_helper_build_readback_artifact.json").write_text(
        json.dumps(
            {
                "cluster_count": len(cluster_map),
                "clusters": cluster_map,
                "route_meta": meta_enabled,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "cluster_membership_provenance_artifact.json").write_text(
        json.dumps({"hits_enabled": hits_enabled}, indent=2),
        encoding="utf-8",
    )
    (outdir / "selector_authority_preservation_artifact.json").write_text(
        json.dumps(
            {
                "lanes_enabled": lanes_enabled,
                "lanes_disabled": lanes_disabled,
                "selector_authority_preserved": lanes_enabled == lanes_disabled,
                "meta_enabled": meta_enabled,
                "meta_disabled": meta_disabled,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "clustering_disabled_fallback_artifact.json").write_text(
        json.dumps(
            {
                "disabled_has_no_cluster_ids": all(not h.get("cluster_id") for h in hits_disabled),
                "disabled_has_no_helper_payload": all(not h.get("clustering_helper") for h in hits_disabled),
                "enabled_sample": hits_enabled[:4],
                "disabled_sample": hits_disabled[:4],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "explanation_trace_helper_only_artifact.json").write_text(
        json.dumps(
            {
                "clustering_helper": meta_enabled,
                "helper_only_truth": {
                    "helper_only": bool(meta_enabled.get("helper_only")),
                    "authoritative": bool(meta_enabled.get("authoritative")) is False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    clustered_rows = [row for row in hits_enabled if row.get("clustering_helper")]

    checks = {
        "helper_clustering_inspectable": len(cluster_map) >= 1,
        "cluster_membership_has_provenance_fields": all(
            row.get("topic_graph") and row.get("category_graph") for row in clustered_rows
        ),
        "selector_authority_preserved": lanes_enabled == lanes_disabled,
        "clustering_disabled_fallback_works": all(not h.get("clustering_helper") for h in hits_disabled),
        "helper_only_truthful_labeling": bool(meta_enabled.get("helper_only"))
        and (bool(meta_enabled.get("authoritative")) is False),
    }

    summary = {
        "schemaVersion": "f021_clustering_helper_slice.v1",
        "artifactDir": str(outdir),
        "checks": checks,
        "traceability": {
            "features": ["F-021", "F-019", "F-020", "F-009"],
            "requirements": ["REQ-029", "REQ-030"],
            "invariants": ["INV-009", "INV-014", "INV-016"],
            "workflows": ["WF-005"],
        },
    }
    summary["passed"] = all(checks.values())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
