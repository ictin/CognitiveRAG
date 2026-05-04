#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import time
from pathlib import Path

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.router import clear_routing_caches, route_and_retrieve


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _scenario(workdir: str):
    return dict(
        query="speed slice deterministic route ordering and metadata parity",
        intent_family=IntentFamily.INVESTIGATIVE,
        session_id="f017-speed-scenario",
        fresh_tail=[{"index": 0, "text": "fresh tail speed seed", "sender": "user"}],
        older_raw=[
            {"index": 1, "text": "older raw source class web promoted metadata", "sender": "assistant"},
            {"index": 2, "text": "older raw lifecycle revalidation pending mention", "sender": "assistant"},
        ],
        summaries=[{"chunk_index": 0, "summary": "speed summary seed for route"}],
        workdir=workdir,
        top_k_per_lane=8,
    )


def _run_iters(*, workdir: str, legacy: bool, iters: int = 300):
    if legacy:
        os.environ["CRAG_F017_LEGACY_SORT"] = "1"
    else:
        os.environ.pop("CRAG_F017_LEGACY_SORT", None)
    clear_routing_caches()
    lat_ms = []
    last = None
    for _ in range(iters):
        clear_routing_caches()
        t0 = time.perf_counter()
        plan, hits = route_and_retrieve(**_scenario(workdir))
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        last = (plan, hits)
    return lat_ms, last


def main() -> int:
    stamp = _stamp()
    outdir = Path("forensics") / f"{stamp}_f017_safe_hot_path_speed_slice"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    before_lat, before_last = _run_iters(workdir=str(workdir), legacy=True)
    after_lat, after_last = _run_iters(workdir=str(workdir), legacy=False)
    before_plan, before_hits = before_last
    after_plan, after_hits = after_last

    before_artifact = {
        "mode": "legacy_sort",
        "iterations": len(before_lat),
        "mean_ms": statistics.mean(before_lat),
        "p50_ms": statistics.median(before_lat),
        "min_ms": min(before_lat),
        "max_ms": max(before_lat),
    }
    after_artifact = {
        "mode": "optimized_sort",
        "iterations": len(after_lat),
        "mean_ms": statistics.mean(after_lat),
        "p50_ms": statistics.median(after_lat),
        "min_ms": min(after_lat),
        "max_ms": max(after_lat),
    }
    (outdir / "before_latency_artifact.json").write_text(json.dumps(before_artifact, indent=2), encoding="utf-8")
    (outdir / "after_latency_artifact.json").write_text(json.dumps(after_artifact, indent=2), encoding="utf-8")

    semantic_equivalence = {
        "route_lanes_before": [x.value for x in before_plan.lanes],
        "route_lanes_after": [x.value for x in after_plan.lanes],
        "selected_hit_ids_before": [h.id for h in before_hits],
        "selected_hit_ids_after": [h.id for h in after_hits],
        "equivalent": [h.id for h in before_hits] == [h.id for h in after_hits]
        and [x.value for x in before_plan.lanes] == [x.value for x in after_plan.lanes],
    }
    (outdir / "semantic_equivalence_artifact.json").write_text(json.dumps(semantic_equivalence, indent=2), encoding="utf-8")

    truth_rows = []
    for a, b in zip(before_hits, after_hits):
        truth_rows.append(
            {
                "id": a.id,
                "lane_before": a.lane.value,
                "lane_after": b.lane.value,
                "memory_type_before": a.memory_type.value,
                "memory_type_after": b.memory_type.value,
                "source_class_before": (a.provenance or {}).get("source_class"),
                "source_class_after": (b.provenance or {}).get("source_class"),
                "lifecycle_before": (a.provenance or {}).get("lifecycle"),
                "lifecycle_after": (b.provenance or {}).get("lifecycle"),
            }
        )
    (outdir / "truthfulness_comparison_artifact.json").write_text(json.dumps(truth_rows, indent=2), encoding="utf-8")

    improvement = before_artifact["mean_ms"] - after_artifact["mean_ms"]
    summary = {
        "schemaVersion": "f017_speed_slice.v1",
        "artifactDir": str(outdir),
        "checks": {
            "before_after_latency_measured": True,
            "measurable_overhead_reduction": improvement > 0.0,
            "semantic_equivalence": bool(semantic_equivalence["equivalent"]),
            "source_class_preserved": all(r["source_class_before"] == r["source_class_after"] for r in truth_rows),
            "lifecycle_metadata_preserved": all(r["lifecycle_before"] == r["lifecycle_after"] for r in truth_rows),
            "selector_authority_preserved": semantic_equivalence["route_lanes_before"] == semantic_equivalence["route_lanes_after"],
        },
        "delta_ms_mean": improvement,
        "traceability": {
            "features": ["F-017"],
            "requirements": ["REQ-033", "REQ-034"],
            "invariants": ["INV-004", "INV-023"],
            "workflows": ["WF-003", "WF-007"],
        },
    }
    summary["passed"] = all(summary["checks"].values())
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
