from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.retrieval.router import (
    clear_routing_caches,
    get_hot_cache_stats,
    get_route_cache_stats,
    get_topic_shortlist_cache_stats,
    route_and_retrieve,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("i-fast-%Y%m%dT%H%M%SZ")


def run_fast_retrieval_benchmark(
    *,
    workdir: str,
    query: str,
    intent_family: IntentFamily = IntentFamily.MEMORY_SUMMARY,
    session_id: str = "bench-fast-retrieval",
    repeats: int = 3,
    top_k_per_lane: int = 6,
) -> Dict[str, Any]:
    clear_routing_caches()

    runs = []
    for idx in range(max(1, int(repeats))):
        start = time.perf_counter()
        plan, hits = route_and_retrieve(
            query=query,
            intent_family=intent_family,
            session_id=session_id,
            fresh_tail=[],
            older_raw=[],
            summaries=[],
            workdir=workdir,
            top_k_per_lane=top_k_per_lane,
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        cache_hit = bool("agent_hot_cache_hit" in (plan.reason or ""))
        runs.append(
            {
                "run_index": idx,
                "latency_ms": duration_ms,
                "cache_hit": cache_hit,
                "hit_count": len(hits),
                "lane_values": sorted({h.lane.value for h in hits}),
                "category_routing": dict((plan.metadata or {}).get("category_routing") or {}),
                "rerank": dict((plan.metadata or {}).get("rerank") or {}),
            }
        )

    latencies = [float(r["latency_ms"]) for r in runs]
    payload = {
        "benchmark_type": "fast_retrieval_latency",
        "run_id": _run_id(),
        "query": query,
        "intent_family": intent_family.value,
        "repeat_count": len(runs),
        "latency": {
            "runs_ms": latencies,
            "avg_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "min_ms": min(latencies) if latencies else 0.0,
            "max_ms": max(latencies) if latencies else 0.0,
        },
        "category_routing": {
            "strong_signal_runs": sum(
                1
                for r in runs
                if bool(dict(r.get("category_routing") or {}).get("strong_signal"))
            ),
            "pruned_hit_count_total": sum(
                int(dict(r.get("category_routing") or {}).get("pruned_hit_count") or 0)
                for r in runs
            ),
            "fallback_lane_events": sum(
                len(list(dict(r.get("category_routing") or {}).get("fallback_lanes") or []))
                for r in runs
            ),
        },
        "cache": {
            "run_hits": sum(1 for r in runs if r["cache_hit"]),
            "run_misses": sum(1 for r in runs if not r["cache_hit"]),
            "router_hot_cache": get_hot_cache_stats(),
            "route_cache": get_route_cache_stats(),
            "topic_shortlist_cache": get_topic_shortlist_cache_stats(),
        },
        "rerank": {
            "applied_runs": sum(1 for r in runs if bool(dict(r.get("rerank") or {}).get("applied"))),
            "fallback_runs": sum(1 for r in runs if not bool(dict(r.get("rerank") or {}).get("applied"))),
            "moved_count_total": sum(int(dict(r.get("rerank") or {}).get("moved_count") or 0) for r in runs),
        },
        "runs": runs,
    }
    return payload


def save_fast_retrieval_benchmark(payload: Dict[str, Any], *, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    rid = str(payload.get("run_id") or _run_id())
    path = os.path.join(output_dir, f"{rid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path
