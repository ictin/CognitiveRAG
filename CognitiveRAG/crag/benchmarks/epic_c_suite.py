from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable
from uuid import uuid4

from CognitiveRAG.crag.benchmarks.c2_c3 import (
    AssembleBenchmarkCase,
    DiscoveryBenchmarkCase,
    run_c2_c3_benchmark_suite,
)
from CognitiveRAG.crag.benchmarks.i_fast_retrieval import (
    run_fast_retrieval_benchmark,
    save_fast_retrieval_benchmark,
)
from CognitiveRAG.crag.contracts.enums import IntentFamily


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("epic-c-%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def _write_json(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def _write_markdown(path: str, lines: list[str]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return path


def run_epic_c_suite(
    *,
    assemble_cases: Iterable[AssembleBenchmarkCase],
    discovery_cases: Iterable[DiscoveryBenchmarkCase],
    fast_workdir: str,
    fast_query: str,
    output_dir: str | None = None,
    assemble_repeats: int = 3,
    discovery_repeats: int = 1,
    fast_repeats: int = 3,
    fast_intent: IntentFamily = IntentFamily.MEMORY_SUMMARY,
) -> Dict[str, Any]:
    run_id = _run_id()
    out_dir = output_dir or os.path.join(os.getcwd(), "data", "benchmarks", "epic_c")
    os.makedirs(out_dir, exist_ok=True)

    c2c3 = run_c2_c3_benchmark_suite(
        assemble_cases=list(assemble_cases),
        discovery_cases=list(discovery_cases),
        assemble_repeats=assemble_repeats,
        discovery_repeats=discovery_repeats,
        output_dir=out_dir,
    )

    fast_payload = run_fast_retrieval_benchmark(
        workdir=fast_workdir,
        query=fast_query,
        intent_family=fast_intent,
        repeats=fast_repeats,
    )
    fast_json_path = save_fast_retrieval_benchmark(fast_payload, output_dir=out_dir)

    c2_assemble_drift = [a.get("explanation_stability", {}).get("drift_summary", "unknown") for a in c2c3.get("assemble", [])]
    c2_changed_count = sum(1 for v in c2_assemble_drift if v == "changed")

    report = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "c2_c3": {
            "critical_failure_count": int(c2c3.get("critical_failure_count", 0)),
            "assemble_case_count": int(c2c3.get("assemble_case_count", 0)),
            "discovery_case_count": int(c2c3.get("discovery_case_count", 0)),
            "assemble_changed_drift_count": c2_changed_count,
            "output_path": c2c3.get("output_path"),
        },
        "fast_retrieval": {
            "repeat_count": int(fast_payload.get("repeat_count", 0)),
            "latency_avg_ms": float(fast_payload.get("latency", {}).get("avg_ms", 0.0)),
            "cache_hits": int(fast_payload.get("cache", {}).get("run_hits", 0)),
            "cache_misses": int(fast_payload.get("cache", {}).get("run_misses", 0)),
            "output_path": fast_json_path,
        },
    }
    report["overall_status"] = (
        "pass"
        if report["c2_c3"]["critical_failure_count"] == 0 and report["fast_retrieval"]["repeat_count"] > 0
        else "fail"
    )

    json_path = _write_json(os.path.join(out_dir, f"{run_id}.json"), report)
    md_lines = [
        "# Epic C Benchmark Report",
        "",
        f"- run_id: `{run_id}`",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- overall_status: `{report['overall_status']}`",
        "",
        "## C2/C3 Summary",
        f"- critical_failure_count: `{report['c2_c3']['critical_failure_count']}`",
        f"- assemble_case_count: `{report['c2_c3']['assemble_case_count']}`",
        f"- discovery_case_count: `{report['c2_c3']['discovery_case_count']}`",
        f"- assemble_changed_drift_count: `{report['c2_c3']['assemble_changed_drift_count']}`",
        f"- c2_c3_json: `{report['c2_c3']['output_path']}`",
        "",
        "## Fast Retrieval Summary",
        f"- repeat_count: `{report['fast_retrieval']['repeat_count']}`",
        f"- latency_avg_ms: `{report['fast_retrieval']['latency_avg_ms']}`",
        f"- cache_hits: `{report['fast_retrieval']['cache_hits']}`",
        f"- cache_misses: `{report['fast_retrieval']['cache_misses']}`",
        f"- fast_json: `{report['fast_retrieval']['output_path']}`",
    ]
    md_path = _write_markdown(os.path.join(out_dir, f"{run_id}.md"), md_lines)

    return {
        **report,
        "report_json_path": json_path,
        "report_markdown_path": md_path,
    }
