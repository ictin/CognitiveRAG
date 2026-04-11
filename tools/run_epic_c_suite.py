#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CognitiveRAG.crag.benchmarks.c2_c3 import AssembleBenchmarkCase, DiscoveryBenchmarkCase
from CognitiveRAG.crag.benchmarks.epic_c_suite import run_epic_c_suite
from CognitiveRAG.crag.cognition.discovery import DiscoveryPolicy
from CognitiveRAG.crag.contracts.enums import DiscoveryMode, IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, DiscoveryPlan, RoleProbe
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore
from CognitiveRAG.session_memory.context_window import WORKDIR


def _seed_raw(session_id: str, count: int = 12) -> None:
    os.makedirs(WORKDIR, exist_ok=True)
    rows = [{"index": i, "text": f"Epic C runtime benchmark line {i}"} for i in range(count)]
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle)


def _seed_fast_store(workdir: Path) -> None:
    store = WebPromotedMemoryStore(workdir / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_epic_c_cli",
        canonical_fact="Epic C suite seeded fact",
        evidence_ids=["ev-cli"],
        confidence=0.88,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/epic-c-cli", "source_class": "web_promoted"},
        now_iso="2026-04-11T10:00:00Z",
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        promotion_tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        origin_tier=WebPromotedMemoryStore.TIER_LOCAL,
        freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_FRESH,
        last_validated_at="2026-04-11T09:00:00Z",
    )


def _discovery_case() -> DiscoveryBenchmarkCase:
    plan = DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        expected_lanes=[RetrievalLane.CORPUS],
        role_conditioned_probes=[
            RoleProbe(
                role="analyst",
                prompt="Find bounded migration evidence",
                purpose="epic-c-suite",
                expected_lanes=[RetrievalLane.CORPUS],
                priority=1,
            )
        ],
    )
    pool = [
        ContextCandidate(
            id="disc-1",
            lane=RetrievalLane.CORPUS,
            memory_type=MemoryType.CORPUS_CHUNK,
            text="Rollback rehearsal should precede production rollout.",
        ),
        ContextCandidate(
            id="disc-2",
            lane=RetrievalLane.CORPUS,
            memory_type=MemoryType.CORPUS_CHUNK,
            text="Missing rehearsal increases incident risk.",
        ),
    ]
    return DiscoveryBenchmarkCase(
        case_id="epic-c-cli-discovery",
        plan=plan,
        candidate_pool=pool,
        policy=DiscoveryPolicy(max_branches=1, max_evidence_per_branch=2, injection_budget_tokens=60, max_injected_discoveries=2),
        expect_bounded=True,
        expect_contradictions=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser("run_epic_c_suite")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "benchmarks" / "epic_c"))
    parser.add_argument("--fast-workdir", default=str(REPO_ROOT / "data" / "benchmarks" / "epic_c_fast"))
    parser.add_argument("--session-id", default="epic-c-cli-session")
    parser.add_argument("--query", default="Investigate migration reliability and rollback readiness.")
    args = parser.parse_args()

    fast_workdir = Path(args.fast_workdir)
    fast_workdir.mkdir(parents=True, exist_ok=True)
    _seed_fast_store(fast_workdir)
    _seed_raw(args.session_id)

    result = run_epic_c_suite(
        assemble_cases=[
            AssembleBenchmarkCase(
                case_id="epic-c-cli-assemble",
                session_id=args.session_id,
                query=args.query,
                budget=1200,
                fresh_tail_count=6,
            )
        ],
        discovery_cases=[_discovery_case()],
        fast_workdir=str(fast_workdir),
        fast_query="epic c suite fast retrieval",
        output_dir=args.output_dir,
        assemble_repeats=2,
        discovery_repeats=1,
        fast_repeats=2,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
