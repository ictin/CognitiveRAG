import json
import os
import shutil
from pathlib import Path

from CognitiveRAG.crag.benchmarks.epic_c_suite import run_epic_c_suite
from CognitiveRAG.crag.benchmarks.c2_c3 import AssembleBenchmarkCase, DiscoveryBenchmarkCase
from CognitiveRAG.crag.cognition.discovery import DiscoveryPolicy
from CognitiveRAG.crag.contracts.enums import DiscoveryMode, IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, DiscoveryPlan, RoleProbe
from CognitiveRAG.crag.web_memory.promoted_store import WebPromotedMemoryStore
from CognitiveRAG.session_memory.context_window import WORKDIR


def _seed_raw_messages(session_id: str, count: int = 12) -> None:
    rows = [{"index": i, "text": f"Epic C benchmark line {i} about migration rollback."} for i in range(count)]
    os.makedirs(WORKDIR, exist_ok=True)
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle)


def _seed_fast_store(workdir: Path) -> None:
    store = WebPromotedMemoryStore(workdir / "web_promoted_memory.sqlite3")
    store.upsert_fact(
        promoted_id="wp_epic_c",
        canonical_fact="Epic C benchmark seeded fact.",
        evidence_ids=["ev-1"],
        confidence=0.9,
        freshness_state="warm",
        metadata={"source_url": "https://example.com/epic-c", "source_class": "web_promoted"},
        now_iso="2026-04-10T10:00:00Z",
        promotion_state=WebPromotedMemoryStore.STATE_TRUSTED,
        promotion_tier=WebPromotedMemoryStore.TIER_WORKSPACE,
        origin_tier=WebPromotedMemoryStore.TIER_LOCAL,
        freshness_lifecycle_state=WebPromotedMemoryStore.FRESHNESS_FRESH,
        last_validated_at="2026-04-10T09:00:00Z",
    )


def _disc_case() -> DiscoveryBenchmarkCase:
    plan = DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        expected_lanes=[RetrievalLane.CORPUS],
        role_conditioned_probes=[
            RoleProbe(
                role="analyst",
                prompt="Find bounded evidence for migration risk",
                purpose="boundedness check",
                expected_lanes=[RetrievalLane.CORPUS],
                priority=1,
            )
        ],
    )
    pool = [
        ContextCandidate(
            id="d1",
            lane=RetrievalLane.CORPUS,
            memory_type=MemoryType.CORPUS_CHUNK,
            text="Rollback drills reduce deployment risk.",
        ),
        ContextCandidate(
            id="d2",
            lane=RetrievalLane.CORPUS,
            memory_type=MemoryType.CORPUS_CHUNK,
            text="Migration checklist includes staged rollout.",
        ),
    ]
    return DiscoveryBenchmarkCase(
        case_id="epic-c-discovery",
        plan=plan,
        candidate_pool=pool,
        policy=DiscoveryPolicy(max_branches=1, max_evidence_per_branch=2, injection_budget_tokens=60, max_injected_discoveries=2),
        expect_bounded=True,
        expect_contradictions=False,
    )


def test_run_epic_c_suite_writes_json_and_markdown(tmp_path: Path):
    session_id = "epic-c-suite-session"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    _seed_raw_messages(session_id)
    _seed_fast_store(tmp_path)

    out = run_epic_c_suite(
        assemble_cases=[
            AssembleBenchmarkCase(
                case_id="epic-c-assemble",
                session_id=session_id,
                query="Investigate migration reliability and rollback readiness.",
                budget=1200,
                fresh_tail_count=6,
            )
        ],
        discovery_cases=[_disc_case()],
        fast_workdir=str(tmp_path),
        fast_query="epic c fast retrieval benchmark",
        output_dir=str(tmp_path / "epic_c_out"),
        assemble_repeats=2,
        discovery_repeats=1,
        fast_repeats=2,
    )

    assert out["overall_status"] in {"pass", "fail"}
    assert out["c2_c3"]["assemble_case_count"] == 1
    assert out["c2_c3"]["discovery_case_count"] == 1
    assert out["fast_retrieval"]["repeat_count"] == 2
    assert os.path.exists(out["report_json_path"])
    assert os.path.exists(out["report_markdown_path"])

    with open(out["report_json_path"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["run_id"]
    assert {"c2_c3", "fast_retrieval", "overall_status"} <= set(payload)

    with open(out["report_markdown_path"], "r", encoding="utf-8") as handle:
        markdown = handle.read()
    assert "Epic C Benchmark Report" in markdown
    assert "C2/C3 Summary" in markdown
    assert "Fast Retrieval Summary" in markdown
