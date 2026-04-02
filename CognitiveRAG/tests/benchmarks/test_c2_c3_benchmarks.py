import json
import os
import shutil

from CognitiveRAG.crag.benchmarks.c2_c3 import (
    AssembleBenchmarkCase,
    DiscoveryBenchmarkCase,
    run_assemble_latency_benchmark,
    run_c2_c3_benchmark_suite,
    run_discovery_latency_benchmark,
)
from CognitiveRAG.crag.cognition.discovery import DiscoveryPolicy
from CognitiveRAG.crag.contracts.enums import DiscoveryMode, IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, DiscoveryPlan, RoleProbe
from CognitiveRAG.session_memory.context_window import WORKDIR


def _seed_raw_messages(session_id: str, count: int = 16) -> None:
    rows = [
        {
            'index': i,
            'text': f'Context stability benchmark line {i} about migration rollback and copywriting hooks.',
            'meta': {'source': 'bench'},
        }
        for i in range(count)
    ]
    os.makedirs(WORKDIR, exist_ok=True)
    with open(os.path.join(WORKDIR, f'raw_{session_id}.json'), 'w', encoding='utf-8') as handle:
        json.dump(rows, handle)


def _candidate(cid: str, lane: RetrievalLane, text: str) -> ContextCandidate:
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=MemoryType.CORPUS_CHUNK,
        text=text,
    )


def _bounded_discovery_case() -> DiscoveryBenchmarkCase:
    plan = DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        expected_lanes=[RetrievalLane.CORPUS, RetrievalLane.SEMANTIC],
        role_conditioned_probes=[
            RoleProbe(
                role='analyst',
                prompt='Find bounded evidence about migration rollout risk',
                purpose='boundedness check',
                expected_lanes=[RetrievalLane.CORPUS, RetrievalLane.SEMANTIC],
                priority=1,
            )
        ],
    )
    pool = [
        _candidate('b1', RetrievalLane.CORPUS, 'Migration rollout requires staged guardrails and rollback plan.'),
        _candidate('b2', RetrievalLane.SEMANTIC, 'Risk emerges when rollback drills are skipped in production.'),
        _candidate('b3', RetrievalLane.CORPUS, 'Unrelated notes about gardening and tomatoes.'),
    ]
    return DiscoveryBenchmarkCase(
        case_id='bounded-discovery-case',
        plan=plan,
        candidate_pool=pool,
        policy=DiscoveryPolicy(max_branches=1, max_evidence_per_branch=2, injection_budget_tokens=40, max_injected_discoveries=1),
        expect_bounded=True,
        expect_contradictions=False,
    )


def _contradiction_discovery_case() -> DiscoveryBenchmarkCase:
    plan = DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        expected_lanes=[RetrievalLane.SEMANTIC],
        role_conditioned_probes=[
            RoleProbe(
                role='skeptic',
                prompt='Check contradiction for feature flag production status',
                purpose='contradiction visibility',
                expected_lanes=[RetrievalLane.SEMANTIC],
                priority=1,
            )
        ],
    )
    pool = [
        _candidate('c1', RetrievalLane.SEMANTIC, 'Feature flag is enabled in production for all users.'),
        _candidate('c2', RetrievalLane.SEMANTIC, 'Feature flag is not enabled in production for all users.'),
    ]
    return DiscoveryBenchmarkCase(
        case_id='contradiction-discovery-case',
        plan=plan,
        candidate_pool=pool,
        policy=DiscoveryPolicy(max_branches=1, max_evidence_per_branch=2, injection_budget_tokens=80, max_injected_discoveries=2),
        expect_bounded=True,
        expect_contradictions=True,
    )


def test_assemble_latency_benchmark_includes_numeric_latency_and_stability_shape():
    session_id = 'c2-bench-assemble'
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    _seed_raw_messages(session_id)

    result = run_assemble_latency_benchmark(
        case=AssembleBenchmarkCase(
            case_id='assemble-latency-case',
            session_id=session_id,
            query='Investigate migration risk and rollback sequence.',
            budget=1200,
            fresh_tail_count=6,
        ),
        repeats=3,
    )

    assert result['benchmark_type'] == 'assemble_latency_stability'
    assert result['repeat_count'] == 3
    assert result['latency']['avg_ms'] >= 0
    assert len(result['latency']['runs_ms']) == 3
    assert result['explanation_stability']['drift_summary'] in {'same', 'partial_drift', 'changed'}
    assert isinstance(result['explanation_stability']['baseline_selected_item_ids'], list)


def test_discovery_latency_benchmark_includes_boundedness_and_contradiction_checks():
    bounded_result = run_discovery_latency_benchmark(case=_bounded_discovery_case(), repeats=2)
    contradiction_result = run_discovery_latency_benchmark(case=_contradiction_discovery_case(), repeats=1)

    assert bounded_result['benchmark_type'] == 'discovery_latency_case'
    assert bounded_result['latency']['avg_ms'] >= 0
    assert bounded_result['checks']['boundedness']['status'] == 'pass'
    assert bounded_result['checks']['contradiction_visibility']['status'] == 'pass'

    assert contradiction_result['checks']['boundedness']['status'] == 'pass'
    assert contradiction_result['checks']['contradiction_visibility']['status'] == 'pass'
    assert contradiction_result['runs'][0]['contradiction_count'] >= 1


def test_c2_c3_suite_writes_stable_json_shape(tmp_path):
    session_id = 'c2-c3-suite-session'
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    _seed_raw_messages(session_id)

    suite = run_c2_c3_benchmark_suite(
        assemble_cases=[
            AssembleBenchmarkCase(
                case_id='suite-assemble-case',
                session_id=session_id,
                query='Investigate what else matters about migration reliability.',
                budget=1300,
                fresh_tail_count=7,
            )
        ],
        discovery_cases=[_bounded_discovery_case(), _contradiction_discovery_case()],
        assemble_repeats=2,
        discovery_repeats=1,
        output_dir=str(tmp_path),
    )

    assert suite['assemble_case_count'] == 1
    assert suite['discovery_case_count'] == 2
    assert 'output_path' in suite
    assert os.path.exists(suite['output_path'])

    with open(suite['output_path'], 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    assert payload['suite_run_id']
    assert 'critical_failure_count' in payload
    assert 'assemble' in payload and isinstance(payload['assemble'], list)
    assert 'discovery' in payload and isinstance(payload['discovery'], list)

    assemble_payload = payload['assemble'][0]
    assert {'run_id', 'case_id', 'repeat_count', 'latency', 'explanation_stability', 'runs'} <= set(assemble_payload)

    discovery_payload = payload['discovery'][0]
    assert {'run_id', 'case_id', 'repeat_count', 'latency', 'checks', 'runs'} <= set(discovery_payload)
