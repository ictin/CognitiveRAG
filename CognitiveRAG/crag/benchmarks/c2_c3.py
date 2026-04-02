from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List
from uuid import uuid4

from CognitiveRAG.crag.cognition.discovery import DiscoveryExecutor, DiscoveryPolicy
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, DiscoveryPlan
from CognitiveRAG.session_memory.context_window import assemble_context


@dataclass(frozen=True)
class AssembleBenchmarkCase:
    case_id: str
    session_id: str
    query: str
    budget: int = 1400
    fresh_tail_count: int = 8


@dataclass(frozen=True)
class DiscoveryBenchmarkCase:
    case_id: str
    plan: DiscoveryPlan
    candidate_pool: List[ContextCandidate]
    policy: DiscoveryPolicy
    expect_bounded: bool = True
    expect_contradictions: bool = False


def _run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'{prefix}-{stamp}-{uuid4().hex[:8]}'


def _duration_ms(start: float, end: float) -> float:
    return round((end - start) * 1000.0, 3)


def _classify_drift(reference: dict[str, Any], current: dict[str, Any]) -> str:
    if reference == current:
        return 'same'
    same_selected = reference.get('selected_item_ids') == current.get('selected_item_ids')
    same_dropped = reference.get('dropped_item_ids') == current.get('dropped_item_ids')
    same_reorder = reference.get('reorder_strategy') == current.get('reorder_strategy')
    if same_selected and (same_dropped or same_reorder):
        return 'partial_drift'
    return 'changed'


def run_assemble_latency_benchmark(
    *,
    case: AssembleBenchmarkCase,
    repeats: int = 3,
    run_id: str | None = None,
) -> Dict[str, Any]:
    if repeats < 1:
        repeats = 1

    result_id = run_id or _run_id('assemble')
    runs: List[Dict[str, Any]] = []

    for run_index in range(repeats):
        start = time.perf_counter()
        assembled = assemble_context(
            case.session_id,
            fresh_tail_count=case.fresh_tail_count,
            budget=case.budget,
            query=case.query,
        )
        end = time.perf_counter()

        explanation = assembled.get('explanation', {})
        selected_item_ids = [b.get('id') for b in explanation.get('selected_blocks', []) if b.get('id')]
        dropped_item_ids = [b.get('id') for b in explanation.get('dropped_blocks', []) if b.get('id')]
        runs.append(
            {
                'run_index': run_index,
                'latency_ms': _duration_ms(start, end),
                'selected_item_ids': selected_item_ids,
                'dropped_item_ids': dropped_item_ids,
                'reorder_strategy': explanation.get('reorder_strategy', 'unknown'),
                'selected_count': len(selected_item_ids),
                'dropped_count': len(dropped_item_ids),
            }
        )

    reference = {
        'selected_item_ids': runs[0]['selected_item_ids'],
        'dropped_item_ids': runs[0]['dropped_item_ids'],
        'reorder_strategy': runs[0]['reorder_strategy'],
    }
    drift_markers = []
    for run in runs:
        drift_markers.append(
            _classify_drift(
                reference,
                {
                    'selected_item_ids': run['selected_item_ids'],
                    'dropped_item_ids': run['dropped_item_ids'],
                    'reorder_strategy': run['reorder_strategy'],
                },
            )
        )

    if all(marker == 'same' for marker in drift_markers):
        drift_summary = 'same'
    elif all(marker in {'same', 'partial_drift'} for marker in drift_markers):
        drift_summary = 'partial_drift'
    else:
        drift_summary = 'changed'

    latencies = [run['latency_ms'] for run in runs]
    return {
        'run_id': result_id,
        'benchmark_type': 'assemble_latency_stability',
        'case_id': case.case_id,
        'session_id': case.session_id,
        'query': case.query,
        'repeat_count': repeats,
        'latency': {
            'runs_ms': latencies,
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'avg_ms': round(sum(latencies) / len(latencies), 3),
        },
        'explanation_stability': {
            'drift_summary': drift_summary,
            'strategy': runs[0]['reorder_strategy'] if runs else 'unknown',
            'baseline_selected_item_ids': reference['selected_item_ids'],
            'baseline_dropped_item_ids': reference['dropped_item_ids'],
            'run_drift_markers': drift_markers,
        },
        'runs': runs,
    }


def run_discovery_latency_benchmark(
    *,
    case: DiscoveryBenchmarkCase,
    repeats: int = 1,
    run_id: str | None = None,
) -> Dict[str, Any]:
    if repeats < 1:
        repeats = 1

    result_id = run_id or _run_id('discovery')
    runs: List[Dict[str, Any]] = []

    for run_index in range(repeats):
        start = time.perf_counter()
        result = DiscoveryExecutor(case.policy).run(plan=case.plan, candidate_pool=case.candidate_pool)
        end = time.perf_counter()

        runs.append(
            {
                'run_index': run_index,
                'latency_ms': _duration_ms(start, end),
                'bounded': bool(result.bounded),
                'used_tokens': int(result.used_tokens),
                'budget_tokens': int(result.budget_tokens),
                'injected_count': len(result.injected_discoveries),
                'contradiction_count': len(result.contradictions),
                'explored_branch_count': len(result.ledger.explored_branches),
                'rejected_branch_count': len(result.ledger.rejected_branches),
            }
        )

    bounded_ok = all(run['bounded'] and run['used_tokens'] <= run['budget_tokens'] for run in runs)
    contradiction_visible = any(run['contradiction_count'] > 0 for run in runs)

    latencies = [run['latency_ms'] for run in runs]
    return {
        'run_id': result_id,
        'benchmark_type': 'discovery_latency_case',
        'case_id': case.case_id,
        'repeat_count': repeats,
        'latency': {
            'runs_ms': latencies,
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'avg_ms': round(sum(latencies) / len(latencies), 3),
        },
        'checks': {
            'boundedness': {
                'expected': case.expect_bounded,
                'actual': bounded_ok,
                'status': 'pass' if (bounded_ok == case.expect_bounded) else 'fail',
            },
            'contradiction_visibility': {
                'expected': case.expect_contradictions,
                'actual': contradiction_visible,
                'status': 'pass' if (contradiction_visible == case.expect_contradictions) else 'fail',
            },
        },
        'runs': runs,
    }


def save_benchmark_result(payload: Dict[str, Any], *, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def run_c2_c3_benchmark_suite(
    *,
    assemble_cases: Iterable[AssembleBenchmarkCase],
    discovery_cases: Iterable[DiscoveryBenchmarkCase],
    assemble_repeats: int = 3,
    discovery_repeats: int = 1,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    suite_run_id = _run_id('c2-c3-suite')
    out_dir = output_dir or os.path.join(os.getcwd(), 'data', 'benchmarks', 'c2_c3')

    assemble_results = [
        run_assemble_latency_benchmark(case=case, repeats=assemble_repeats, run_id=suite_run_id)
        for case in assemble_cases
    ]
    discovery_results = [
        run_discovery_latency_benchmark(case=case, repeats=discovery_repeats, run_id=suite_run_id)
        for case in discovery_cases
    ]

    critical_failures = []
    for result in discovery_results:
        for check_name, check in result.get('checks', {}).items():
            if check.get('status') != 'pass':
                critical_failures.append({'case_id': result.get('case_id'), 'check': check_name})

    aggregate = {
        'suite_run_id': suite_run_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'assemble_case_count': len(assemble_results),
        'discovery_case_count': len(discovery_results),
        'critical_failure_count': len(critical_failures),
        'critical_failures': critical_failures,
        'assemble': assemble_results,
        'discovery': discovery_results,
    }

    path = save_benchmark_result(aggregate, output_dir=out_dir, filename=f'{suite_run_id}.json')
    aggregate['output_path'] = path
    return aggregate
