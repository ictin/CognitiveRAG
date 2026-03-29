from CognitiveRAG.crag.cognition.discovery import DiscoveryExecutor, DiscoveryPolicy
from CognitiveRAG.crag.contracts.enums import DiscoveryMode, IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, DiscoveryPlan, RoleProbe


def _candidate(cid, lane, text, contradiction_risk=0.0, cluster_id=None):
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=MemoryType.SUMMARY if lane == RetrievalLane.DISCOVERY else MemoryType.CORPUS_CHUNK,
        text=text,
        contradiction_risk=contradiction_risk,
        cluster_id=cluster_id,
    )


def test_discovery_consumes_plan_and_produces_bounded_injection_with_ledger():
    plan = DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        web_sensitive=False,
        risk_mode='normal',
        bounded=True,
        query_variants=['investigate migration risk'],
        adjacent_topics=['migration', 'rollback'],
        expected_lanes=[RetrievalLane.SEMANTIC, RetrievalLane.CORPUS, RetrievalLane.EPISODIC],
        role_conditioned_probes=[
            RoleProbe(
                role='skeptic',
                prompt='Find conflicting migration evidence',
                purpose='contradiction search',
                expected_lanes=[RetrievalLane.SEMANTIC, RetrievalLane.EPISODIC],
                priority=1,
            ),
            RoleProbe(
                role='domain specialist',
                prompt='Find novel migration signals',
                purpose='novelty search',
                expected_lanes=[RetrievalLane.CORPUS, RetrievalLane.SEMANTIC],
                priority=2,
            ),
        ],
    )
    pool = [
        _candidate('a', RetrievalLane.SEMANTIC, 'Migration works with staged rollout and rollback plan', cluster_id='m1'),
        _candidate('b', RetrievalLane.EPISODIC, 'Migration does not work without rollback checks', contradiction_risk=0.6, cluster_id='m1'),
        _candidate('c', RetrievalLane.CORPUS, 'Checklist includes rollback rehearsals and incident drills', cluster_id='m2'),
        _candidate('d', RetrievalLane.CORPUS, 'Unrelated gardening note for tomatoes', cluster_id='x1'),
    ]

    result = DiscoveryExecutor(
        DiscoveryPolicy(max_branches=2, max_evidence_per_branch=2, injection_budget_tokens=80, max_injected_discoveries=2)
    ).run(plan=plan, candidate_pool=pool)

    assert result.bounded is True
    assert result.used_tokens <= result.budget_tokens
    assert len(result.injected_discoveries) <= 2
    assert len(result.ledger.explored_branches) >= 1


def test_contradiction_records_are_emitted_not_silently_merged():
    plan = DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        role_conditioned_probes=[
            RoleProbe(
                role='skeptic',
                prompt='Compare contradictions about feature flag status',
                purpose='contradiction',
                expected_lanes=[RetrievalLane.SEMANTIC],
                priority=1,
            )
        ],
        expected_lanes=[RetrievalLane.SEMANTIC],
    )
    pool = [
        _candidate('p', RetrievalLane.SEMANTIC, 'Feature flag is enabled in production for all users'),
        _candidate('n', RetrievalLane.SEMANTIC, 'Feature flag is not enabled in production for all users'),
    ]

    result = DiscoveryExecutor().run(plan=plan, candidate_pool=pool)
    assert len(result.contradictions) >= 1
