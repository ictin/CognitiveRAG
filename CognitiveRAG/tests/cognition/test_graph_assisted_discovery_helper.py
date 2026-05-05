from CognitiveRAG.crag.cognition.discovery import DiscoveryExecutor, DiscoveryPolicy
from CognitiveRAG.crag.contracts.enums import DiscoveryMode, IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, DiscoveryPlan, RoleProbe


def _candidate(cid: str, lane: RetrievalLane, text: str, provenance: dict) -> ContextCandidate:
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=MemoryType.CORPUS_CHUNK,
        text=text,
        provenance=provenance,
    )


def test_graph_helper_suggestions_are_traceable_and_bounded(monkeypatch):
    monkeypatch.delenv("CRAG_DISABLE_GRAPH_DISCOVERY_HELPER", raising=False)
    plan = DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        bounded=True,
        role_conditioned_probes=[
            RoleProbe(
                role="skeptic",
                prompt="Find conflicting migration evidence",
                purpose="contradiction search",
                expected_lanes=[RetrievalLane.SEMANTIC, RetrievalLane.CORPUS],
                priority=1,
            )
        ],
    )
    pool = [
        _candidate(
            "a",
            RetrievalLane.CORPUS,
            "Migration rollback checklist with timeout fallback",
            {
                "category_graph": {"categories": [{"category": "engineering_db", "score": 0.8}]},
                "topic_graph": {"topics": [{"topic": "migration_rollout_safety", "score": 0.7}]},
                "clustering_helper": {"cluster_id": "cl:abc"},
                "source_class": "corpus",
            },
        ),
        _candidate("b", RetrievalLane.SEMANTIC, "Feature flag contradiction for rollout readiness", {"source_class": "reasoning"}),
    ]

    result = DiscoveryExecutor(
        DiscoveryPolicy(max_branches=2, max_evidence_per_branch=2, injection_budget_tokens=80, max_injected_discoveries=2)
    ).run(plan=plan, candidate_pool=pool)

    assert result.bounded is True
    assert result.used_tokens <= result.budget_tokens
    helper = dict(result.helper_metadata or {})
    assert helper.get("helper_enabled") is True
    assert int(helper.get("suggested_branch_count") or 0) >= 1
    assert any(b.helper_source_type in {"category_graph", "topic_graph", "clustering"} for b in result.ledger.explored_branches + result.ledger.rejected_branches)
    assert any("graph_helper" in (d.provenance or {}) for d in result.injected_discoveries)


def test_graph_helper_disable_fallback_keeps_discovery_operational(monkeypatch):
    monkeypatch.setenv("CRAG_DISABLE_GRAPH_DISCOVERY_HELPER", "1")
    plan = DiscoveryPlan(
        intent_family=IntentFamily.INVESTIGATIVE,
        discovery_mode=DiscoveryMode.ACTIVE,
        bounded=True,
        role_conditioned_probes=[
            RoleProbe(
                role="skeptic",
                prompt="Find conflicting migration evidence",
                purpose="contradiction search",
                expected_lanes=[RetrievalLane.SEMANTIC],
                priority=1,
            )
        ],
    )
    pool = [
        _candidate("x", RetrievalLane.SEMANTIC, "Rollback contradiction evidence", {"source_class": "reasoning"}),
        _candidate("y", RetrievalLane.SEMANTIC, "Rollback does not need checks", {"source_class": "reasoning"}),
    ]
    result = DiscoveryExecutor().run(plan=plan, candidate_pool=pool)
    assert result.bounded is True
    assert result.helper_metadata.get("helper_enabled") is False
    assert int(result.helper_metadata.get("suggested_branch_count") or 0) == 0
