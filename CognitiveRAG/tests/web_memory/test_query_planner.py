from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.web_memory.query_planner import detect_web_need, plan_web_queries


def test_query_planner_freshness_sensitive():
    need = detect_web_need(
        query="latest postgres release notes",
        intent_family=IntentFamily.INVESTIGATIVE,
        local_evidence_count=2,
    )
    assert need.needed is True
    assert need.freshness_sensitive is True
    plan = plan_web_queries(query="latest postgres release notes", decision=need, max_variants=3)
    assert len(plan.variants) >= 1
    assert len(plan.variants) <= 3
    assert plan.freshness_sensitive is True


def test_query_planner_not_everything_routes_to_web():
    need = detect_web_need(
        query="what did we say earlier about token X",
        intent_family=IntentFamily.EXACT_RECALL,
        local_evidence_count=5,
    )
    assert need.needed is False
