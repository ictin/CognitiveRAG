from CognitiveRAG.crag.cognition.controller import CognitiveController
from CognitiveRAG.crag.contracts.enums import DiscoveryMode, IntentFamily, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import DiscoveryPlan


def test_vague_query_produces_multi_lane_investigative_plan():
    controller = CognitiveController()
    plan = controller.build_plan(query='Help me figure out what else matters for this migration risk')

    assert plan.intent_family == IntentFamily.INVESTIGATIVE
    assert len(plan.expected_lanes) >= 4
    assert RetrievalLane.SEMANTIC in plan.expected_lanes
    assert RetrievalLane.LEXICAL in plan.expected_lanes
    assert plan.bounded is True


def test_architecture_query_stays_bounded_and_non_discovery():
    controller = CognitiveController()
    plan = controller.build_plan(query='How is your memory organized and where did this answer come from?')

    assert plan.intent_family == IntentFamily.ARCHITECTURE_EXPLANATION
    assert plan.discovery_mode == DiscoveryMode.OFF
    assert plan.web_sensitive is False
    assert len(plan.novelty_probes) == 0


def test_recent_query_triggers_web_sensitive_plan():
    controller = CognitiveController()
    plan = controller.build_plan(query='What are the latest updates today and can you verify with sources?')

    assert plan.web_sensitive is True
    assert RetrievalLane.WEB in plan.expected_lanes
    assert plan.discovery_mode in {DiscoveryMode.PASSIVE, DiscoveryMode.ACTIVE}


def test_discovery_plan_schema_validation_is_stable():
    controller = CognitiveController()
    plan = controller.build_plan(query='Compare current options and verify contradictions')

    parsed = DiscoveryPlan.model_validate(plan.model_dump(mode='json'))
    assert parsed.intent_family == plan.intent_family
    assert len(parsed.query_variants) <= 6
    assert len(parsed.adjacent_topics) <= 8
