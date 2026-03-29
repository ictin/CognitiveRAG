from CognitiveRAG.crag.cognition.controller import CognitiveController
from CognitiveRAG.crag.contracts.enums import IntentFamily


def test_mock_controller_mode_is_deterministic_without_llm_dependency():
    controller = CognitiveController.mock()

    p1 = controller.build_plan(query='Vague question about tradeoffs and unknown unknowns')
    p2 = controller.build_plan(query='Vague question about tradeoffs and unknown unknowns')

    assert p1.plan_version == 'm9-mock'
    assert p1.risk_mode == 'mock'
    assert p1.model_dump(mode='json') == p2.model_dump(mode='json')
    assert 'mock-controller-mode' in p1.notes


def test_mock_controller_respects_intent_hint():
    controller = CognitiveController.mock()
    plan = controller.build_plan(
        query='Any query text',
        hinted_intent=IntentFamily.ARCHITECTURE_EXPLANATION,
    )

    assert plan.intent_family == IntentFamily.ARCHITECTURE_EXPLANATION
    assert plan.discovery_mode.value == 'off'
