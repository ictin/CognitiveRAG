from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.context_selection.policies import get_policy


def test_required_profiles_exist():
    for intent in [
        IntentFamily.EXACT_RECALL,
        IntentFamily.MEMORY_SUMMARY,
        IntentFamily.CORPUS_OVERVIEW,
        IntentFamily.PLANNING,
        IntentFamily.ARCHITECTURE_EXPLANATION,
        IntentFamily.INVESTIGATIVE,
    ]:
        p = get_policy(intent)
        assert p.intent_family == intent
        assert p.hard_reservation_tokens > 0
        assert p.front_anchor_budget >= 1
        assert p.back_anchor_budget >= 1


def test_memory_summary_requires_promoted_lane_minimum():
    p = get_policy(IntentFamily.MEMORY_SUMMARY)
    assert int(p.lane_minima.get("promoted", 0)) >= 1
