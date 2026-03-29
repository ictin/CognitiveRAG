from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.explanation import build_explanation


def test_explanation_contains_lane_totals_selected_and_dropped():
    c1 = ContextCandidate(id="x", lane=RetrievalLane.CORPUS, memory_type=MemoryType.CORPUS_CHUNK, text="hello", tokens=12, provenance={})
    c2 = ContextCandidate(id="y", lane=RetrievalLane.EPISODIC, memory_type=MemoryType.EPISODIC_RAW, text="world", tokens=8, provenance={})

    ex = build_explanation(
        intent_family=IntentFamily.CORPUS_OVERVIEW,
        total_budget=200,
        reserved_tokens=40,
        selected=[(c1, 1.2)],
        dropped=[(c2, "budget_or_lane_cap")],
    )

    assert ex.lane_totals["corpus"] == 12
    assert ex.selected_blocks[0].id == "x"
    assert ex.dropped_blocks[0].id == "y"
    assert ex.total_budget == 200
    assert ex.reserved_tokens == 40
