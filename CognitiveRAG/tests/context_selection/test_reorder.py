from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.policies import get_policy
from CognitiveRAG.crag.context_selection.reorder import reorder_for_prompt


def test_reorder_front_back_anchor_deterministic():
    policy = get_policy(IntentFamily.MEMORY_SUMMARY)
    c1 = ContextCandidate(id="a", lane=RetrievalLane.FRESH_TAIL, memory_type=MemoryType.EPISODIC_RAW, text="a", tokens=5, provenance={}, must_include=True)
    c2 = ContextCandidate(id="b", lane=RetrievalLane.SESSION_SUMMARY, memory_type=MemoryType.SUMMARY, text="b", tokens=5, provenance={})
    c3 = ContextCandidate(id="c", lane=RetrievalLane.PROMOTED, memory_type=MemoryType.PROMOTED_FACT, text="c", tokens=5, provenance={})

    ordered = reorder_for_prompt([(c2, 0.2), (c3, 0.5), (c1, 0.1)], policy)
    assert ordered[0][0].id == "a"
    assert [c.id for c, _ in ordered] == ["a", "c", "b"]
