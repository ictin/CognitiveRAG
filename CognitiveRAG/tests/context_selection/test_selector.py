from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.policies import get_policy
from CognitiveRAG.crag.context_selection.selector import select_context


def _cand(cid: str, lane: RetrievalLane, mtype: MemoryType, text: str, tokens: int, cluster: str | None = None, must=False):
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=mtype,
        text=text,
        tokens=tokens,
        provenance={"x": 1},
        lexical_score=0.8,
        semantic_score=0.7,
        recency_score=0.5,
        freshness_score=0.6,
        trust_score=0.7,
        novelty_score=0.5,
        contradiction_risk=0.0,
        cluster_id=cluster,
        must_include=must,
    )


def test_selector_honors_reservation_caps_and_is_deterministic():
    policy = get_policy(IntentFamily.EXACT_RECALL)
    policy.lane_maxima[RetrievalLane.EPISODIC.value] = 1

    candidates = [
        _cand("must", RetrievalLane.FRESH_TAIL, MemoryType.EPISODIC_RAW, "tail", 20, must=True),
        _cand("e1", RetrievalLane.EPISODIC, MemoryType.EPISODIC_RAW, "alpha", 30),
        _cand("e2", RetrievalLane.EPISODIC, MemoryType.EPISODIC_RAW, "alpha beta", 30),
        _cand("s1", RetrievalLane.SESSION_SUMMARY, MemoryType.SUMMARY, "sum", 20),
    ]

    selected, dropped, explanation = select_context(
        candidates=candidates,
        policy=policy,
        total_budget=120,
        reserved_tokens=40,
        intent_family=IntentFamily.EXACT_RECALL,
    )

    selected_ids = [c.id for c, _ in selected]
    assert "must" in selected_ids
    assert len([sid for sid in selected_ids if sid in {"e1", "e2"}]) <= 1
    assert explanation.total_budget == 120
    assert explanation.reserved_tokens == 40


def test_cluster_bonus_changes_outcome():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    c1 = _cand("c1", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, "same tokens", 30, cluster="A")
    c2 = _cand("c2", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, "same tokens", 30, cluster="A")
    c3 = _cand("c3", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, "same tokens", 30, cluster="B")

    selected, _, _ = select_context(
        candidates=[c1, c2, c3],
        policy=policy,
        total_budget=120,
        reserved_tokens=0,
        intent_family=IntentFamily.INVESTIGATIVE,
    )
    ids = [c.id for c, _ in selected]
    assert "c1" in ids
    assert "c3" in ids
