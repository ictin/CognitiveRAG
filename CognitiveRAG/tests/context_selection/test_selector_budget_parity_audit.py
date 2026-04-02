from __future__ import annotations

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.lane_pruning import prune_lane_local
from CognitiveRAG.crag.context_selection.policies import get_policy
from CognitiveRAG.crag.context_selection.selector import select_context
from CognitiveRAG.crag.context_selection.utility import score_candidate


def _cand(
    cid: str,
    lane: RetrievalLane,
    mtype: MemoryType,
    *,
    text: str,
    tokens: int,
    lex: float = 0.6,
    sem: float = 0.6,
    rec: float = 0.5,
    fresh: float = 0.5,
    trust: float = 0.6,
    nov: float = 0.5,
    contradiction: float = 0.0,
    cluster: str | None = None,
    must: bool = False,
    compressible: bool = True,
) -> ContextCandidate:
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=mtype,
        text=text,
        tokens=tokens,
        provenance={"source": "audit"},
        lexical_score=lex,
        semantic_score=sem,
        recency_score=rec,
        freshness_score=fresh,
        trust_score=trust,
        novelty_score=nov,
        contradiction_risk=contradiction,
        cluster_id=cluster,
        must_include=must,
        compressible=compressible,
    )


def test_a_pruning_is_deterministic_and_more_than_dedup():
    huge = "token " * 420
    inp = [
        _cand("dup1", RetrievalLane.EPISODIC, MemoryType.EPISODIC_RAW, text="same text", tokens=12),
        _cand("dup2", RetrievalLane.EPISODIC, MemoryType.EPISODIC_RAW, text="same   text", tokens=12),
        _cand("big", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, text=huge, tokens=500, compressible=True),
        _cand("tiny1", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, text="tiny one", tokens=4, cluster="k"),
        _cand("tiny2", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, text="tiny two", tokens=4, cluster="k"),
    ]

    out1 = prune_lane_local(inp, max_candidate_tokens=100)
    out2 = prune_lane_local(inp, max_candidate_tokens=100)

    sig1 = [(c.id, c.text, c.tokens, c.lane.value, c.cluster_id) for c in out1]
    sig2 = [(c.id, c.text, c.tokens, c.lane.value, c.cluster_id) for c in out2]
    assert sig1 == sig2

    ids = [c.id for c in out1]
    assert "dup2" not in ids  # dedupe
    assert any(i.startswith("big#part") for i in ids)  # split
    assert "tiny1+tiny2" in ids  # adjacent tiny merge


def test_b_intent_conditioned_scoring_changes_ordering_for_same_pool():
    pool = [
        _cand("episodic", RetrievalLane.EPISODIC, MemoryType.EPISODIC_RAW, text="user said x", tokens=20),
        _cand("summary", RetrievalLane.SESSION_SUMMARY, MemoryType.SUMMARY, text="summary x", tokens=20),
        _cand("corpus", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, text="book x", tokens=20),
    ]

    exact_policy = get_policy(IntentFamily.EXACT_RECALL)
    mem_policy = get_policy(IntentFamily.MEMORY_SUMMARY)

    exact_ranked = sorted(
        pool,
        key=lambda c: score_candidate(c, [], exact_policy, IntentFamily.EXACT_RECALL, False),
        reverse=True,
    )
    mem_ranked = sorted(
        pool,
        key=lambda c: score_candidate(c, [], mem_policy, IntentFamily.MEMORY_SUMMARY, False),
        reverse=True,
    )

    assert exact_ranked[0].id == "episodic"
    assert mem_ranked[0].id == "summary"


def test_c_budget_and_must_include_can_beat_higher_utility_candidate():
    policy = get_policy(IntentFamily.EXACT_RECALL)

    must_low = _cand(
        "must_low",
        RetrievalLane.FRESH_TAIL,
        MemoryType.EPISODIC_RAW,
        text="mandatory pointer",
        tokens=20,
        lex=0.05,
        sem=0.05,
        rec=0.1,
        fresh=0.1,
        trust=0.2,
        nov=0.0,
        must=True,
    )
    hi_utility_big = _cand(
        "hi_big",
        RetrievalLane.EPISODIC,
        MemoryType.EPISODIC_RAW,
        text="very relevant but large",
        tokens=70,
        lex=0.95,
        sem=0.95,
        rec=0.9,
        fresh=0.8,
        trust=0.9,
        nov=0.7,
    )
    medium_fit = _cand(
        "medium_fit",
        RetrievalLane.EPISODIC,
        MemoryType.EPISODIC_RAW,
        text="smaller relevant",
        tokens=20,
        lex=0.5,
        sem=0.5,
        rec=0.5,
        fresh=0.5,
        trust=0.5,
        nov=0.5,
    )

    assert score_candidate(hi_utility_big, [], policy, IntentFamily.EXACT_RECALL, False) > score_candidate(
        must_low, [], policy, IntentFamily.EXACT_RECALL, False
    )

    selected, dropped, _ = select_context(
        candidates=[must_low, hi_utility_big, medium_fit],
        policy=policy,
        total_budget=90,
        reserved_tokens=30,
        intent_family=IntentFamily.EXACT_RECALL,
    )

    selected_ids = [c.id for c, _ in selected]
    dropped_by_id = {c.id: reason for c, reason in dropped}

    # must_include survives even with low utility
    assert "must_low" in selected_ids
    # high-utility candidate is blocked by budget after reservation+must_include
    assert dropped_by_id.get("hi_big") == "budget_or_lane_cap"


def test_d_lane_caps_and_minima_are_real_constraints():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    policy.lane_maxima[RetrievalLane.SEMANTIC.value] = 1
    policy.lane_minima[RetrievalLane.CORPUS.value] = 1

    cands = [
        _cand("sem1", RetrievalLane.SEMANTIC, MemoryType.EPISODIC_RAW, text="semantic A", tokens=20, lex=0.9, sem=0.9),
        _cand("sem2", RetrievalLane.SEMANTIC, MemoryType.EPISODIC_RAW, text="semantic B", tokens=20, lex=0.85, sem=0.85),
        _cand("cor_low", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, text="corpus low", tokens=20, lex=0.1, sem=0.1),
    ]

    selected, _, _ = select_context(
        candidates=cands,
        policy=policy,
        total_budget=60,
        reserved_tokens=0,
        intent_family=IntentFamily.INVESTIGATIVE,
    )
    ids = [c.id for c, _ in selected]

    assert "cor_low" in ids  # lane minimum preserved representation
    assert len([sid for sid in ids if sid in {"sem1", "sem2"}]) == 1  # lane cap enforced


def test_e_non_topk_selection_due_to_lane_minimum_and_budget():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    policy.lane_minima[RetrievalLane.FRESH_TAIL.value] = 1

    s1 = _cand("s1", RetrievalLane.SEMANTIC, MemoryType.EPISODIC_RAW, text="semantic 1", tokens=25, lex=0.95, sem=0.95)
    s2 = _cand("s2", RetrievalLane.SEMANTIC, MemoryType.EPISODIC_RAW, text="semantic 2", tokens=25, lex=0.9, sem=0.9)
    s3 = _cand("s3", RetrievalLane.SEMANTIC, MemoryType.EPISODIC_RAW, text="semantic 3", tokens=25, lex=0.85, sem=0.85)
    f1 = _cand(
        "f1",
        RetrievalLane.FRESH_TAIL,
        MemoryType.EPISODIC_RAW,
        text="fresh low utility",
        tokens=25,
        lex=0.05,
        sem=0.05,
        rec=0.1,
        fresh=0.1,
        trust=0.2,
        nov=0.0,
    )

    selected, _, _ = select_context(
        candidates=[s1, s2, s3, f1],
        policy=policy,
        total_budget=50,
        reserved_tokens=0,
        intent_family=IntentFamily.INVESTIGATIVE,
    )
    selected_ids = [c.id for c, _ in selected]

    # Under naive top-k by utility (budget 50, each 25), we'd pick s1+s2.
    # Real selector picks f1 first to satisfy lane minimum, then one semantic.
    assert "f1" in selected_ids
    assert len([sid for sid in selected_ids if sid in {"s1", "s2", "s3"}]) == 1
    assert not ({"s1", "s2"} <= set(selected_ids))
