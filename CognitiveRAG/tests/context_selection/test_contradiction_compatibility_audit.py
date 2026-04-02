from __future__ import annotations

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.policies import get_policy
from CognitiveRAG.crag.context_selection.selector import select_context
from CognitiveRAG.crag.context_selection.utility import score_candidate


def _cand(
    cid: str,
    *,
    text: str,
    risk: float,
    lex: float = 0.8,
    sem: float = 0.8,
    lane: RetrievalLane = RetrievalLane.CORPUS,
    mtype: MemoryType = MemoryType.CORPUS_CHUNK,
) -> ContextCandidate:
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=mtype,
        text=text,
        tokens=20,
        provenance={"source": "b3-audit"},
        lexical_score=lex,
        semantic_score=sem,
        recency_score=0.5,
        freshness_score=0.6,
        trust_score=0.7,
        novelty_score=0.4,
        contradiction_risk=risk,
        cluster_id="cluster-x",
        must_include=False,
        compressible=True,
    )


def test_contradiction_threshold_is_hard_drop_and_reason_is_explicit():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    high_risk = _cand("high", text="X is true", risk=0.99, lex=0.99, sem=0.99)
    low_risk = _cand("low", text="supporting evidence", risk=0.1, lex=0.4, sem=0.4)

    selected, dropped, explanation = select_context(
        candidates=[high_risk, low_risk],
        policy=policy,
        total_budget=80,
        reserved_tokens=0,
        intent_family=IntentFamily.INVESTIGATIVE,
    )

    assert [c.id for c, _ in selected] == ["low"]
    dropped_map = {c.id: reason for c, reason in dropped}
    assert dropped_map["high"] == "contradiction_risk"
    dropped_block = next(db for db in explanation.dropped_blocks if db.id == "high")
    assert dropped_block.reason == "contradiction_risk"
    assert dropped_block.contradiction_risk == 0.99


def test_below_threshold_contradiction_is_soft_penalty_not_pairwise_exclusion():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    policy.lane_maxima[RetrievalLane.CORPUS.value] = 10

    # Intentionally contradictory texts but both below hard threshold.
    a = _cand("a", text="Feature flag is enabled for all users.", risk=0.2, lex=0.8, sem=0.8)
    b = _cand("b", text="Feature flag is NOT enabled for all users.", risk=0.25, lex=0.8, sem=0.8)

    selected, dropped, explanation = select_context(
        candidates=[a, b],
        policy=policy,
        total_budget=100,
        reserved_tokens=0,
        intent_family=IntentFamily.INVESTIGATIVE,
    )

    selected_ids = [c.id for c, _ in selected]
    # No pairwise compatibility logic exists, so both survive.
    assert set(selected_ids) == {"a", "b"}
    assert dropped == []

    selected_block_ids = {b.id for b in explanation.selected_blocks}
    assert selected_block_ids == {"a", "b"}
    assert all(b.contradiction_risk > 0 for b in explanation.selected_blocks)


def test_contradiction_penalty_changes_utility_heuristically():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    low = _cand("low", text="same claim", risk=0.1, lex=0.7, sem=0.7)
    high = _cand("high", text="same claim", risk=0.8, lex=0.7, sem=0.7)

    low_score = score_candidate(low, [], policy, IntentFamily.INVESTIGATIVE, False)
    high_score = score_candidate(high, [], policy, IntentFamily.INVESTIGATIVE, False)

    assert low_score > high_score


def test_conflict_behavior_is_deterministic_across_repeated_runs():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    cands = [
        _cand("a", text="Claim A true", risk=0.2),
        _cand("b", text="Claim A false", risk=0.2),
        _cand("c", text="neutral note", risk=0.0),
    ]

    runs = []
    for _ in range(3):
        selected, dropped, _ = select_context(
            candidates=cands,
            policy=policy,
            total_budget=60,
            reserved_tokens=0,
            intent_family=IntentFamily.INVESTIGATIVE,
        )
        runs.append(([c.id for c, _ in selected], [(c.id, reason) for c, reason in dropped]))

    assert runs[0] == runs[1] == runs[2]
