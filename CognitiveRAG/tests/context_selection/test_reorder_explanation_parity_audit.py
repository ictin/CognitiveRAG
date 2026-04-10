from __future__ import annotations

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.policies import get_policy
from CognitiveRAG.crag.context_selection.selector import select_context


def _cand(
    cid: str,
    lane: RetrievalLane,
    mtype: MemoryType,
    *,
    utility_hint: float,
    tokens: int = 20,
    must: bool = False,
) -> ContextCandidate:
    # utility_hint is mapped into lexical/semantic for deterministic ranking.
    return ContextCandidate(
        id=cid,
        lane=lane,
        memory_type=mtype,
        text=f"{cid} text",
        tokens=tokens,
        provenance={"source": "b4-audit"},
        lexical_score=utility_hint,
        semantic_score=utility_hint,
        recency_score=0.5,
        freshness_score=0.5,
        trust_score=0.6,
        novelty_score=0.5,
        contradiction_risk=0.0,
        cluster_id=None,
        must_include=must,
        compressible=True,
    )


def test_a_reorder_front_back_anchor_changes_final_order_not_input_order():
    policy = get_policy(IntentFamily.MEMORY_SUMMARY)
    policy.front_anchor_budget = 1
    policy.back_anchor_budget = 1

    critical = _cand("critical", RetrievalLane.FRESH_TAIL, MemoryType.EPISODIC_RAW, utility_hint=0.1, must=True)
    ordinary = _cand("ordinary", RetrievalLane.PROMOTED, MemoryType.PROMOTED_FACT, utility_hint=0.9)
    background = _cand("background", RetrievalLane.SESSION_SUMMARY, MemoryType.SUMMARY, utility_hint=0.2)

    # Deliberately shuffled input order.
    selected, _, explanation = select_context(
        candidates=[ordinary, background, critical],
        policy=policy,
        total_budget=200,
        reserved_tokens=0,
        intent_family=IntentFamily.MEMORY_SUMMARY,
    )

    final_ids = [c.id for c, _ in selected]
    assert final_ids != ["ordinary", "background", "critical"]
    assert final_ids == ["critical", "ordinary", "background"]
    assert explanation.reorder_strategy == "front_back_anchor"


def test_b_front_back_anchor_behavior_is_real_and_stable():
    policy = get_policy(IntentFamily.MEMORY_SUMMARY)
    policy.front_anchor_budget = 2
    policy.back_anchor_budget = 1

    c1 = _cand("must", RetrievalLane.FRESH_TAIL, MemoryType.EPISODIC_RAW, utility_hint=0.1, must=True)
    c2 = _cand("high", RetrievalLane.PROMOTED, MemoryType.PROMOTED_FACT, utility_hint=0.95)
    c3 = _cand("mid", RetrievalLane.SESSION_SUMMARY, MemoryType.SUMMARY, utility_hint=0.5)
    c4 = _cand("low", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, utility_hint=0.2)

    runs = []
    for _ in range(3):
        selected, _, _ = select_context(
            candidates=[c3, c2, c4, c1],
            policy=policy,
            total_budget=300,
            reserved_tokens=0,
            intent_family=IntentFamily.MEMORY_SUMMARY,
        )
        runs.append([c.id for c, _ in selected])

    assert runs[0] == runs[1] == runs[2]
    # must_include is anchored first; one lower-priority item ends up in back bucket.
    assert runs[0][0] == "must"
    assert runs[0][-1] in {"mid", "low"}


def test_c_explanation_completeness_and_budget_fields_present():
    policy = get_policy(IntentFamily.CORPUS_OVERVIEW)
    keep = _cand("keep", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, utility_hint=0.8, tokens=20)
    drop = _cand("drop", RetrievalLane.EPISODIC, MemoryType.EPISODIC_RAW, utility_hint=0.1, tokens=80)

    selected, dropped, explanation = select_context(
        candidates=[keep, drop],
        policy=policy,
        total_budget=60,
        reserved_tokens=20,
        intent_family=IntentFamily.CORPUS_OVERVIEW,
    )

    assert selected and dropped
    assert explanation.selected_blocks
    assert explanation.dropped_blocks
    assert explanation.total_budget == 60
    assert explanation.reserved_tokens == 20
    assert explanation.lane_totals
    assert explanation.reorder_strategy == "front_back_anchor"

    d0 = explanation.dropped_blocks[0]
    assert d0.reason in {"budget_or_lane_cap", "contradiction_risk"}


def test_d_explanation_includes_final_order_index_but_not_anchor_membership_labels():
    policy = get_policy(IntentFamily.MEMORY_SUMMARY)
    policy.front_anchor_budget = 1
    policy.back_anchor_budget = 1

    a = _cand("a", RetrievalLane.FRESH_TAIL, MemoryType.EPISODIC_RAW, utility_hint=0.1, must=True)
    b = _cand("b", RetrievalLane.PROMOTED, MemoryType.PROMOTED_FACT, utility_hint=0.8)
    c = _cand("c", RetrievalLane.SESSION_SUMMARY, MemoryType.SUMMARY, utility_hint=0.2)

    selected, _, explanation = select_context(
        candidates=[b, c, a],
        policy=policy,
        total_budget=200,
        reserved_tokens=0,
        intent_family=IntentFamily.MEMORY_SUMMARY,
    )

    selected_ids = [x.id for x, _ in selected]
    explained_ids = [x.id for x in explanation.selected_blocks]
    order_idx = [x.order_index for x in explanation.selected_blocks]

    assert explained_ids == selected_ids
    assert order_idx == list(range(len(explanation.selected_blocks)))

    # Current artifact exposes strategy and final order index, but not explicit anchor bucket labels.
    as_dict = explanation.model_dump()
    assert "reorder_strategy" in as_dict
    assert "anchor_bucket" not in as_dict["selected_blocks"][0]


def test_e_selected_dropped_and_why_fields_are_stable_across_repeated_runs():
    policy = get_policy(IntentFamily.CORPUS_OVERVIEW)

    keep = _cand("keep", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, utility_hint=0.9, tokens=20)
    risky = _cand("risky", RetrievalLane.CORPUS, MemoryType.CORPUS_CHUNK, utility_hint=0.95, tokens=20)
    risky.contradiction_risk = 0.99
    budget_drop = _cand("budget_drop", RetrievalLane.EPISODIC, MemoryType.EPISODIC_RAW, utility_hint=0.5, tokens=120)

    runs: list[dict] = []
    for _ in range(3):
        selected, dropped, explanation = select_context(
            candidates=[budget_drop, keep, risky],
            policy=policy,
            total_budget=90,
            reserved_tokens=20,
            intent_family=IntentFamily.CORPUS_OVERVIEW,
        )
        runs.append(
            {
                "selected_ids": [c.id for c, _ in selected],
                "dropped": [(c.id, reason) for c, reason in dropped],
                "selected_blocks": [(b.id, b.order_index) for b in explanation.selected_blocks],
                "dropped_blocks": [(b.id, b.reason) for b in explanation.dropped_blocks],
            }
        )

    assert runs[0] == runs[1] == runs[2]
