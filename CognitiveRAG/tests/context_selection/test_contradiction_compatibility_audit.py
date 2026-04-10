from __future__ import annotations

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.compatibility import (
    NLIBackedCompatibilityEngine,
    load_runtime_compatibility_engine_from_env,
    resolve_compatibility_engine,
)
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


def test_pairwise_compatibility_gate_excludes_conflicting_claim_values():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    policy.lane_maxima[RetrievalLane.CORPUS.value] = 10

    # Both below hard contradiction threshold; compatibility gate decides conflict.
    a = _cand("a", text="Feature flag is enabled for all users.", risk=0.2, lex=0.8, sem=0.8)
    b = _cand("b", text="Feature flag is NOT enabled for all users.", risk=0.25, lex=0.8, sem=0.8)
    a.provenance["claim_key"] = "feature_flag_all_users"
    a.provenance["claim_value"] = "enabled"
    b.provenance["claim_key"] = "feature_flag_all_users"
    b.provenance["claim_value"] = "disabled"

    selected, dropped, explanation = select_context(
        candidates=[a, b],
        policy=policy,
        total_budget=100,
        reserved_tokens=0,
        intent_family=IntentFamily.INVESTIGATIVE,
    )

    selected_ids = [c.id for c, _ in selected]
    assert selected_ids == ["a"]
    dropped_map = {c.id: reason for c, reason in dropped}
    assert dropped_map["b"] == "compatibility_conflict"

    selected_block_ids = {block.id for block in explanation.selected_blocks}
    assert selected_block_ids == {"a"}
    dropped_block = next(block for block in explanation.dropped_blocks if block.id == "b")
    assert dropped_block.reason == "compatibility_conflict"


def test_no_claim_conflict_keeps_below_threshold_candidates():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    policy.lane_maxima[RetrievalLane.CORPUS.value] = 10

    a = _cand("a", text="Feature flag is enabled for all users.", risk=0.2, lex=0.8, sem=0.8)
    b = _cand("b", text="Feature flag is enabled for all users.", risk=0.25, lex=0.8, sem=0.8)
    a.provenance["claim_key"] = "feature_flag_all_users"
    a.provenance["claim_value"] = "enabled"
    b.provenance["claim_key"] = "feature_flag_all_users"
    b.provenance["claim_value"] = "enabled"

    selected, dropped, _ = select_context(
        candidates=[a, b],
        policy=policy,
        total_budget=100,
        reserved_tokens=0,
        intent_family=IntentFamily.INVESTIGATIVE,
    )

    assert {c.id for c, _ in selected} == {"a", "b"}
    assert dropped == []


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


class _MockNLIAdapter:
    def __init__(self, conflict_pairs: set[tuple[str, str]] | None = None):
        self._pairs = conflict_pairs or set()

    def contradiction_score(self, left: str, right: str) -> float:
        key = (left.strip().lower(), right.strip().lower())
        return 0.92 if key in self._pairs else 0.12


class _AlwaysContradictAdapter:
    def contradiction_score(self, left: str, right: str) -> float:
        return 0.91


def test_engine_selection_defaults_and_nli_fallback():
    heuristic_engine = resolve_compatibility_engine(mode="heuristic")
    assert heuristic_engine.name == "heuristic"

    # NLI mode without adapter must stay safe and deterministic via heuristic fallback.
    nli_engine = resolve_compatibility_engine(mode="nli", adapter=None)
    assert nli_engine.name == "nli"
    assert isinstance(nli_engine, NLIBackedCompatibilityEngine)


def test_runtime_engine_loading_defaults_to_heuristic(monkeypatch):
    monkeypatch.delenv("CRAG_COMPAT_ENGINE", raising=False)
    monkeypatch.delenv("CRAG_COMPAT_NLI_BACKEND", raising=False)
    engine, state = load_runtime_compatibility_engine_from_env()
    assert engine.name == "heuristic"
    assert state.resolved_engine == "heuristic"
    assert state.fallback_active is False


def test_runtime_engine_loading_nli_unavailable_falls_back(monkeypatch):
    monkeypatch.setenv("CRAG_COMPAT_ENGINE", "nli")
    monkeypatch.setenv("CRAG_COMPAT_NLI_BACKEND", "transformers")
    monkeypatch.setattr(
        "CognitiveRAG.crag.context_selection.compatibility.TransformersNLIAdapter",
        lambda model_name=None: (_ for _ in ()).throw(ImportError("missing transformers")),
    )
    engine, state = load_runtime_compatibility_engine_from_env()
    assert engine.name == "nli"
    assert state.backend_available is False
    assert state.fallback_active is True
    assert state.reason.startswith("adapter_unavailable:")


def test_runtime_engine_loading_nli_available(monkeypatch):
    monkeypatch.setenv("CRAG_COMPAT_ENGINE", "nli")
    monkeypatch.setenv("CRAG_COMPAT_NLI_BACKEND", "transformers")
    monkeypatch.setattr(
        "CognitiveRAG.crag.context_selection.compatibility.TransformersNLIAdapter",
        lambda model_name=None: _AlwaysContradictAdapter(),
    )
    engine, state = load_runtime_compatibility_engine_from_env()
    assert engine.name == "nli"
    assert state.backend_available is True
    assert state.fallback_active is False
    assert state.reason == "adapter_loaded"


def test_selector_nli_mode_can_block_conflict_heuristic_would_allow():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    policy.lane_maxima[RetrievalLane.CORPUS.value] = 10

    # No claim-key conflict and little lexical overlap: heuristic permits both.
    a = _cand("a", text="alpha release checklist approved", risk=0.1, lex=0.8, sem=0.8)
    b = _cand("b", text="beta rollback process denied", risk=0.1, lex=0.8, sem=0.8)

    selected_h, dropped_h, _ = select_context(
        candidates=[a, b],
        policy=policy,
        total_budget=100,
        reserved_tokens=0,
        intent_family=IntentFamily.INVESTIGATIVE,
    )
    assert {c.id for c, _ in selected_h} == {"a", "b"}
    assert dropped_h == []

    adapter = _MockNLIAdapter(conflict_pairs={(a.text.lower(), b.text.lower())})
    nli_engine = resolve_compatibility_engine(mode="nli", adapter=adapter, contradiction_threshold=0.75)
    selected_nli, dropped_nli, _ = select_context(
        candidates=[a, b],
        policy=policy,
        total_budget=100,
        reserved_tokens=0,
        intent_family=IntentFamily.INVESTIGATIVE,
        compatibility_engine=nli_engine,
    )
    assert [c.id for c, _ in selected_nli] == ["a"]
    assert {c.id: reason for c, reason in dropped_nli}["b"] == "compatibility_conflict_nli"


def test_selector_nli_mode_is_deterministic_across_repeated_runs():
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    policy.lane_maxima[RetrievalLane.CORPUS.value] = 10

    a = _cand("a", text="service A availability confirmed", risk=0.1, lex=0.8, sem=0.8)
    b = _cand("b", text="service A availability denied", risk=0.1, lex=0.8, sem=0.8)
    adapter = _MockNLIAdapter(conflict_pairs={(a.text.lower(), b.text.lower())})
    nli_engine = resolve_compatibility_engine(mode="nli", adapter=adapter, contradiction_threshold=0.75)

    runs = []
    for _ in range(3):
        selected, dropped, _ = select_context(
            candidates=[a, b],
            policy=policy,
            total_budget=100,
            reserved_tokens=0,
            intent_family=IntentFamily.INVESTIGATIVE,
            compatibility_engine=nli_engine,
        )
        runs.append(([c.id for c, _ in selected], [(c.id, reason) for c, reason in dropped]))

    assert runs[0] == runs[1] == runs[2]
